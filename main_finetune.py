"""
微调主程序
该文件是MetasurfaceVIT项目的微调主程序，用于：
1. 使用预训练的Vision Transformer模型进行微调
2. 从Jones矩阵预测超表面结构参数
3. 支持单GPU和分布式多GPU训练
4. 支持混合精度训练（Nvidia Apex或PyTorch AMP）
5. 支持评估模式以预测新的Jones矩阵对应的结构参数

主要功能：
- 微调：加载预训练权重并在微调数据集上训练
- 评估：使用训练好的模型预测结构参数
- 吞吐量测试：测试模型的推理速度
"""

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.utils import AverageMeter

from config import get_config
from model import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from apex import amp
from torch.cuda.amp import autocast, GradScaler


def parse_option():
    """
    解析命令行参数
    
    该函数解析微调和评估模式下的所有命令行参数，包括：
    - 训练超参数（epoch、学习率、批次大小等）
    - 数据配置（数据路径、文件夹名称等）
    - 预训练模型路径
    - 混合精度训练配置
    - 分布式训练配置
    - 评估模式配置
    
    返回:
        args: 解析后的命令行参数对象
        config: 根据参数构建的配置对象
    """
    parser = argparse.ArgumentParser('MetaVIT 微调与评估', add_help=False)
    
    # 通用配置参数
    parser.add_argument(
        "--opts",
        help="通过添加'KEY VALUE'对来修改配置选项",
        default=None,
        nargs='+',
    )

    # 训练超参数
    parser.add_argument('--epoch', type=int, help="总训练轮数")
    parser.add_argument('--batch-size', type=int, help="单GPU的批次大小")
    
    # 数据配置
    parser.add_argument('--data_path', type=str, help='微调训练数据路径（data_path + data_folder_name）')
    parser.add_argument('--data_folder_name', type=str, help='数据集路径')
    
    # 模型恢复和预训练
    parser.add_argument('--resume', help='从检查点恢复训练')
    
    # 训练优化参数
    parser.add_argument('--accumulation-steps', type=int, help="梯度累积步数")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="是否使用梯度检查点以节省内存")
    
    # 混合精度训练参数
    parser.add_argument('--amp_type', type=str, default='apex',
                        help="自动混合精度类型：apex为nvidia apex.amp，pytorch为pytorch.amp")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='混合精度优化级别，O0表示不使用amp')
    
    # 输出和实验配置
    parser.add_argument('--output', default='finetune_output', type=str, metavar='PATH',
                        help='输出文件夹根路径，完整路径为<output>/<model_name>/<tag>（默认：finetune_output）')
    parser.add_argument('--pretrained_output', default='output', type=str, metavar='PATH',
                        help='预训练模型文件的根路径')
    parser.add_argument('--eval_output', type=str,
                        help='放置评估结果的路径（数据文件而非模型文件）')
    parser.add_argument('--tag', help='实验标签')
    
    # 模式选择
    parser.add_argument('--eval', action='store_true', help='仅执行评估')
    parser.add_argument('--throughput', action='store_true', help='仅测试吞吐量')
    
    # 分布式训练参数
    parser.add_argument("--local_rank", default=0, type=int, help='分布式数据并行的本地进程编号')
    
    # 评估相关参数
    parser.add_argument('--recon_type', type=int, help="超表面设计期间确定的设计类型1-4")
    parser.add_argument('--treatment', help="手动输入要用于参数预测的时间戳（例如2024-10-14）")

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    """
    主微调或评估函数
    
    该函数执行两种模式之一：
    1. 微调模式：加载预训练模型并在结构参数预测任务上微调
    2. 评估模式：使用训练好的模型预测重建Jones矩阵对应的结构参数
    
    参数:
        config: 配置对象，包含所有训练和模型参数
    
    工作流程：
        - 构建训练和验证数据加载器
        - 创建模型（Vision Transformer用于回归任务）
        - 设置优化器、损失函数和学习率调度器
        - 配置分布式训练（如果使用多GPU）
        - 加载预训练权重
        - 执行微调或评估
    """
    # 构建数据加载器
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config, logger, type='finetune')
    
    # 创建模型（微调模式：用于结构参数回归）
    logger.info(f"创建模型：{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=False)
    model.cuda()
    logger.info(str(model))

    # 构建优化器（使用层级学习率衰减）
    optimizer = build_optimizer(config, model, logger, is_pretrain=False)
    
    # 配置混合精度训练
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    
    # 配置分布式数据并行
    if torch.cuda.device_count() != 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # 计算并记录模型参数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"参数数量：{n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"GFLOPs数量：{flops / 1e9}")

    # 构建学习率调度器
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # 定义损失函数（平滑L1损失，对异常值更鲁棒）
    criterion = torch.nn.SmoothL1Loss()

    # 自动恢复训练
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"自动恢复将检查点从{config.MODEL.RESUME}更改为{resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'从{resume_file}自动恢复')
        else:
            logger.info(f'在{config.OUTPUT}中未找到检查点，忽略自动恢复')

    # 加载检查点或预训练权重
    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        
        # 评估模式：预测结构参数
        if config.EVAL_MODE:
            dataset_pred, data_loader_pred = build_loader(config, logger, type='predict')
            predict_para, loss = validate(config, data_loader_pred, model)
            logger.info(f"网络在{len(dataset_pred)}个重建图像上的损失：{loss:.1f}%")
            
            # 保存预测的参数（路径在config.py中硬编码）
            np.savetxt(config.PREDICT_PARA_PATH + 'type_' + str(config.RECON_TYPE) + '_'
                       + config.TREATMENT + '.txt', predict_para, fmt='%.3f')
            return
        else:
            loss = validate(config, data_loader_val, model)
            logger.info(f"LOSS of the network on the {len(dataset_val)} test images: {loss:.1f}%")

    else:  # starts at the beginning from pretrained mode;
        resume_file = auto_resume_helper(config.PRETRAINED_OUTPUT, logger)
        if resume_file:
            config.defrost()
            config.MODEL.PRETRAIN = resume_file
            config.freeze()
            logger.info(f'start from pretrained: {resume_file}')
        else:
            raise ValueError("can't find relevant files (neither fine-tune resume nor pretrained file)")
        load_pretrained(config, resume_file, model_without_ddp, logger)

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if torch.cuda.device_count() != 1:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler)
        if torch.cuda.device_count() != 1:
            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model_without_ddp, optimizer, lr_scheduler, logger)
        else:
            if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
                save_checkpoint(config, epoch, model_without_ddp, optimizer, lr_scheduler, logger)

        loss = validate(config, data_loader_val, model)
        logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss:.1f}%")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad()
    
    logger.info(f'Current learning rate for different parameter groups: {[it["lr"] for it in optimizer.param_groups]}')

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        outputs = model(samples)
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS
        # pytorch amp
        if config.AMP_TYPE == 'pytorch':
            scaler.scale(loss).backward()
            grad_norm = handle_gradient(model.parameters())
        # nvidia apex amp
        elif config.AMP_OPT_LEVEL != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            grad_norm = handle_gradient(amp.master_params(optimizer))
        # no amp
        else:
            loss.backward()
            grad_norm = handle_gradient(model.parameters())

        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            if config.AMP_TYPE == "pytorch":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[-1]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def handle_gradient(parameters):
    if config.TRAIN.CLIP_GRAD:
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.TRAIN.CLIP_GRAD)
    else:
        grad_norm = get_grad_norm(parameters)
    return grad_norm


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.SmoothL1Loss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    amp = config.DATA.BATCH_SIZE
    if config.EVAL_MODE:
        para_pred = np.zeros((len(data_loader) * amp, 6))
    end = time.time()
    # images for JM ; target for para
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(images)
        # print('images size', images.shape)  # 128 1 20 6
        # print('target size', target.shape)  # 128 6
        # print('output size', output.shape)  # 128 6
        # measure accuracy and record loss
        loss = criterion(output, target)
        if config.EVAL_MODE:
            if images.size(0) == amp:
                para_pred[idx * amp: (idx + 1) * amp, :] = output.cpu().numpy()
            else:
                para_pred[idx * amp: idx * amp + images.size(0), :] = output.cpu().numpy()

        if torch.cuda.device_count() != 1:
            loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB')

    if config.EVAL_MODE:
        return para_pred, loss_meter.avg
    else:
        return loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_TYPE == 'pytorch':
        scaler = GradScaler()
    elif config.AMP_TYPE == 'apex' and config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if torch.cuda.device_count() != 1:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
        else:
            rank = -1
            world_size = -1

        torch.cuda.set_device(config.LOCAL_RANK)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.distributed.barrier()
        seed = config.SEED + dist.get_rank()
    else:
        seed = config.SEED

    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    factor = 1 if torch.cuda.device_count() == 1 else dist.get_world_size()
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * factor / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * factor / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * factor / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    rank_indicator = 0 if torch.cuda.device_count() == 1 else dist.get_rank()
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=rank_indicator, name=f"{config.MODEL.NAME}")

    if rank_indicator == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
