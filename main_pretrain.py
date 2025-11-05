"""
预训练主程序
该文件是MetasurfaceVIT项目的预训练主程序，用于：
1. 使用SimMIM（掩码图像建模）方法对Vision Transformer进行自监督预训练
2. 支持多种掩码策略（5种不同的掩码类型）
3. 支持单GPU和分布式多GPU训练
4. 支持混合精度训练（Nvidia Apex或PyTorch AMP）
5. 支持Jones矩阵重建模式

主要功能：
- 预训练：从头开始或继续训练Vision Transformer模型
- 重建：使用训练好的模型重建设计的Jones矩阵
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
from config import get_config, get_params_from_preprocess

from model import build_model
# 对于重建模式，数据加载器使用独立文件：data_recon.py
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper
from apex import amp
from torch.cuda.amp import autocast, GradScaler
# 注意：提供选择使用已弃用的apex.amp或torch.cuda.amp
# DeprecatedFeatureWarning: apex.amp已弃用，将在2023年2月底移除
# 建议使用 [PyTorch AMP](https://pytorch.org/docs/stable/amp.html)


def parse_option():
    """
    解析命令行参数
    
    该函数解析预训练和重建模式下的所有命令行参数，包括：
    - 训练超参数（epoch、学习率、批次大小等）
    - 掩码类型（5种不同的掩码策略）
    - 数据配置（数据大小、开始索引等）
    - 混合精度训练配置
    - 分布式训练配置
    - 重建模式配置
    
    返回:
        args: 解析后的命令行参数对象
        config: 根据参数构建的配置对象
    """
    parser = argparse.ArgumentParser('MetaVIT 掩码预训练', add_help=False)
    
    # 通用配置参数
    parser.add_argument(
        "--opts",
        help="通过添加'KEY VALUE'对来修改配置选项",
        default=None,
        nargs='+',
    )

    # 训练超参数
    parser.add_argument('--epoch', type=int, help="总训练轮数")
    parser.add_argument('--mask_type', type=int,
                        help="预训练的掩码类型：\n"
                             "0 - 随机选择1-5类型之一\n"
                             "1 - 掩码n-1个波长通道，仅保留一个波长的完整Jones矩阵\n"
                             "2 - 保留所有振幅分量（所有波长），仅保留一个波长的相位分量\n"
                             "3 - 使用与类型1相同的掩码机制，但仅保留11极化分量，掩码12和22分量\n"
                             "4 - 使用与类型2相同的掩码机制，但仅保留11极化分量，掩码12和22分量\n"
                             "5 - 掩码所有波长的12和22极化分量，保留所有波长的11极化分量")
    parser.add_argument('--data_size', type=int,
                        help='数据大小倍数：1表示基础量，2表示基础量的2倍，3表示基础量的3倍')
    parser.add_argument('--data_start', type=int,
                        help='数据起始索引：如果data_start=1且data_size=2，将加载training_data_1和training_data_2；'
                             '如果data_start=5且data_size=1，将加载training_data_5')

    # 学习率相关参数
    parser.add_argument('--base_lr', type=float, help='基础学习率')
    parser.add_argument('--warmup_lr', type=float, help='预热学习率')
    parser.add_argument('--min_lr', type=float, help='最小学习率')
    
    # 批次和数据参数
    parser.add_argument('--batch_size', type=int, help="单GPU的批次大小")
    parser.add_argument('--resume', help='从检查点恢复训练的路径')
    parser.add_argument('--data_folder_name', help='可以指定数据文件夹名称如training_data_2，否则将使用最新的')

    # 训练优化参数
    parser.add_argument('--accumulation_steps', type=int, help="梯度累积步数")
    parser.add_argument('--use_checkpoint', action='store_true',
                        help="是否使用梯度检查点以节省内存")
    
    # 混合精度训练参数
    parser.add_argument('--amp_type', type=str, default='apex',
                        help="自动混合精度类型：apex为nvidia apex.amp，pytorch为pytorch.amp")
    parser.add_argument('--amp_opt_level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='混合精度优化级别，O0表示不使用amp')
    
    # 输出和实验配置
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='输出文件夹根路径，完整路径为<output>/<model_name>/<tag>（默认：output）')
    parser.add_argument('--tag', help='实验标签')
    
    # 分布式训练参数
    parser.add_argument("--local_rank", type=int, help='分布式数据并行的本地进程编号')
    
    # 重建模式参数
    parser.add_argument("--recon", action='store_true', help="是否运行重建模式")
    parser.add_argument('--recon_path', help='设计的Jones矩阵和掩码的路径，用于重建')
    parser.add_argument('--recon_type', type=int, help="超表面设计期间确定的设计类型1-4")
    parser.add_argument('--treatment', default=None,
                        help="个性化字符串以匹配设计的Jones矩阵和掩码，也将用于命名后续生成的数据。"
                             "应为日期格式如'2024-08-20'。保持None时，代码将自动查找最新日期标记的数据。")
    
    args = parser.parse_args()
    config = get_config(args)

    return args, config


def main(config):
    """
    主训练或重建函数
    
    该函数执行两种模式之一：
    1. 预训练模式：从头或从检查点开始训练Vision Transformer模型
    2. 重建模式：使用训练好的模型重建设计的Jones矩阵
    
    参数:
        config: 配置对象，包含所有训练和模型参数
    
    工作流程：
        - 构建数据加载器（根据模式选择预训练或重建数据）
        - 创建模型（SimMIM包装的Vision Transformer）
        - 设置优化器和学习率调度器
        - 配置分布式训练（如果使用多GPU）
        - 执行训练循环或重建
    """
    # 根据模式构建数据加载器
    if config.RECON_MODE:
        # 重建模式：加载设计的Jones矩阵用于重建
        data_loader = build_loader(config, logger, "reconstruct")
    else:
        # 预训练模式：加载训练数据
        data_loader = build_loader(config, logger, "pre_trained")

    # 创建模型
    logger.info(f"创建模型：{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=True)
    model.cuda()
    logger.info(str(model))

    # 构建优化器
    optimizer = build_optimizer(config, model, logger, is_pretrain=True)
    
    # 配置混合精度训练
    if config.AMP_TYPE == 'apex' and config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    
    # 配置分布式数据并行（DDP）
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
        flops = model_without_ddp.flops()  # 每秒浮点运算次数
        logger.info(f"GFLOPs数量：{flops / 1e9}")

    # 构建学习率调度器
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader))

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

    # 重建模式：加载检查点并执行重建
    if config.RECON_MODE:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        loss, JM_recon = validate(config, data_loader, model)
        # 保存重建的Jones矩阵
        os.makedirs(config.RECON_PATH + '/reconJMs/', exist_ok=True)
        np.savetxt(config.RECON_PATH + '/reconJMs/type_' + str(config.RECON_TYPE) + '_' + config.TREATMENT + '.txt',
                   JM_recon, fmt='%.3f')
        return

    # 预训练模式：如果指定了检查点，则加载
    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    # 开始训练
    logger.info("开始训练")
    start_time = time.time()
    daily_time = time.time()
    
    # 计算每轮数据的epoch数
    round_len = config.TRAIN.EPOCHS // config.DATA.DATA_SIZE
    
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # 更新数据路径（逐步加载不同的数据集以减轻内存压力）
        round_no = epoch // round_len
        if epoch != 0 and epoch % round_len == 0:
            config.defrost()
            config.DATA.FOLDER_NAME = f'training_data_{round_no}/'
            get_params_from_preprocess(config)
            config.freeze()
        
        # 设置分布式采样器的epoch
        if torch.cuda.device_count() != 1:
            data_loader.sampler.set_epoch(epoch)
        
        # 训练一个epoch
        train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler)
        
        # 保存检查点（仅主进程且满足保存频率）
        if ((torch.cuda.device_count() != 1 and dist.get_rank() == 0) or torch.cuda.device_count() == 1) and (
                epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, optimizer, lr_scheduler, logger)
        
        # 可选的睡眠控制（用于办公室环境，控制仅在夜间运行）
        daily_period = time.time() - daily_time
        if daily_period > 60 * 60 * 12:  # 超过12小时
            logger.info("睡眠接近12小时...")
            save_checkpoint(config, epoch, model_without_ddp, optimizer, lr_scheduler, logger)
            time.sleep(60 * 60 * 24 - daily_period)
            daily_time = time.time()

    # 记录总训练时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'训练时间：{total_time_str}')


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler):
    """
    训练一个epoch
    
    该函数执行一个完整epoch的训练，包括：
    - 前向传播计算损失
    - 反向传播计算梯度
    - 梯度裁剪和优化器更新
    - 学习率调度
    - 训练指标记录
    
    参数:
        config: 配置对象
        model: 神经网络模型
        data_loader: 数据加载器
        optimizer: 优化器
        epoch: 当前epoch编号
        lr_scheduler: 学习率调度器
    
    训练流程：
        1. 从数据加载器获取批次数据（Jones矩阵和掩码）
        2. 前向传播计算重建损失
        3. 根据梯度累积步数进行反向传播
        4. 梯度裁剪防止梯度爆炸
        5. 更新模型参数和学习率
    """
    model.train()
    optimizer.zero_grad()

    # 训练指标记录器
    num_steps = len(data_loader)
    batch_time = AverageMeter()  # 批次处理时间
    loss_meter = AverageMeter()  # 损失值
    norm_meter = AverageMeter()  # 梯度范数

    start = time.time()
    end = time.time()
    
    # 遍历数据批次
    for idx, combine in enumerate(data_loader):
        img, mask = combine  # img: Jones矩阵数据, mask: 掩码
        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        
        # 前向传播（支持混合精度）
        if config.AMP_TYPE == 'pytorch':
            # 使用PyTorch原生混合精度
            with autocast():
                loss = model(img, mask)
        else:
            loss = model(img, mask)
        
        # 梯度累积：将损失按累积步数平均
        loss = loss / config.TRAIN.ACCUMULATION_STEPS
        
        # 反向传播（根据不同的混合精度类型）
        if config.AMP_TYPE == 'pytorch':
            # PyTorch AMP：使用GradScaler
            scaler.scale(loss).backward()
            grad_norm = handle_gradient(model.parameters())
        elif config.AMP_OPT_LEVEL != "O0":
            # Nvidia Apex AMP
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            grad_norm = handle_gradient(amp.master_params(optimizer))
        else:
            # 不使用混合精度
            loss.backward()
            grad_norm = handle_gradient(model.parameters())

        # 梯度累积达到指定步数后更新参数
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            if config.AMP_TYPE == "pytorch":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            # 更新学习率
            lr_scheduler.step_update(epoch * num_steps + idx)

        # 同步CUDA操作
        torch.cuda.synchronize()
        
        # 更新指标
        loss_meter.update(loss.item(), img.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        
        # 定期打印训练信息
        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'训练: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'预计剩余时间 {datetime.timedelta(seconds=int(etas))} 学习率 {lr:.6f}\t'
                f'时间 {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'损失 {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'梯度范数 {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'内存 {memory_used:.0f}MB')
    
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} 训练耗时 {datetime.timedelta(seconds=int(epoch_time))}")


def handle_gradient(parameters):
    """
    处理梯度（裁剪或计算范数）
    
    根据配置决定是否对梯度进行裁剪以防止梯度爆炸。
    
    参数:
        parameters: 模型参数或优化器的主参数
        
    返回:
        梯度的总范数值
    """
    if config.TRAIN.CLIP_GRAD:
        # 梯度裁剪：限制梯度范数的最大值
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.TRAIN.CLIP_GRAD)
    else:
        # 仅计算梯度范数，不裁剪
        grad_norm = get_grad_norm(parameters)
    return grad_norm


@torch.no_grad()
def validate(config, data_loader, model):
    """
    验证/重建函数
    
    在重建模式下，使用训练好的模型重建设计的Jones矩阵。
    该函数不更新模型参数，仅进行前向传播。
    
    参数:
        config: 配置对象
        data_loader: 数据加载器（包含设计的Jones矩阵和掩码）
        model: 训练好的模型
        
    返回:
        loss_meter.avg: 平均重建损失
        JM_recon: 重建的Jones矩阵，形状为[N, wavelengths * 6]
    
    处理流程：
        1. 从数据加载器获取设计的Jones矩阵和掩码
        2. 使用模型重建被掩码的部分
        3. 收集所有重建结果
        4. 将结果重塑为适合保存的格式
    """
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    
    # 初始化重建结果数组
    amp = config.DATA.BATCH_SIZE
    JM_recon = np.zeros((len(data_loader) * amp, 1, config.DATA.SIZE_X, config.DATA.SIZE_Y))

    end = time.time()
    
    # 遍历数据批次进行重建
    for idx, (images, masks) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        # 前向传播：重建Jones矩阵
        loss, recon = model(images, masks)

        loss_meter.update(loss.item(), images.size(0))

        # 保存重建结果
        if images.size(0) == amp:
            JM_recon[idx * amp: (idx + 1) * amp, :, :, :] = recon.cpu().numpy()
        else:
            # 处理最后一个不完整的批次
            JM_recon[idx * amp: idx * amp + images.size(0), :, :, :] = recon.cpu().numpy()

        # 记录处理时间
        batch_time.update(time.time() - end)
        end = time.time()

        # 定期打印验证信息
        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'测试: [{idx}/{len(data_loader)}]\t'
                f'时间 {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'损失 {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'内存 {memory_used:.0f}MB')
    
    # 重塑为[样本数, 波长数 * 6]的格式
    JM_recon = JM_recon.reshape((JM_recon.shape[0] * JM_recon.shape[1], JM_recon.shape[2] * JM_recon.shape[3]))
    return loss_meter.avg, JM_recon


if __name__ == '__main__':
    """
    主程序入口
    
    执行流程：
    1. 解析命令行参数
    2. 配置混合精度训练
    3. 设置分布式训练环境
    4. 配置随机种子
    5. 调整学习率（线性缩放规则）
    6. 创建输出目录和日志记录器
    7. 保存配置
    8. 启动主训练或重建流程
    """
    
    # 解析命令行参数
    _, config = parse_option()

    # 配置混合精度训练
    if config.AMP_TYPE == 'pytorch':
        # 使用PyTorch原生AMP
        scaler = GradScaler()
    elif config.AMP_TYPE == 'apex' and config.AMP_OPT_LEVEL != "O0":
        # 使用Nvidia Apex AMP
        assert amp is not None, "amp未安装！"

    # 配置分布式训练
    if torch.cuda.device_count() != 1:
        # 多GPU环境：初始化分布式进程组
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            print(f"环境变量中的RANK和WORLD_SIZE：{rank}/{world_size}")
        else:
            rank = -1
            world_size = -1

        torch.cuda.set_device(config.LOCAL_RANK)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.distributed.barrier()
        seed = config.SEED + dist.get_rank()
    else:
        # 单GPU环境
        seed = config.SEED

    # 设置随机种子以确保可重复性
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # 应用学习率的线性缩放规则
    # 根据批次大小、GPU数量和梯度累积步数调整学习率
    factor = 1 if torch.cuda.device_count() == 1 else dist.get_world_size()
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * factor / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * factor / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * factor / 512.0
    
    # 考虑梯度累积的影响
    linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
    linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
    linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS

    # 更新配置中的学习率
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    # 创建输出目录
    os.makedirs(config.OUTPUT, exist_ok=True)
    
    # 创建日志记录器（仅主进程）
    rank_indicator = 0 if torch.cuda.device_count() == 1 else dist.get_rank()
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=rank_indicator, name=f"{config.MODEL.NAME}")

    # 保存完整配置到JSON文件（仅主进程）
    if rank_indicator == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"完整配置已保存到 {path}")

    # 打印配置
    logger.info(config.dump())

    # 启动主流程
    main(config)
