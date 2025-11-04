"""
工具函数模块
该文件包含训练过程中使用的各种工具函数，包括：
- 模型检查点的加载和保存
- 梯度范数计算
- 自动恢复训练
- 预训练模型权重的加载和重映射
- 相对位置编码的几何插值
"""

import os
import torch
import torch.distributed as dist
import numpy as np
from scipy import interpolate
from apex import amp


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    """
    加载模型检查点
    
    从指定路径加载模型权重、优化器状态和学习率调度器状态。
    支持从URL或本地文件加载。
    
    参数:
        config: 配置对象，包含检查点路径等信息
        model: 神经网络模型
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        logger: 日志记录器
    """
    logger.info(f">>>>>>>>>> Resuming from {config.MODEL.RESUME} ..........")
    
    # 加载检查点文件
    if config.MODEL.RESUME.startswith('https'):
        # 从URL加载
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        # 从本地文件加载
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    # 加载模型权重
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    
    # 加载优化器和调度器状态（仅在非评估模式下）
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        # 更新起始epoch
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        
        # 加载混合精度训练状态（仅在使用apex且配置匹配时）
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")

    # 清理内存
    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, optimizer, lr_scheduler, logger):
    """
    保存模型检查点
    
    保存模型权重、优化器状态、学习率调度器状态和训练配置。
    
    参数:
        config: 配置对象
        epoch: 当前epoch数
        model: 神经网络模型
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        logger: 日志记录器
    """
    # 构建保存状态字典
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'epoch': epoch,
                  'config': config}
    
    # 如果使用apex混合精度训练，保存amp状态
    if config.AMP_TYPE == 'apex' and config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    # 保存到文件
    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    """
    计算参数梯度的范数
    
    用于监控训练过程中的梯度大小，帮助诊断梯度爆炸或消失问题。
    
    参数:
        parameters: 模型参数（可以是单个Tensor或参数列表）
        norm_type: 范数类型，默认为2（L2范数）
        
    返回:
        梯度的总范数值
    """
    # 将单个参数转换为列表
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    # 过滤出有梯度的参数
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    
    # 计算总范数
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    
    return total_norm


def auto_resume_helper(output_dir, logger):
    """
    自动恢复训练的辅助函数
    
    在输出目录中查找最新的检查点文件，用于自动恢复训练。
    
    参数:
        output_dir: 输出目录路径
        logger: 日志记录器
        
    返回:
        最新检查点文件的路径，如果没有找到则返回None
    """
    # 列出目录中所有的.pth文件
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    logger.info(f"All checkpoints founded in {output_dir}: {checkpoints}")
    
    # 找到最新的检查点
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        logger.info(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    
    return resume_file


def reduce_tensor(tensor):
    """
    在分布式训练中规约张量
    
    将所有进程的张量求和并平均，用于分布式训练中的指标同步。
    
    参数:
        tensor: 要规约的张量
        
    返回:
        规约后的张量（所有进程的平均值）
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def load_pretrained(config, path, model, logger):
    """
    加载预训练模型权重用于微调
    
    从预训练检查点加载模型权重，自动处理编码器前缀和相对位置编码的重映射。
    
    参数:
        config: 配置对象
        path: 预训练模型检查点路径
        model: 要加载权重的模型
        logger: 日志记录器
    """
    logger.info(f">>>>>>>>>> Fine-tuned from {path} ..........")
    checkpoint = torch.load(path, map_location='cpu')
    checkpoint_model = checkpoint['model']
    
    # 检查并移除'encoder.'前缀
    # 预训练模型使用SimMIM包装，参数名有'encoder.'前缀
    # 微调模型直接使用VisionTransformer，没有这个前缀
    if any([True if 'encoder.' in k else False for k in checkpoint_model.keys()]):
        checkpoint_model = {k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if k.startswith('encoder.')}
        logger.info('Detect pre-trained model, remove [encoder.] prefix.')
    else:
        logger.info('Detect non-pre-trained model, pass without doing anything.')

    # 重映射预训练模型的键以适配当前模型
    logger.info(f">>>>>>>>>> Remapping pre-trained keys for VIT ..........")
    checkpoint_model = remap_pretrained_keys_vit(model, checkpoint_model, logger)

    # 加载权重（允许部分匹配）
    msg = model.load_state_dict(checkpoint_model, strict=False)
    logger.info(msg)
    
    # 清理内存
    del checkpoint
    torch.cuda.empty_cache()
    logger.info(f">>>>>>>>>> loaded successfully '{path}'")


def remap_pretrained_keys_vit(model, checkpoint_model, logger):
    """
    重映射Vision Transformer预训练模型的键
    
    处理两个主要问题：
    1. 将共享的相对位置偏置复制到每个Transformer层
    2. 当patch size不匹配时，对相对位置编码进行几何插值
    
    参数:
        model: 目标模型
        checkpoint_model: 预训练模型的状态字典
        logger: 日志记录器
        
    返回:
        重映射后的状态字典
    """
    # 1. 处理共享的相对位置偏置
    # 如果模型使用相对位置偏置且检查点中有共享的相对位置偏置表
    if getattr(model, 'use_rel_pos_bias', False) and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
        logger.info("Expand the shared relative position embedding to each transformer block.")
        num_layers = model.get_num_layers()
        rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
        
        # 将共享的相对位置偏置复制到每个Transformer块
        for i in range(num_layers):
            checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()
        
        # 移除原始的共享偏置
        checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")
    
    # 2. 处理相对位置编码的几何插值
    # 当预训练和微调的patch size不匹配时需要插值
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        # 移除相对位置索引（这些会在模型初始化时重新计算）
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

        # 对相对位置偏置表进行插值
        if "relative_position_bias_table" in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.patch_shape
            
            # 确保patch是正方形（目前不支持非正方形patch）
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            
            # 计算额外的token数量（如cls_token）
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            
            # 如果尺寸不匹配，进行插值
            if src_size != dst_size:
                logger.info("Position interpolate for %s from %dx%d to %dx%d" % (key, src_size, src_size, dst_size, dst_size))
                
                # 分离额外的token
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    """计算几何级数和"""
                    return a * (1.0 - r ** n) / (1.0 - r)

                # 使用二分搜索找到合适的几何级数比率
                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # 生成源位置的几何级数分布
                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                # 构建源位置坐标
                r_ids = [-_ for _ in reversed(dis)]
                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                # 构建目标位置坐标（均匀分布）
                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                logger.info("Original positions = %s" % str(x))
                logger.info("Target positions = %s" % str(dx))

                # 对每个注意力头进行二维插值
                all_rel_pos_bias = []
                for i in range(num_attn_heads):
                    # 将一维的偏置重塑为二维
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    # 使用三次插值
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    # 插值到目标尺寸
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                # 合并所有注意力头的结果
                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                # 重新添加额外的token
                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                checkpoint_model[key] = new_rel_pos_bias
    
    return checkpoint_model
