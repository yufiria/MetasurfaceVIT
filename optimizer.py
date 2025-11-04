"""
优化器构建模块
该文件用于构建预训练和微调阶段的优化器。
支持不同的优化器类型（SGD、AdamW）和参数组配置（权重衰减、层级学习率衰减等）。
"""

import json
from functools import partial
from torch import optim as optim


def build_optimizer(config, model, logger, is_pretrain):
    """
    构建优化器的主函数
    根据训练阶段（预训练或微调）选择相应的优化器构建方法
    
    参数:
        config: 配置对象
        model: 神经网络模型
        logger: 日志记录器
        is_pretrain: 布尔值，True表示预训练阶段，False表示微调阶段
        
    返回:
        配置好的优化器对象
    """
    if is_pretrain:
        return build_pretrain_optimizer(config, model, logger)
    else:
        return build_finetune_optimizer(config, model, logger)


def build_pretrain_optimizer(config, model, logger):
    """
    构建预训练阶段的优化器
    
    为预训练阶段配置优化器，处理需要跳过权重衰减的参数。
    包括位置编码、偏置等参数不应用权重衰减。
    
    参数:
        config: 配置对象
        model: 神经网络模型
        logger: 日志记录器
        
    返回:
        配置好的预训练优化器
    """
    logger.info('>>>>>>>>>> Build Optimizer for Pre-training Stage')
    
    # 获取模型中需要跳过权重衰减的参数名称
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
        logger.info(f'No weight decay: {skip}')

    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
        logger.info(f'No weight decay keywords: {skip_keywords}')

    # 获取参数组（分为需要权重衰减和不需要权重衰减两组）
    parameters = get_pretrain_param_groups(model, logger, skip, skip_keywords)

    # 根据配置选择优化器类型
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        # SGD优化器：带动量和Nesterov加速
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        # AdamW优化器：Adam的改进版本，解耦权重衰减
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    logger.info(optimizer)
    return optimizer
    

def get_pretrain_param_groups(model, logger, skip_list=(), skip_keywords=()):
    """
    获取预训练阶段的参数组
    
    将模型参数分为两组：需要权重衰减的参数和不需要权重衰减的参数。
    通常偏置项、LayerNorm参数、位置编码等不应用权重衰减。
    
    参数:
        model: 神经网络模型
        logger: 日志记录器
        skip_list: 需要跳过权重衰减的参数名称列表
        skip_keywords: 需要跳过权重衰减的参数名称关键词
        
    返回:
        包含两个参数组的列表
    """
    has_decay = []  # 需要权重衰减的参数
    no_decay = []   # 不需要权重衰减的参数
    has_decay_name = []  # 记录参数名称用于日志
    no_decay_name = []

    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 判断参数是否需要权重衰减
        # 一维参数（如偏置）、名称在跳过列表中的参数不应用权重衰减
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    
    # 记录参数分组情况
    logger.info(f'No decay params: {no_decay_name}')
    logger.info(f'Has decay params: {has_decay_name}')
    
    # 返回两个参数组：第一组应用权重衰减，第二组不应用
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def build_finetune_optimizer(config, model, logger):
    """
    构建微调阶段的优化器
    
    为微调阶段配置优化器，支持层级学习率衰减。
    不同深度的Transformer层使用不同的学习率，越靠近输入层学习率越小。
    
    参数:
        config: 配置对象
        model: 神经网络模型
        logger: 日志记录器
        
    返回:
        配置好的微调优化器
    """
    logger.info('>>>>>>>>>> Build Optimizer for Fine-tuning Stage')
    
    # 获取模型层数
    num_layers = config.MODEL.VIT.DEPTH
    # 创建层级识别函数（+2是为了包含patch_embed和head层）
    get_layer_func = partial(get_vit_layer, num_layers=num_layers + 2)

    # 计算每层的学习率缩放因子
    # 使用指数衰减：越深的层学习率越大
    scales = list(config.TRAIN.LAYER_DECAY ** i for i in reversed(range(num_layers + 2)))
    
    # 获取需要跳过权重衰减的参数
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
        logger.info(f'No weight decay: {skip}')
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
        logger.info(f'No weight decay keywords: {skip_keywords}')

    # 获取微调参数组（按层分组，每层有不同的学习率）
    parameters = get_finetune_param_groups(
        model, logger, config.TRAIN.BASE_LR, config.TRAIN.WEIGHT_DECAY,
        get_layer_func, scales, skip, skip_keywords)
    
    # 根据配置选择优化器类型
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    logger.info(optimizer)
    return optimizer


def get_vit_layer(name, num_layers):
    """
    确定Vision Transformer参数所属的层级
    
    根据参数名称确定其在模型中的层级位置，用于层级学习率衰减。
    
    参数:
        name: 参数名称
        num_layers: 总层数
        
    返回:
        层级编号（0表示最浅层）
    """
    # 特殊token和位置编码属于第0层
    if name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    # Patch embedding属于第0层
    elif name.startswith("patch_embed"):
        return 0
    # 相对位置偏置属于最后一层
    elif name.startswith("rel_pos_bias"):
        return num_layers - 1
    # Transformer blocks按序号分层
    elif name.startswith("blocks"):
        layer_id = int(name.split('.')[1])
        return layer_id + 1
    # 其他参数（如head）属于最后一层
    else:
        return num_layers - 1


def get_finetune_param_groups(model, logger, lr, weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()):
    """
    获取微调阶段的参数组
    
    将参数按层级和是否需要权重衰减分组，每组有不同的学习率和权重衰减设置。
    
    参数:
        model: 神经网络模型
        logger: 日志记录器
        lr: 基础学习率
        weight_decay: 权重衰减系数
        get_layer_func: 获取参数层级的函数
        scales: 每层的学习率缩放因子列表
        skip_list: 需要跳过权重衰减的参数名称列表
        skip_keywords: 需要跳过权重衰减的参数名称关键词
        
    返回:
        参数组列表，每组包含参数、学习率、权重衰减等信息
    """
    parameter_group_names = {}  # 用于记录日志
    parameter_group_vars = {}   # 实际的参数组

    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 确定是否需要权重衰减
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        
        # 确定层级
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        # 创建参数组
        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            # 记录参数组信息（用于日志）
            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
            # 实际的参数组（包含参数张量）
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale
            }

        # 将参数添加到对应的组中
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    
    # 记录参数分组情况
    logger.info("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def check_keywords_in_name(name, keywords=()):
    """
    检查参数名称中是否包含指定关键词
    
    参数:
        name: 参数名称
        keywords: 关键词元组
        
    返回:
        布尔值，True表示包含关键词
    """
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin