"""
配置文件
该文件用于定义MetasurfaceVIT项目的所有配置参数，包括数据设置、模型设置、训练设置等。
使用YACS库进行配置管理，支持从命令行参数和配置文件中更新配置。
"""

import os
from yacs.config import CfgNode as CN

# 创建全局配置对象
_C = CN()
# 基础配置文件列表
_C.BASE = ['']

# -----------------------------------------------------------------------------
# 数据相关设置
# -----------------------------------------------------------------------------
_C.DATA = CN()
# 单个GPU的批次大小，可通过命令行参数覆盖
_C.DATA.BATCH_SIZE = 128
# 数据集路径，将从预处理阶段写入
_C.DATA.PATH = './preprocess/'
# 数据文件夹名称
_C.DATA.FOLDER_NAME = ''
# 预训练数据集的对应路径，主要用于main_metalens.py，将通过与微调数据集配对自动填充
_C.DATA.PREFOLDER_NAME = ''
# Jones矩阵数据文件前缀
_C.DATA.PREFIX_JM = ''
# 结构参数数据文件前缀
_C.DATA.PREFIX_PARAM = ''
# 数据文件后缀
_C.DATA.SUFFIX = ''
# 输入数据X维度大小（波长点数），需要从预处理设置中覆盖
_C.DATA.SIZE_X = 20
# 输入数据Y维度大小（Jones矩阵展平后的维度）
_C.DATA.SIZE_Y = 6
# 在DataLoader中固定CPU内存以更高效地传输到GPU
_C.DATA.PIN_MEMORY = True
# 数据加载线程数，预训练和微调可能需要不同的设置
_C.DATA.NUM_WORKERS = 4
# 掩码类型，整数值从0到5。0表示随机使用1-5类型的掩码
_C.DATA.MASK_TYPE = 1
# 数据分割数量，将从预处理阶段填充
_C.DATA.DIVIDE_NUM = None
# 由于训练数据对缓存和内存压力较大，在预训练期间逐步加载不同类型的数据
_C.DATA.DATA_SIZE = 3
# 数据开始索引
_C.DATA.DATA_START = 1

# -----------------------------------------------------------------------------
# 模型相关设置
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# 模型名称
_C.MODEL.NAME = 'metaVIT'
# 要恢复的检查点路径，可通过命令行参数覆盖
_C.MODEL.RESUME = ''
# 结构参数的数量，用于微调阶段
_C.MODEL.NUM_PARA = 6
# Dropout比率（以下参数尚未进行详尽的网格搜索）
_C.MODEL.DROP_RATE = 0.0
# DropPath比率（随机深度）
_C.MODEL.DROP_PATH_RATE = 0.1
# 标签平滑参数
_C.MODEL.LABEL_SMOOTHING = 0.1
# 损失计算类型，确定使用Jones矩阵的哪部分来计算损失
# 0: 计算整个Jones矩阵的损失
# 1: 仅计算被掩码部分的损失
# 2: 仅计算未被掩码部分的损失
_C.MODEL.LOSS_TYPE = 0

# Vision Transformer 参数
_C.MODEL.VIT = CN()
# Patch大小
_C.MODEL.VIT.PATCH_SIZE = 1
# 输入通道数
_C.MODEL.VIT.IN_CHANS = 1
# 嵌入维度
_C.MODEL.VIT.EMBED_DIM = 512
# Transformer层数
_C.MODEL.VIT.DEPTH = 12
# 注意力头数
_C.MODEL.VIT.NUM_HEADS = 12
# MLP隐藏层与嵌入维度的比率
_C.MODEL.VIT.MLP_RATIO = 4
# 是否对QKV使用偏置
_C.MODEL.VIT.QKV_BIAS = True
# 初始化值
_C.MODEL.VIT.INIT_VALUES = 0.1
# 是否使用绝对位置编码
_C.MODEL.VIT.USE_APE = True
# 是否使用相对位置偏置
_C.MODEL.VIT.USE_RPB = False
# 是否使用共享的相对位置偏置
_C.MODEL.VIT.USE_SHARED_RPB = False
# 是否使用平均池化
_C.MODEL.VIT.USE_MEAN_POOLING = False


# -----------------------------------------------------------------------------
# 训练相关设置
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# 训练起始轮数
_C.TRAIN.START_EPOCH = 0
# 总训练轮数
_C.TRAIN.EPOCHS = 300
# 预热轮数
_C.TRAIN.WARMUP_EPOCHS = 5
# 权重衰减系数
_C.TRAIN.WEIGHT_DECAY = 0.05
# 基础学习率
_C.TRAIN.BASE_LR = 5e-4
# 预热学习率
_C.TRAIN.WARMUP_LR = 5e-7
# 最小学习率
_C.TRAIN.MIN_LR = 5e-6
# 梯度裁剪范数
_C.TRAIN.CLIP_GRAD = 5.0
# 从最新检查点自动恢复
_C.TRAIN.AUTO_RESUME = True
# 梯度累积步数
_C.TRAIN.ACCUMULATION_STEPS = 1
# 是否使用梯度检查点以节省内存
_C.TRAIN.USE_CHECKPOINT = False

# 学习率调度器配置
_C.TRAIN.LR_SCHEDULER = CN()
# 调度器名称
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# 学习率衰减间隔轮数，用于StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# 学习率衰减率，用于StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# Gamma值和多步数值，用于MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# 优化器配置
_C.TRAIN.OPTIMIZER = CN()
# 优化器名称
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# 优化器Epsilon值
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# 优化器Betas参数
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD动量
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
# 层级学习率衰减
_C.TRAIN.LAYER_DECAY = 1.0

# -----------------------------------------------------------------------------
# 数据增强设置（此部分从SIMMIM模型借鉴而来，可能不完全适用于本项目）
# -----------------------------------------------------------------------------
_C.AUG = CN()
# 颜色抖动因子
_C.AUG.COLOR_JITTER = 0.4
# AutoAugment策略，"v0"或"original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# 随机擦除概率
_C.AUG.REPROB = 0.25
# 随机擦除模式
_C.AUG.REMODE = 'pixel'
# 随机擦除数量
_C.AUG.RECOUNT = 1
# Mixup alpha值，>0时启用mixup
_C.AUG.MIXUP = 0.8
# Cutmix alpha值，>0时启用cutmix
_C.AUG.CUTMIX = 1.0
# Cutmix最小/最大比率，设置后会覆盖alpha并启用cutmix
_C.AUG.CUTMIX_MINMAX = None
# 执行mixup或cutmix的概率
_C.AUG.MIXUP_PROB = 1.0
# 当mixup和cutmix都启用时，切换到cutmix的概率
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# mixup/cutmix参数的应用方式："batch"、"pair"或"elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# 其他杂项设置
# -----------------------------------------------------------------------------
# 混合精度优化级别，如果是O0则不使用amp（'O0', 'O1', 'O2'）
# 可通过命令行参数覆盖
_C.AMP_OPT_LEVEL = ''
# 混合精度类型：'apex'或'pytorch'
_C.AMP_TYPE = 'apex'

# 输出文件夹路径，可通过命令行参数覆盖
_C.OUTPUT = ''
# 预训练输出路径
_C.PRETRAINED_OUTPUT = ''

# 实验标签，可通过命令行参数覆盖
_C.TAG = 'default'
# 保存检查点的频率
_C.SAVE_FREQ = 3
# 日志打印频率
_C.PRINT_FREQ = 50

# 随机种子
_C.SEED = 0

# 评估模式
_C.EVAL_MODE = False
# 参数预测路径
_C.PREDICT_PARA_PATH = './evaluation/metasurface_verification/predict_params/'
# 仅测试吞吐量，可通过命令行参数覆盖
_C.THROUGHPUT_MODE = False
# DistributedDataParallel的本地进程编号，通过命令行参数给出
_C.LOCAL_RANK = 0

# 重建相关设置（可能需要开一个新的配置分支如_C.RECON.xxx）
# 重建模式
_C.RECON_MODE = False
# 重建路径
_C.RECON_PATH = './evaluation/metasurface_design'
# 重建类型
_C.RECON_TYPE = 6
# 时间戳标记
_C.TREATMENT = ''

# 金属透镜相关设置
_C.LENS = CN()
# 是否预先存在
_C.LENS.PREEXIST = True
# 透镜名称
_C.LENS.NAME = 'metalens'
# 输出路径
_C.LENS.OUTPUT = 'metalens_output'
# X方向总像素数
_C.LENS.X_TOTAL = 256
# 焦距（单位：米）
_C.LENS.FOCUS = 75 * 10 ** -6
# X方向单元尺寸（单位：米）
_C.LENS.X_UNIT = 0.8 * 10 ** -6
# Y方向单元尺寸（单位：米）
_C.LENS.Y_UNIT = 0.4 * 10 ** -6
# 变化率
_C.LENS.CHANGE_RATE = 0.9
# 上限值（单位：米）
_C.LENS.UPLIMIT = 50 * 10 ** -9
# 偏置值
_C.LENS.BIAS = 0.3
# 目标文件
_C.LENS.TARGET = ''


def update_config(config, args):
    """
    更新配置参数
    根据命令行参数更新配置对象
    
    参数:
        config: 配置对象
        args: 命令行参数对象
    """
    config.defrost()

    def _check_args(name):
        """检查参数是否存在且有值"""
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # 从特定参数合并配置
    if _check_args('epoch'):
        config.TRAIN.EPOCHS = args.epoch
    if _check_args('mask_type'):
        config.DATA.MASK_TYPE = args.mask_type
    if _check_args('data_size'):
        config.DATA.DATA_SIZE = args.data_size
    if _check_args('data_start'):
        config.DATA.DATA_START = args.data_start
    if _check_args('base_lr'):
        config.TRAIN.BASE_LR = args.base_lr
    if _check_args('warmup_lr'):
        config.TRAIN.WARMUP_LR = args.warmup_lr
    if _check_args('min_lr'):
        config.TRAIN.MIN_LR = args.min_lr
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('data_folder_name'):
        config.DATA.FOLDER_NAME = args.data_folder_name
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('amp_type'):
        config.AMP_TYPE = args.amp_type
    if _check_args('amp_opt_level'):
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('pretrained_output'):
        config.PRETRAINED_OUTPUT = args.pretrained_output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('recon'):
        config.RECON_MODE = True
    if _check_args('recon_path'):
        config.RECON_PATH = args.recon_path
    if _check_args('recon_type'):
        config.RECON_TYPE = args.recon_type
    if _check_args('treatment'):
        config.TREATMENT = args.treatment
    if _check_args('eval_output'):
        config.PREDICT_PARA_PATH = args.eval_output
    if _check_args('target_file'):
        config.LENS.TARGET = args.target_file

    # 从预处理阶段更新参数
    if config.DATA.FOLDER_NAME == '':
        config.DATA.FOLDER_NAME = f'training_data_{config.DATA.DATA_START}'
    get_params_from_preprocess(config)
    
    # 从命令行输入更新参数（作为键值对）
    if args.opts:
        config.merge_from_list(args.opts)

    # 设置分布式训练的本地进程编号
    config.LOCAL_RANK = args.local_rank

    # 设置输出文件夹
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    config.PRETRAINED_OUTPUT = os.path.join(config.PRETRAINED_OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_params_from_preprocess(config):
    """
    从预处理阶段获取参数
    读取预处理生成的参数文件，更新配置中的数据相关参数
    
    参数:
        config: 配置对象
    """
    # 构建参数文件路径
    fixed_path = config.DATA.PATH + config.DATA.FOLDER_NAME + '/params_from_preprocess.txt'
    if not os.path.exists(fixed_path):
        raise ValueError('Required config file in the folder [preprocess] doesnt exist. You might run preprocess first.')

    # 读取参数文件内容
    with open(fixed_path, 'r') as file:
        content = file.read()
        content = content.split()

    # 将参数合并到配置中
    config.merge_from_list(content)


def get_config(args):
    """
    获取配置对象
    克隆全局配置并根据命令行参数更新
    
    参数:
        args: 命令行参数对象
        
    返回:
        更新后的配置对象
    """
    config = _C.clone()
    update_config(config, args)

    return config


def get_static_config():
    """
    获取静态配置对象
    不使用argparse更新的配置，直接返回默认配置的克隆
    
    返回:
        默认配置对象
    """
    config = _C.clone()
    return config

