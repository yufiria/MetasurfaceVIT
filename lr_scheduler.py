"""
学习率调度器模块
该文件用于构建和管理训练过程中的学习率调度策略。
支持余弦退火、线性衰减、步进衰减和多步衰减等多种调度方式，并包含预热机制。
"""

from collections import Counter
from bisect import bisect_right

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler


def build_scheduler(config, optimizer, n_iter_per_epoch):
    """
    构建学习率调度器
    
    根据配置创建相应的学习率调度器，支持余弦、线性、步进和多步调度策略。
    所有调度器都支持预热（warmup）阶段。
    
    参数:
        config: 配置对象，包含调度器类型和参数
        optimizer: PyTorch优化器对象
        n_iter_per_epoch: 每个epoch的迭代次数
        
    返回:
        配置好的学习率调度器对象
        
    注意:
        某些调度器尚未经过充分调试
    """
    # 计算总步数和各阶段步数
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)  # 总训练步数
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)  # 预热步数
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)  # 衰减步数
    multi_steps = [i * n_iter_per_epoch for i in config.TRAIN.LR_SCHEDULER.MULTISTEPS]  # 多步衰减的里程碑

    lr_scheduler = None
    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        # 余弦退火调度器：学习率按余弦曲线从初始值衰减到最小值
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,  # 总步数
            lr_min=config.TRAIN.MIN_LR,  # 最小学习率
            warmup_lr_init=config.TRAIN.WARMUP_LR,  # 预热初始学习率
            warmup_t=warmup_steps,  # 预热步数
            cycle_limit=1,  # 循环次数
            t_in_epochs=False,  # 以步数而非epoch为单位
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'linear':
        # 线性衰减调度器：学习率线性从初始值衰减到最小值
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,  # 最小学习率比率
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'step':
        # 步进衰减调度器：每隔固定步数将学习率乘以衰减率
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,  # 衰减间隔
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,  # 衰减率
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'multistep':
        # 多步衰减调度器：在指定的里程碑处将学习率乘以gamma
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            milestones=multi_steps,  # 里程碑步数列表
            gamma=config.TRAIN.LR_SCHEDULER.GAMMA,  # 衰减系数
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    """
    线性学习率调度器
    
    在预热阶段后，学习率从基础值线性衰减到最小值。
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        """
        初始化线性学习率调度器
        
        参数:
            optimizer: PyTorch优化器
            t_initial: 总训练步数或epoch数
            lr_min_rate: 最小学习率占初始学习率的比率
            warmup_t: 预热步数或epoch数
            warmup_lr_init: 预热初始学习率
            t_in_epochs: 是否以epoch为单位（False表示以步数为单位）
            noise_range_t: 添加噪声的范围
            noise_pct: 噪声百分比
            noise_std: 噪声标准差
            noise_seed: 噪声随机种子
            initialize: 是否初始化调度器
        """
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        
        # 计算预热阶段的学习率增长步长
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        """
        计算给定时间步的学习率
        
        参数:
            t: 当前时间步（步数或epoch数）
            
        返回:
            学习率列表（对应优化器中的各个参数组）
        """
        if t < self.warmup_t:
            # 预热阶段：线性增长
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            # 训练阶段：线性衰减
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        """获取指定epoch的学习率（当t_in_epochs为True时使用）"""
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        """获取指定更新步数的学习率（当t_in_epochs为False时使用）"""
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


class MultiStepLRScheduler(Scheduler):
    """
    多步学习率调度器
    
    在指定的里程碑处将学习率乘以gamma进行衰减。
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, milestones, gamma=0.1, warmup_t=0, warmup_lr_init=0, t_in_epochs=True) -> None:
        """
        初始化多步学习率调度器
        
        参数:
            optimizer: PyTorch优化器
            milestones: 里程碑列表，在这些时间点衰减学习率
            gamma: 学习率衰减系数
            warmup_t: 预热步数或epoch数
            warmup_lr_init: 预热初始学习率
            t_in_epochs: 是否以epoch为单位
        """
        super().__init__(optimizer, param_group_field="lr")
        
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        
        # 计算预热阶段的学习率增长步长
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]
        
        # 确保预热期不超过第一个里程碑
        assert self.warmup_t <= min(self.milestones)
    
    def _get_lr(self, t):
        """
        计算给定时间步的学习率
        
        参数:
            t: 当前时间步（步数或epoch数）
            
        返回:
            学习率列表（对应优化器中的各个参数组）
        """
        if t < self.warmup_t:
            # 预热阶段：线性增长
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            # 训练阶段：在每个里程碑处乘以gamma
            # bisect_right返回t应插入的位置，即已经过了几个里程碑
            lrs = [v * (self.gamma ** bisect_right(self.milestones, t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        """获取指定epoch的学习率（当t_in_epochs为True时使用）"""
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        """获取指定更新步数的学习率（当t_in_epochs为False时使用）"""
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None
