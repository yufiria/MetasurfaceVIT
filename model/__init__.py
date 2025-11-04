"""
模型构建模块
该文件提供了统一的模型构建接口。
根据训练阶段（预训练或微调）自动选择相应的模型架构。
"""

from .vision_transformer import build_vit
from .simmim import build_simmim


def build_model(config, is_pretrain=True):
    """
    构建模型的统一接口
    
    根据训练阶段选择合适的模型架构：
    - 预训练阶段：使用SimMIM模型进行掩码图像建模
    - 微调阶段：使用Vision Transformer模型进行参数预测
    
    参数:
        config: 配置对象，包含模型的所有超参数
        is_pretrain: 是否为预训练阶段
            True - 构建SimMIM预训练模型
            False - 构建Vision Transformer微调模型
            
    返回:
        配置好的模型实例
    """
    if is_pretrain:
        # 预训练阶段：SimMIM模型（包含编码器和解码器）
        model = build_simmim(config)
    else:
        # 微调阶段：Vision Transformer模型（包含分类头）
        model = build_vit(config)

    return model
