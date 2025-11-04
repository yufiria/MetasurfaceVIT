"""
数据加载模块
该文件提供了统一的数据加载接口。
根据不同的训练/评估模式，自动选择相应的数据加载器。
"""

from .data_simmim import build_loader_simmim
from .data_finetune import build_loader_finetune, build_loader_prediction
from .data_recon import build_loader_recon


def build_loader(config, logger, type):
    """
    构建数据加载器的统一接口
    
    根据训练/评估类型选择合适的数据加载器：
    - pre_trained: SimMIM预训练数据加载器（带掩码的Jones矩阵）
    - finetune: 微调数据加载器（Jones矩阵和结构参数对）
    - reconstruct: 重建数据加载器（用于从设计的Jones矩阵重建）
    - predict: 预测数据加载器（用于结构参数预测）
    
    参数:
        config: 配置对象，包含数据路径、批次大小等信息
        logger: 日志记录器
        type: 数据加载类型，可选值：
            'pre_trained' - 预训练数据
            'finetune' - 微调数据
            'reconstruct' - 重建数据
            'predict' - 预测数据
            
    返回:
        配置好的DataLoader实例
        
    抛出:
        ValueError: 如果type参数无效
    """
    if type == 'pre_trained':
        # 预训练：加载Jones矩阵数据并生成掩码
        return build_loader_simmim(config, logger)
    elif type == 'finetune':
        # 微调：加载Jones矩阵和对应的结构参数
        return build_loader_finetune(config, logger)
    elif type == 'reconstruct':
        # 重建：加载设计的Jones矩阵进行重建
        return build_loader_recon(config, logger)
    elif type == 'predict':
        # 预测：加载重建的Jones矩阵进行参数预测
        return build_loader_prediction(config, logger)
    else:
        raise ValueError("Invalid type (should be 'pre_trained', 'finetune', 'reconstruct', or 'predict')!")
