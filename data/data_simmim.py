"""
SimMIM预训练数据加载模块
该文件实现了用于SimMIM预训练的数据加载和掩码生成功能。
包括Jones矩阵数据的读取、掩码生成策略和数据集类。

主要组件：
- MaskGenerator：生成多种类型的掩码策略
- DataTransform：数据转换和掩码应用
- MyDataSet：Jones矩阵数据集
- 数据读取和加载函数
"""

import random
import numpy as np
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset


class MaskGenerator:
    """
    掩码生成器
    
    为Jones矩阵数据生成不同类型的掩码，用于自监督预训练。
    支持5种掩码策略，分别用于不同的预训练任务。
    
    掩码类型说明：
    1. 掩码n-1个波长通道，只保留一个波长的完整Jones矩阵
    2. 保留所有振幅分量，只保留一个波长的相位分量
    3. 与类型1相同，但只保留11极化分量，掩码12和22分量
    4. 与类型2相同，但只保留11极化分量，掩码12和22分量
    5. 掩码所有波长的12和22极化分量，保留所有波长的11分量
    """
    
    def __init__(self, input_x=20, input_y=6):
        """
        初始化掩码生成器
        
        参数:
            input_x: 输入数据X维度（波长点数）
            input_y: 输入数据Y维度（Jones矩阵展平后的维度，通常为6）
        """
        self.input_x = input_x
        self.input_y = input_y

    def __call__(self, mask_type):
        """
        生成指定类型的掩码
        
        参数:
            mask_type: 掩码类型（1-5）
            
        返回:
            掩码张量，shape为[1, input_x, input_y]
            1表示被掩码（需要预测），0表示保留（作为输入）
        """
        # 随机选择一个波长索引
        random_num1 = random.randrange(self.input_x)
        # 初始化掩码（1表示掩码，0表示保留）
        mask = np.ones((self.input_x, self.input_y), dtype='float16')

        if mask_type == 1:
            # 类型1：只保留一个波长的完整Jones矩阵
            mask[random_num1, :] = 0
        elif mask_type == 2:
            # 类型2：保留所有振幅（前3列），只保留一个波长的相位（后3列）
            mask[:, :3] = 0  # 保留所有振幅
            mask[random_num1, 3:] = 0  # 保留一个波长的相位
        elif mask_type == 3:
            # 类型3：只保留一个波长的11分量（振幅和相位）
            mask[random_num1, [0, 3]] = 0
        elif mask_type == 4:
            # 类型4：保留所有波长的11振幅，只保留一个波长的11相位
            mask[:, 0] = 0  # 保留所有11振幅
            mask[random_num1, 3] = 0  # 保留一个波长的11相位
        elif mask_type == 5:
            # 类型5：保留所有波长的11分量
            mask[:, [0, 3]] = 0
        else:
            raise ValueError('mask_type should be int: 0 1 2 3 4 5')

        # 转换为PyTorch张量并添加channel维度
        mask = torch.tensor(mask)
        mask = mask.unsqueeze(0)
        return mask


class DataTransform:
    """
    数据转换类
    
    为Jones矩阵数据应用掩码转换，用于SimMIM预训练。
    """
    
    def __init__(self, config):
        """
        初始化数据转换
        
        参数:
            config: 配置对象，包含掩码类型等信息
        """
        self.mask_generator = MaskGenerator(input_x=config.DATA.SIZE_X, input_y=config.DATA.SIZE_Y)
        self.mask_type = config.DATA.MASK_TYPE

    def __call__(self, img):
        """
        应用数据转换
        
        参数:
            img: 输入Jones矩阵数据
            
        返回:
            (转换后的图像张量, 掩码张量)
        """
        # 转换为张量
        img = torch.tensor(img)
        # 如果mask_type为0，随机选择1-5中的一种掩码类型
        self.mask_type = random.randrange(1, 6) if self.mask_type == 0 else self.mask_type
        # 生成掩码
        mask = self.mask_generator(self.mask_type)
        return img, mask


class MyDataSet(Dataset):
    """
    Jones矩阵数据集
    
    用于加载和处理Jones矩阵数据的PyTorch数据集类。
    """
    
    def __init__(self, spectra: dict, transform=None):
        """
        初始化数据集
        
        参数:
            spectra: Jones矩阵数据字典，键为索引，值为数据
            transform: 数据转换函数
        """
        self.spectra = spectra
        self.transform = transform

    def __len__(self):
        """返回数据集大小"""
        return len(self.spectra)

    def __getitem__(self, item):
        """
        获取单个数据样本
        
        参数:
            item: 数据索引
            
        返回:
            (Jones矩阵, 掩码)元组
        """
        spec = self.spectra[item]
        if self.transform is not None:
            spec, mask = self.transform(spec)
        else:
            raise ValueError('Mask has not been generated!')
        return spec, mask

    @staticmethod
    def collate_fn(batch):
        """
        批次整理函数
        
        将多个样本整理成批次。
        
        参数:
            batch: 样本列表
            
        返回:
            (批次Jones矩阵, 批次掩码)元组
        """
        spectra, mask = tuple(zip(*batch))
        spectra = torch.stack(spectra, dim=0)
        mask = torch.stack(mask, dim=0)
        return spectra, mask


def read_split_data(config):
    """
    读取和合并分割的数据文件
    
    由于Jones矩阵数据量大，数据被分割成多个文件存储。
    此函数读取所有分割文件并合并成完整的数据集。
    
    参数:
        config: 配置对象，包含数据路径和分割信息
        
    返回:
        (参数字典, Jones矩阵字典)元组
        - 参数字典：{索引: 结构参数数组}
        - Jones矩阵字典：{索引: Jones矩阵数组}
    """
    # 构建参数文件路径
    param_path = config.DATA.PATH + config.DATA.FOLDER_NAME + config.DATA.PREFIX_PARAM + config.DATA.SUFFIX
    # 构建所有Jones矩阵文件路径
    JM_paths = [config.DATA.PATH + config.DATA.FOLDER_NAME + config.DATA.PREFIX_JM + str(i) + config.DATA.SUFFIX
                for i in range(config.DATA.DIVIDE_NUM)]
    
    # 检查文件是否存在
    assert os.path.exists(param_path), "dataset root: {} does not exist.".format(param_path)
    for JM_path in JM_paths:
        assert os.path.exists(JM_path), "dataset root: {} does not exist.".format(JM_path)

    # 读取参数数据
    para = np.loadtxt(param_path, dtype='float16')  # shape: [total_num, 6]
    total = para.shape[0]
    batch = total // config.DATA.DIVIDE_NUM
    lenx, leny = config.DATA.SIZE_X, config.DATA.SIZE_Y
    
    # 初始化Jones矩阵数组
    JM = np.zeros((total, lenx * leny), dtype='float16')
    
    # 逐个读取并合并分割的Jones矩阵文件
    for num in range(config.DATA.DIVIDE_NUM):
        if num == config.DATA.DIVIDE_NUM-1:
            # 最后一个文件可能包含不完整的批次
            JM[num*batch:, :] = np.loadtxt(JM_paths[num], dtype='float16')
        else:
            JM[num*batch:(num+1)*batch, :] = np.loadtxt(JM_paths[num], dtype='float16')

    # 重塑Jones矩阵：[total, lenx*leny] -> [total, 1, lenx, leny]
    JM = JM.reshape((total, lenx, leny))
    JM = np.expand_dims(JM, axis=1)  # 添加通道维度
    
    # 转换为字典格式（索引 -> 数据）
    para_with_index = dict((k, v) for k, v in enumerate(para))
    JM_with_index = dict((k, v) for k, v in enumerate(JM))
    
    return para_with_index, JM_with_index


def build_loader_simmim(config, logger):
    """
    构建SimMIM预训练的数据加载器
    
    创建用于预训练的DataLoader，支持分布式训练。
    
    参数:
        config: 配置对象
        logger: 日志记录器
        
    返回:
        配置好的DataLoader
    """
    # 创建数据转换器
    transform = DataTransform(config)
    logger.info(f'Pre-train data transform:\n{transform}')
    
    # 检查所有数据文件是否存在
    check_all_data(config)
    
    # 读取训练数据（只使用Jones矩阵，不使用参数）
    _, train_JM = read_split_data(config)
    
    # 创建数据集
    dataset = MyDataSet(spectra=train_JM, transform=transform)
    logger.info(f'Build dataset: train images = {len(dataset)}')

    # 创建数据加载器
    if torch.cuda.device_count() != 1:
        # 分布式训练：使用DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
        dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=True, drop_last=True, collate_fn=dataset.collate_fn)
    else:
        # 单GPU训练：直接使用shuffle
        dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, shuffle=True, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=True, drop_last=True, collate_fn=dataset.collate_fn)
    
    return dataloader


def check_all_data(config):
    """
    检查所有数据文件的存在性和一致性
    
    验证：
    1. 所有指定的数据文件夹是否存在
    2. 不同文件夹中的数据波长点数是否一致
    
    参数:
        config: 配置对象
        
    抛出:
        ValueError: 如果数据文件不存在或波长点数不一致
    """
    num_set = set()
    
    # 检查每个数据文件夹
    for i in range(config.DATA.DATA_SIZE):
        path = config.DATA.PATH + f'training_data_{i+config.DATA.DATA_START}/params_from_preprocess.txt'
        if not os.path.exists(path):
            raise ValueError(f'Path: {path} doesnt exist, so DATA_SIZE & DATA_START setting is unmatched with '
                             f'actual number of groups of data')
        
        # 读取并检查波长点数
        with open(path, 'r') as file:
            params_list = file.read().split()
        index = params_list.index("DATA.SIZE_X") + 1
        size_x = int(params_list[index])
        num_set.add(size_x)
        
        # 确保所有数据的波长点数一致
        if len(num_set) > 1:
            raise ValueError('data in different folders have different wavelength points')
