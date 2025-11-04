"""
日志管理模块
该文件用于创建和配置日志记录器，支持分布式训练环境下的日志管理。
提供带颜色的控制台输出和文件日志记录功能。
"""

import os
import sys
import logging
import functools
from termcolor import colored


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    """
    创建日志记录器
    
    该函数创建一个配置好的日志记录器，支持控制台和文件两种输出方式。
    使用LRU缓存避免重复创建相同的logger。
    
    参数:
        output_dir: 日志文件输出目录
        dist_rank: 分布式训练中的进程排名，默认为0（主进程）
        name: 日志记录器的名称，默认为空字符串
        
    返回:
        配置好的日志记录器对象
    """
    # 创建logger对象
    logger = logging.getLogger(name)
    # 设置日志级别为DEBUG，记录所有级别的日志
    logger.setLevel(logging.DEBUG)
    # 禁止日志传播到父logger
    logger.propagate = False

    # 创建日志格式化器
    # 基础格式：时间、名称、文件名、行号、日志级别、消息
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    # 彩色格式：为控制台输出添加颜色（注意：某些环境下颜色可能不显示）
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # 为主进程创建控制台处理器
    # 只有主进程（rank 0）才输出到控制台，避免分布式训练时的重复输出
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # 创建文件处理器
    # 每个进程都会创建自己的日志文件，文件名包含进程排名
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger



