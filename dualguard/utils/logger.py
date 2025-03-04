import os

import logging
from dualguard.utils.configs import LogArgs


def create_logger(log_args: LogArgs) -> logging.Logger:
    """
    设置单独的logger

    参数:
        name: 日志记录器的名称
        log_file: 日志文件的路径
        level: 日志级别
    """
    file_name=log_args.log_file_name
    log_file_path = os.path.join(log_args.log_dir, file_name)
    # 创建Logger
    logger = logging.getLogger(file_name)
    logger.setLevel(log_args.log_level)

    # 创建Handler并设置输出的文件
    handler = logging.FileHandler(log_file_path,mode=log_args.mode)
    handler.setLevel(log_args.log_level)

    # 设置日志格式
    formatter = logging.Formatter(log_args.format)
    handler.setFormatter(formatter)

    # 添加Handler到Logger
    logger.addHandler(handler)
    logger.propagate = False

    return logger