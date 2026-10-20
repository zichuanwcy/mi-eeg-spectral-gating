"""日志工具模块

提供统一的日志配置：同时输出到控制台和文件。
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = "eeg_experiment",
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    创建并配置 Logger。

    日志同时输出到：
      - 控制台（StreamHandler）
      - 文件（FileHandler），文件名含时间戳

    Parameters
    ----------
    name : str
        Logger 名称（实验名）
    log_dir : str
        日志文件存放目录
    level : int
        日志级别，默认 INFO

    Returns
    -------
    logging.Logger
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # 避免重复添加 handler（多次调用时）
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # 控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件 Handler
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file    = Path(log_dir) / f"{name}_{timestamp}.log"
    file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
