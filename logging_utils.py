"""
日志管理模块
提供统一的日志记录功能
"""

import logging
import os
from datetime import datetime
from typing import Optional


class LoggerManager:
    """日志管理器"""
    
    _instance = None
    _logger = None
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, log_dir: Optional[str] = None, log_file: Optional[str] = None, 
                 log_level: str = 'INFO', log_to_file: bool = True):
        """
        初始化日志管理器
        
        Args:
            log_dir: 日志目录
            log_file: 日志文件名
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: 是否记录到文件
        """
        if self._initialized:
            return
        
        self._initialized = True
        self.log_dir = log_dir or 'logs'
        self.log_file = log_file or 'drone_vision.log'
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_to_file = log_to_file
        
        # 创建日志目录
        if self.log_to_file:
            os.makedirs(self.log_dir, exist_ok=True)
        
        # 配置日志
        self._setup_logger()
    
    def _setup_logger(self):
        """配置日志系统"""
        # 创建logger
        self._logger = logging.getLogger('DroneVision')
        self._logger.setLevel(self.log_level)
        
        # 避免重复添加handler
        if self._logger.handlers:
            return
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        
        # 文件处理器
        if self.log_to_file:
            log_path = os.path.join(self.log_dir, self.log_file)
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
        
        # 记录初始化信息
        self._logger.info("=" * 50)
        self._logger.info("日志系统初始化完成")
        self._logger.info(f"日志级别: {logging.getLevelName(self.log_level)}")
        self._logger.info(f"日志目录: {self.log_dir}")
        self._logger.info(f"日志文件: {self.log_file}")
        self._logger.info("=" * 50)
    
    @property
    def logger(self) -> logging.Logger:
        """获取logger实例"""
        return self._logger
    
    def set_level(self, level: str):
        """
        设置日志级别
        
        Args:
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_level = getattr(logging, level.upper(), logging.INFO)
        self._logger.setLevel(self.log_level)
        for handler in self._logger.handlers:
            handler.setLevel(self.log_level)
        self._logger.info(f"日志级别已更新为: {level}")
    
    def log_processing_time(self, start_time: float, end_time: float, operation: str):
        """
        记录处理时间
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            operation: 操作名称
        """
        duration = end_time - start_time
        self._logger.info(f"{operation} 处理时间: {duration:.3f}秒")
        return duration
    
    def log_frame_info(self, frame_idx: int, features: int, segments: int, quality: float):
        """
        记录帧处理信息
        
        Args:
            frame_idx: 帧索引
            features: 特征点数量
            segments: 分割区域数
            quality: 质量分数
        """
        self._logger.info(
            f"帧 {frame_idx}: 特征点={features}, 分割={segments}, 质量={quality:.3f}"
        )
    
    def log_error(self, error: Exception, context: str = ""):
        """
        记录错误
        
        Args:
            error: 异常对象
            context: 错误上下文
        """
        import traceback
        self._logger.error(f"{context} - {str(error)}")
        self._logger.debug(traceback.format_exc())
    
    def log_warning(self, message: str):
        """
        记录警告
        
        Args:
            message: 警告信息
        """
        self._logger.warning(message)
    
    def log_info(self, message: str):
        """
        记录信息
        
        Args:
            message: 信息内容
        """
        self._logger.info(message)
    
    def log_debug(self, message: str):
        """
        记录调试信息
        
        Args:
            message: 调试信息
        """
        self._logger.debug(message)


# 便捷函数
def get_logger(log_dir: Optional[str] = None, log_file: Optional[str] = None,
               log_level: str = 'INFO', log_to_file: bool = True) -> logging.Logger:
    """
    获取logger实例（便捷函数）
    
    Args:
        log_dir: 日志目录
        log_file: 日志文件名
        log_level: 日志级别
        log_to_file: 是否记录到文件
        
    Returns:
        logger: Logger实例
    """
    manager = LoggerManager(log_dir, log_file, log_level, log_to_file)
    return manager.logger


# 使用示例
if __name__ == "__main__":
    # 获取logger
    logger = get_logger(log_dir='logs', log_file='test.log', log_level='DEBUG')
    
    # 记录不同级别的日志
    logger.debug("这是调试信息")
    logger.info("这是普通信息")
    logger.warning("这是警告信息")
    try:
        raise ValueError("测试错误")
    except Exception as e:
        logger.error(f"发生错误: {e}")
    
    # 记录处理时间
    import time
    start = time.time()
    time.sleep(0.5)
    end = time.time()
    manager = LoggerManager()
    manager.log_processing_time(start, end, "测试操作")
    
    # 记录帧信息
    manager.log_frame_info(1, 100, 3, 0.95)
    
    print("日志测试完成，请查看 logs/ 目录")