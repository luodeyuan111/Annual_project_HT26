"""
配置管理模块
用于加载和管理项目配置
"""

import yaml
import os
from typing import Dict, Any, Optional


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.config = self._get_default_config()
            print("使用默认配置")
    
    def load_config(self, config_path: str) -> None:
        """
        从YAML文件加载配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            print(f"配置已加载: {config_path}")
        except Exception as e:
            print(f"加载配置失败: {e}，使用默认配置")
            self.config = self._get_default_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        
        Args:
            key: 配置键，支持点号分隔的嵌套键（如 'neural.device'）
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置项
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, save_path: str) -> None:
        """
        保存配置到文件
        
        Args:
            save_path: 保存路径
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            print(f"配置已保存: {save_path}")
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            # 设备配置
            'device': {
                'device_type': 'cpu',  # cpu 或 cuda
                'use_mixed_precision': False
            },
            
            # 神经网络配置
            'neural': {
                'raft': {
                    'model_path': 'models/RAFT/models/raft-things.pth',
                    'iterations': 20
                },
                'monodepth2': {
                    'model_path': 'models/monodepth2/mono+stereo_640x192',
                    'input_width': 640,
                    'input_height': 192,
                    'depth_min': 0.1,
                    'depth_max': 100.0,
                    'scale_factor': 5.4
                },
                'features': {
                    'grid_step': 8,
                    'flow_threshold': 0.5,
                    'min_features': 50
                },
                'segmentation': {
                    'method': 'kmeans',
                    'n_clusters': 3,
                    'min_region_size': 50
                },
                'quality': {
                    'min_valid_depth_ratio': 0.3,
                    'flow_confidence_threshold': 0.5
                }
            },
            
            # 视觉处理配置
            'vision': {
                'camera': {
                    'fx': 320.0,  # 焦距（需要根据实际摄像头调整）
                    'fy': 320.0,
                    'cx': 320.0,  # 主点（图像中心）
                    'cy': 240.0
                },
                'obstacle': {
                    'num_angles': 360,
                    'max_distance': 10.0,
                    'fov_horizontal': 90.0,
                    'history_size': 5,
                    'decay_factor': 0.7,
                    'safety_distance': 2.0
                },
                'pose': {
                    'min_flow_magnitude': 0.5,
                    'ransac_threshold': 2.0,
                    'min_inliers': 30
                }
            },
            
            # 无人机配置
            'drone': {
                'move_speed': 10.0,
                'camera_name': 'front_camera',
                'save_images': False
            },
            
            # 日志配置
            'logging': {
                'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
                'log_to_file': True,
                'log_dir': 'logs',
                'log_file': 'drone_vision.log'
            },
            
            # 调试配置
            'debug': {
                'save_visualizations': False,
                'vis_dir': 'visualizations',
                'print_debug_info': False
            }
        }


# 使用示例
if __name__ == "__main__":
    # 创建配置管理器
    config = ConfigManager()
    
    # 获取配置
    print(f"设备类型: {config.get('device.device_type')}")
    print(f"RAFT迭代次数: {config.get('neural.raft.iterations')}")
    print(f"最大障碍物距离: {config.get('vision.obstacle.max_distance')}米")
    
    # 设置配置
    config.set('device.device_type', 'cuda')
    print(f"更新后的设备类型: {config.get('device.device_type')}")
    
    # 保存配置
    config.save_config('config/default.yaml')
    print("默认配置已保存到 config/default.yaml")