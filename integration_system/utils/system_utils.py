"""
系统工具函数
"""

import os
import json
import yaml
from datetime import datetime

def create_output_dir(base_dir="outputs"):
    """创建输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    # 创建子目录
    subdirs = ['images', 'flow', 'depth', 'segmentation', 'pose', 'logs']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    return output_dir

def save_result(result, output_dir, index=0):
    """保存处理结果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存为JSON
    json_path = os.path.join(output_dir, f"result_{index:04d}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {json_path}")
    return json_path

def load_config(config_path, default_config=None):
    """加载配置文件"""
    if not os.path.exists(config_path):
        print(f"警告: 配置文件不存在: {config_path}")
        return default_config or {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        elif config_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path}")