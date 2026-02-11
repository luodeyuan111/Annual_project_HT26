"""
无人机接口模块
"""

import cv2
import numpy as np
from PIL import Image

class DroneController:
    """无人机控制器"""
    
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.connected = True
        self.frame_count = 0
        print("无人机控制器初始化（模拟模式）")
    
    def capture_frame(self):
        """捕获一帧图像"""
        width, height = 640, 480
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 创建背景
        for y in range(height):
            color = int(100 + 100 * y / height)
            img_array[y, :] = [color, color//2, 255-color]
        
        self.frame_count += 1
        
        # 添加特征
        center_x, center_y = width//2, height//2
        cv2.circle(img_array, (center_x, center_y), 80, (255, 100, 50), -1)
        cv2.rectangle(img_array, (center_x-100, center_y-50), 
                     (center_x+100, center_y+50), (50, 150, 255), 2)
        
        # 添加文本
        cv2.putText(img_array, f"Drone Frame {self.frame_count}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return Image.fromarray(img_array)
    
    def disconnect(self):
        print("无人机控制器断开")

# 导出
__all__ = ['DroneController']
