"""
AirSim API 数据获取模块

功能：
- 从AirSim API直接获取深度图和位姿数据
- 不使用神经网络估计，使用真实数据
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import airsim
import numpy as np


@dataclass
class AirSimData:
    """从AirSim API获取的数据"""
    timestamp: float
    depth_image: Optional[np.ndarray] = None
    rgb_image: Optional[np.ndarray] = None
    position: Optional[np.ndarray] = None  # [x, y, z]
    orientation: Optional[np.ndarray] = None  # 四元数 [w, x, y, z]
    linear_velocity: Optional[np.ndarray] = None  # [vx, vy, vz]
    angular_velocity: Optional[np.ndarray] = None  # [wx, wy, wz]


class AirSimDataExtractor:
    """
    AirSim API数据提取器
    
    直接从AirSim获取：
    - 深度图（真实深度）
    - RGB图像
    - 无人机位姿和速度
    """
    
    def __init__(
        self,
        client: Optional[airsim.MultirotorClient] = None,
        camera_name: str = "0",
        image_type: airsim.ImageType = airsim.ImageType.DepthPlanar,
    ):
        """
        初始化数据提取器
        
        Args:
            client: AirSim客户端（如果为None则创建新连接）
            camera_name: 摄像头名称
            image_type: 图像类型
        """
        self.camera_name = camera_name
        self.image_type = image_type
        
        if client is None:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
        else:
            self.client = client
    
    def get_depth_image(self) -> np.ndarray:
        """
        获取深度图
        
        Returns:
            深度图（米），H x W
        """
        response = self.client.simGetImages([
            airsim.ImageRequest(
                self.camera_name,
                self.image_type,
                pixels_as_float=True,
                compress=False
            )
        ])
        
        if response and response[0].image_data_float:
            depth_image = np.array(response[0].image_data_float, dtype=np.float32)
            depth_image = depth_image.reshape(
                response[0].height, response[0].width
            )
            return depth_image
        else:
            raise ValueError("Failed to get depth image")
    
    def get_rgb_image(self) -> np.ndarray:
        """
        获取RGB图像
        
        Returns:
            RGB图像，H x W x 3
        """
        response = self.client.simGetImages([
            airsim.ImageRequest(
                self.camera_name,
                airsim.ImageType.Scene,
                pixels_as_float=False,
                compress=True
            )
        ])
        
        if response and response[0].image_data_uint8:
            img1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response[0].height, response[0].width, 3)
            return img_rgb
        else:
            raise ValueError("Failed to get RGB image")
    
    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取无人机位姿
        
        Returns:
            (position, orientation): 
                - position: [x, y, z] 位置（NED坐标系）
                - orientation: [w, x, y, z] 四元数
        """
        state = self.client.getMultirotorState()
        kinematics = state.kinematics_estimated
        
        position = np.array([
            kinematics.position.x_val,
            kinematics.position.y_val,
            kinematics.position.z_val
        ])
        
        orientation = np.array([
            kinematics.orientation.w_val,
            kinematics.orientation.x_val,
            kinematics.orientation.y_val,
            kinematics.orientation.z_val
        ])
        
        return position, orientation
    
    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取无人机速度
        
        Returns:
            (linear_velocity, angular_velocity):
                - linear_velocity: [vx, vy, vz] 线性速度
                - angular_velocity: [wx, wy, wz] 角速度
        """
        state = self.client.getMultirotorState()
        kinematics = state.kinematics_estimated
        
        linear = np.array([
            kinematics.linear_velocity.x_val,
            kinematics.linear_velocity.y_val,
            kinematics.linear_velocity.z_val
        ])
        
        angular = np.array([
            kinematics.angular_velocity.x_val,
            kinematics.angular_velocity.y_val,
            kinematics.angular_velocity.z_val
        ])
        
        return linear, angular
    
    def get_all_data(self) -> AirSimData:
        """获取所有数据"""
        import time
        timestamp = time.time()
        
        try:
            depth = self.get_depth_image()
        except:
            depth = None
        
        try:
            rgb = self.get_rgb_image()
        except:
            rgb = None
        
        try:
            position, orientation = self.get_pose()
        except:
            position = None
            orientation = None
        
        try:
            linear_vel, angular_vel = self.get_velocity()
        except:
            linear_vel = None
            angular_vel = None
        
        return AirSimData(
            timestamp=timestamp,
            depth_image=depth,
            rgb_image=rgb,
            position=position,
            orientation=orientation,
            linear_velocity=linear_vel,
            angular_velocity=angular_vel,
        )


class MultiCameraAirSimExtractor:
    """
    多摄像头AirSim数据提取器
    
    同时获取多个摄像头的深度图
    """
    
    CAMERA_NAMES = {
        "front": "front_camera",
        "back": "back_camera", 
        "left": "left_camera",
        "right": "right_camera",
    }
    
    def __init__(
        self,
        client: Optional[airsim.MultirotorClient] = None,
        active_cameras: Optional[List[str]] = None,
    ):
        """
        初始化多摄像头提取器
        
        Args:
            client: AirSim客户端
            active_cameras: 激活的摄像头列表 ["front", "back", "left", "right"]
        """
        if client is None:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
        else:
            self.client = client
        
        self.active_cameras = active_cameras or list(self.CAMERA_NAMES.keys())
        self.extractors = {
            name: AirSimDataExtractor(client=self.client, camera_name=self.CAMERA_NAMES[name])
            for name in self.active_cameras
            if name in self.CAMERA_NAMES
        }
    
    def get_all_depths(self) -> Dict[str, np.ndarray]:
        """获取所有摄像头的深度图"""
        return {
            name: extractor.get_depth_image()
            for name, extractor in self.extractors.items()
        }
    
    def get_all_data(self) -> Dict[str, AirSimData]:
        """获取所有摄像头的数据"""
        return {
            name: extractor.get_all_data()
            for name, extractor in self.extractors.items()
        }
    
    def get_pose_and_velocity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """获取位姿和速度（所有摄像头共享）"""
        state = self.client.getMultirotorState()
        kinematics = state.kinematics_estimated
        
        position = np.array([
            kinematics.position.x_val,
            kinematics.position.y_val,
            kinematics.position.z_val
        ])
        
        orientation = np.array([
            kinematics.orientation.w_val,
            kinematics.orientation.x_val,
            kinematics.orientation.y_val,
            kinematics.orientation.z_val
        ])
        
        linear_vel = np.array([
            kinematics.linear_velocity.x_val,
            kinematics.linear_velocity.y_val,
            kinematics.linear_velocity.z_val
        ])
        
        angular_vel = np.array([
            kinematics.angular_velocity.x_val,
            kinematics.angular_velocity.y_val,
            kinematics.angular_velocity.z_val
        ])
        
        return position, orientation, linear_vel, angular_vel
