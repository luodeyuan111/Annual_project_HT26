"""
AirSim API 数据适配器

功能：
- 将AirSim API获取的原始数据转换为VisualState格式
- 直接从深度图提取障碍物信息
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from Visual_process.types import EgoMotion, ObstaclePolarFrame, VisualState
except ImportError:
    from src.Visual_process.types import EgoMotion, ObstaclePolarFrame, VisualState


@dataclass
class CameraIntrinsics:
    """相机内参"""
    fx: float = 320.0
    fy: float = 320.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
        }


class DepthToObstacleConverter:
    """
    从深度图直接提取障碍物极坐标信息
    不依赖ObstacleProcessor，直接实现
    """
    
    def __init__(self, num_angles: int = 72, max_distance: float = 20.0):
        self.num_angles = num_angles
        self.max_distance = max_distance
        self.angles = np.linspace(-np.pi, np.pi, num_angles, endpoint=False).astype(np.float32)
    
    def process(self, depth_image: np.ndarray, intrinsics: Dict) -> ObstaclePolarFrame:
        """
        从深度图提取障碍物信息
        
        Args:
            depth_image: 深度图 H x W（米）
            intrinsics: 相机内参 {'fx', 'fy', 'cx', 'cy'}
            
        Returns:
            ObstaclePolarFrame
        """
        fx = intrinsics.get('fx', 320.0)
        fy = intrinsics.get('fy', 320.0)
        cx = intrinsics.get('cx', 320.0)
        cy = intrinsics.get('cy', 240.0)
        
        height, width = depth_image.shape
        
        # AirSim深度图处理：过滤掉极大的值（表示无限远）
        # 使用百分位数来确定"空旷"阈值
        depth_flat = depth_image.flatten()
        valid_depths = depth_flat[depth_flat < 1000]  # 过滤掉异常大的值
        if len(valid_depths) > 0:
            far_threshold = np.percentile(valid_depths, 95)  # 95百分位作为远边界
            far_threshold = max(far_threshold, 50)  # 至少50米
        else:
            far_threshold = 50
        
        # 深度裁剪
        depth_clipped = np.clip(depth_image, 0.1, far_threshold)
        
        # 选取中间几行（无人机安全飞行高度），而不是全列最小值
        # 这样可以避免检测到地面
        scan_rows = np.linspace(height * 0.45, height * 0.55, 3).astype(int)
        
        # 对每一列，在安全高度范围内取最小深度
        min_depth_per_col = np.min(depth_clipped[scan_rows, :], axis=0)
        
        # 初始化距离数组
        distances = np.full(self.num_angles, self.max_distance, dtype=np.float32)
        
        # 障碍物阈值：使用动态阈值
        OBSTACLE_THRESHOLD = far_threshold * 0.8  # 小于80%的"远边界"才认为是障碍物
        
        # 对每一列计算角度并填充
        for u in range(width):
            depth = min_depth_per_col[u]
            
            # 只有当深度小于阈值时才认为是有效障碍物
            if 0.1 <= depth < OBSTACLE_THRESHOLD:
                # 简单线性映射：列0 -> -π，列中间 -> 0，列最后 -> π
                u_normalized = u / (width - 1) if width > 1 else 0.5
                angle_rad = (u_normalized * 2 - 1) * np.pi  # 映射到 [-π, π]
                
                idx = int((angle_rad + np.pi) / (2 * np.pi) * (self.num_angles - 1))
                idx = np.clip(idx, 0, self.num_angles - 1)
                
                if depth < distances[idx]:
                    distances[idx] = depth
        
        # 计算最近障碍物
        valid_mask = distances < self.max_distance
        if np.any(valid_mask):
            closest_idx = np.argmin(distances)
            closest_angle = float(np.degrees(self.angles[closest_idx]))
            closest_depth = float(distances[closest_idx])
        else:
            closest_angle = 0.0
            closest_depth = self.max_distance
        
        # 安全区域
        safety_mask = distances < 5.0
        
        return ObstaclePolarFrame(
            angles=self.angles,
            depths=distances,
            safety_mask=safety_mask,
            forbidden_mask=np.zeros(self.num_angles, dtype=bool),
            closest_angle=closest_angle,
            closest_depth=closest_depth,
            coverage_ratio=float(np.mean(valid_mask)),
            n_clusters=0,
            warnings=[],
        )


class AirSimVisualAdapter:
    """
    AirSim数据到VisualState的适配器
    """
    
    def __init__(
        self,
        num_angles: int = 72,
        max_distance: float = 20.0,
        intrinsics: Optional[CameraIntrinsics] = None,
    ):
        self.num_angles = num_angles
        self.max_distance = max_distance
        self.intrinsics = intrinsics or CameraIntrinsics()
        
        self.depth_converter = DepthToObstacleConverter(
            num_angles=num_angles,
            max_distance=max_distance,
        )
        
        self.last_position = None
    
    def process_depth_only(
        self,
        depth_image: np.ndarray,
        timestamp: float = 0.0,
    ) -> VisualState:
        intrinsics_dict = self.intrinsics.to_dict()
        obstacle_frame = self.depth_converter.process(depth_image, intrinsics_dict)
        
        return VisualState(
            timestamp=timestamp,
            frame_idx=-1,
            ego_motion=EgoMotion(),
            obstacle_frame=obstacle_frame,
            quality={'api_confidence': 1.0},
        )
    
    def process_with_pose(
        self,
        depth_image: np.ndarray,
        position: np.ndarray,
        orientation: np.ndarray,
        linear_velocity: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
        frame_idx: int = -1,
    ) -> VisualState:
        intrinsics_dict = self.intrinsics.to_dict()
        obstacle_frame = self.depth_converter.process(depth_image, intrinsics_dict)
        
        if linear_velocity is not None:
            velocity = linear_velocity
        else:
            velocity = np.zeros(3)
        
        ego_motion = EgoMotion(
            translation=position.astype(np.float32),
            velocity=velocity.astype(np.float32),
            confidence=1.0,
            inlier_ratio=1.0,
            n_inliers=1,
        )
        
        return VisualState(
            timestamp=timestamp,
            frame_idx=frame_idx,
            ego_motion=ego_motion,
            obstacle_frame=obstacle_frame,
            quality={'api_confidence': 1.0},
        )
    
    def process_dual_frame(
        self,
        depth_t: np.ndarray,
        depth_t1: np.ndarray,
        position_t: np.ndarray,
        position_t1: np.ndarray,
        orientation_t: np.ndarray,
        orientation_t1: np.ndarray,
        linear_vel_t: Optional[np.ndarray] = None,
        linear_vel_t1: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
        frame_idx: int = -1,
    ) -> VisualState:
        intrinsics_dict = self.intrinsics.to_dict()
        
        obstacle_frame = self.depth_converter.process(depth_t1, intrinsics_dict)
        
        if linear_vel_t1 is not None:
            velocity = linear_vel_t1
        else:
            velocity = position_t1 - position_t
        
        ego_motion = EgoMotion(
            translation=(position_t1 - position_t).astype(np.float32),
            velocity=velocity.astype(np.float32),
            confidence=1.0,
            inlier_ratio=1.0,
            n_inliers=1,
        )
        
        return VisualState(
            timestamp=timestamp,
            frame_idx=frame_idx,
            ego_motion=ego_motion,
            obstacle_frame=obstacle_frame,
            quality={'api_confidence': 1.0},
        )


class MultiCameraAirSimAdapter:
    """
    多摄像头AirSim数据适配器
    """
    
    CAMERA_ANGLES = {
        "front_camera": 0,
        "right_camera": 90,
        "back_camera": 180,
        "left_camera": -90,
    }
    
    def __init__(
        self,
        num_angles: int = 72,
        max_distance: float = 20.0,
        intrinsics: Optional[CameraIntrinsics] = None,
    ):
        self.num_angles = num_angles
        self.max_distance = max_distance
        self.intrinsics = intrinsics or CameraIntrinsics()
        
        self.depth_converter = DepthToObstacleConverter(
            num_angles=num_angles,
            max_distance=max_distance,
        )
    
    def process_multi_camera(
        self,
        depth_images: Dict[str, np.ndarray],
        position: np.ndarray,
        linear_velocity: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
        frame_idx: int = -1,
    ) -> VisualState:
        intrinsics_dict = self.intrinsics.to_dict()
        
        # 融合360度空间
        fused_depths = np.full(self.num_angles, self.max_distance, dtype=np.float32)
        
        # 分别处理每个摄像头并融合
        for camera_name, depth_image in depth_images.items():
            if depth_image is not None:
                # 获取该摄像头的角度偏移
                angle_offset = self.CAMERA_ANGLES.get(camera_name, 0)
                
                # 处理深度图
                single_obs = self.depth_converter.process(depth_image, intrinsics_dict)
                
                # 将该摄像头的角度映射到全局360度
                for i, (angle, depth) in enumerate(zip(single_obs.angles, single_obs.depths)):
                    global_angle = angle + np.radians(angle_offset)
                    # 归一化到[-π, π]
                    while global_angle > np.pi:
                        global_angle -= 2 * np.pi
                    while global_angle < -np.pi:
                        global_angle += 2 * np.pi
                    
                    # 映射到索引
                    global_idx = int((global_angle + np.pi) / (2 * np.pi) * (self.num_angles - 1))
                    global_idx = np.clip(global_idx, 0, self.num_angles - 1)
                    
                    if depth < fused_depths[global_idx]:
                        fused_depths[global_idx] = depth
        
        # 计算最近障碍物
        valid_mask = fused_depths < self.max_distance
        if np.any(valid_mask):
            closest_idx = np.argmin(fused_depths)
            closest_angle = float(np.degrees(self.depth_converter.angles[closest_idx]))
            closest_depth = float(fused_depths[closest_idx])
        else:
            closest_angle = 0.0
            closest_depth = self.max_distance
        
        obstacle_frame = ObstaclePolarFrame(
            angles=self.depth_converter.angles,
            depths=fused_depths,
            safety_mask=fused_depths < 5.0,
            forbidden_mask=np.zeros(self.num_angles, dtype=bool),
            closest_angle=closest_angle,
            closest_depth=closest_depth,
            coverage_ratio=float(np.mean(valid_mask)),
            n_clusters=0,
            warnings=[],
        )
        
        if linear_velocity is not None:
            velocity = linear_velocity
        else:
            velocity = np.zeros(3)
        
        ego_motion = EgoMotion(
            translation=position.astype(np.float32),
            velocity=velocity.astype(np.float32),
            confidence=1.0,
            inlier_ratio=1.0,
            n_inliers=1,
        )
        
        return VisualState(
            timestamp=timestamp,
            frame_idx=frame_idx,
            ego_motion=ego_motion,
            obstacle_frame=obstacle_frame,
            quality={'api_confidence': 1.0, 'n_cameras': sum(1 for v in depth_images.values() if v is not None)},
        )
