import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple

from neural_processing.neural_perception import NeuralOutput

class ObstacleProcessor:
    """
    障碍物极坐标映射与历史融合模块
    负责将神经网络生成的深度图转化为控制层可用的扇区障碍物信息
    """
    
    def __init__(self, 
                 num_angles: int = 360, 
                 max_distance: float = 10.0, 
                 fov_horizontal: float = 90.0,
                 history_size: int = 5,
                 decay_factor: float = 0.7):
        """
        Args:
            num_angles: 全周向离散角度数（通常360对应1度1格）
            max_distance: 最大有效观测距离
            fov_horizontal: 单个摄像头的水平视角
            history_size: 时间序列缓冲区大小
            decay_factor: 历史数据融合权重权重 (0-1)
        """
        self.num_angles = num_angles
        self.max_distance = max_distance
        self.fov_horizontal = fov_horizontal
        self.history_size = history_size
        self.decay_factor = decay_factor
        
        # 极坐标缓冲区：存储最近几帧的距离数据
        self.polar_buffer = deque(maxlen=history_size)
        
        # 预计算角度索引映射（假设相机水平居中）
        self.angle_res = fov_horizontal / num_angles # 如果num_angles是360，这里可以覆盖全周，但目前仅关心FOV内
        
        print(f"ObstacleProcessor初始化: FOV={fov_horizontal}, 缓冲区={history_size}")

    def update(self, neural_output: NeuralOutput, intrinsics: Dict):
        """
        处理新感知的帧并融合历史
        
        Returns:
            ObstaclePolarFrame: 完整的极坐标帧对象
        """
        # 1. 从当前神经网络输出中提取极坐标信息
        current_polar = self._extract_from_depth(neural_output, intrinsics)
        
        # 2. 推入缓冲区（即使是无效帧也推入，维持时序）
        self.polar_buffer.append(current_polar)
        
        # 3. 历史融合
        fused_distances = self._fuse_temporal()
        
        # 4. 获取最近障碍物信息
        closest_depth, closest_angle = self.get_closest_obstacle(fused_distances)
        
        # 5. 构造角度数组
        angles = np.linspace(-np.pi, np.pi, self.num_angles, endpoint=False).astype(np.float32)
        
        # 6. 创建安全掩码
        safety_mask = self.get_safety_zones(fused_distances)
        
        # 7. 返回完整的 ObstaclePolarFrame 对象
        from .types import ObstaclePolarFrame
        
        return ObstaclePolarFrame(
            angles=angles,
            depths=fused_distances.astype(np.float32),
            safety_mask=safety_mask.astype(bool),
            forbidden_mask=np.zeros(self.num_angles, dtype=bool),
            closest_angle=closest_angle,
            closest_depth=float(closest_depth),
            coverage_ratio=float(np.mean(fused_distances < np.inf)),
            warnings=[]
        )

    def _extract_from_depth(self, neural_output: NeuralOutput, intrinsics: Dict) -> np.ndarray:
        """
        Extract obstacle distribution from depth map.

        Args:
            neural_output: NeuralOutput containing depth_t
            intrinsics: Camera intrinsics {'fx', 'cx'}

        Returns:
            distances: Polar distance array [num_angles]
        """
        # 初始化为无穷大
        distances = np.full(self.num_angles, np.inf)

        # 检查深度图有效性
        depth_image = neural_output.depth_maps.get('depth_t')
        if depth_image is None or neural_output.quality_metrics.get('overall_confidence', 0) < 0.1:
            return distances  # 返回全无穷大的空帧

        height, width = depth_image.shape
        fx = intrinsics['fx']
        cx = intrinsics['cx']

        # 参考用户提供的逻辑：取中间多行进行鲁棒采样
        # 增加垂直采样覆盖，防止无人机俯仰时丢失障碍物
        scan_rows = np.linspace(height * 0.35, height * 0.65, 7).astype(int)

        for v in scan_rows:
            for u in range(0, width, 2):  # 步长为2，加速计算
                depth = depth_image[v, u]

                if 0 < depth <= self.max_distance:
                    # 计算相对于相机光轴的角度 (弧度)
                    angle_rad = np.arctan((u - cx) / fx)
                    angle_deg = np.degrees(angle_rad)

                    # 修正角度映射：将 FOV 范围 (-fov/2 ~ fov/2) 映射到数组索引
                    # 假设 num_angles=360, fov=90, 则 -45~45 度映射到 90~270 索引 (前方180度)
                    fov_half = self.fov_horizontal / 2.0
                    if -fov_half <= angle_deg <= fov_half:
                        # 归一化到 [0, 1] 范围
                        normalized_angle = (angle_deg + fov_half) / self.fov_horizontal
                        # 映射到数组索引 (前方扇区)
                        front_sector_start = (self.num_angles - int(self.num_angles * self.fov_horizontal / 360.0)) // 2
                        front_sector_size = int(self.num_angles * self.fov_horizontal / 360.0)
                        final_idx = front_sector_start + int(normalized_angle * front_sector_size)
                        final_idx = np.clip(final_idx, 0, self.num_angles - 1)

                        if depth < distances[final_idx]:
                            distances[final_idx] = depth

        return distances

    def _fuse_temporal(self) -> np.ndarray:
        """
        多帧时序融合：使用加权最小滤波器（障碍物取最近原则 + 指数衰减）
        """
        if not self.polar_buffer:
            return np.full(self.num_angles, np.inf)
        
        # 初始化为最后一帧
        fused = np.array(self.polar_buffer[-1])
        
        # 从新到旧进行融合
        weight = 1.0
        for i in range(len(self.polar_buffer) - 2, -1, -1):
            weight *= self.decay_factor
            prev_frame = self.polar_buffer[i]
            
            # 对于每个角度，障碍物距离取当前和历史的加权考虑
            # 逻辑：如果历史帧有更近的物体，且距离当前帧不远，则保留（防止闪烁）
            mask = prev_frame < fused
            fused[mask] = prev_frame[mask] * (1 - weight) + fused[mask] * weight
            
        return fused

    def get_closest_obstacle(self, distances: np.ndarray) -> Tuple[float, float]:
        """
        Get closest obstacle distance and angle.

        Args:
            distances: Polar distance array

        Returns:
            (distance, angle_deg): Closest obstacle info
        """
        min_dist = np.min(distances)
        if min_dist == np.inf:
            return self.max_distance, 0.0

        min_idx = np.argmin(distances)
        # 转换为相对角度 (0度为正前方)
        angle = (min_idx / self.num_angles) * 360.0
        if angle > 180:
            angle -= 360

        return float(min_dist), float(angle)

    def get_safety_zones(self, distances: np.ndarray, safety_dist: float = 2.0) -> np.ndarray:
        """
        Generate safety mask for obstacles within safety distance.

        Args:
            distances: Polar distance array
            safety_dist: Safety threshold distance

        Returns:
            safety_mask: Boolean mask for safe zones
        """
        return distances > safety_dist
