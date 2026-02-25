"""
可视化工具模块
提供深度图、光流、障碍物极坐标的可视化功能
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict

# 检查PIL是否可用（用于中文显示）
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class Visualizer:
    """视觉处理可视化工具类"""

    def __init__(self, enable_depth: bool = True, enable_obstacle: bool = True, 
                 enable_flow: bool = False, radar_size: int = 200):
        """
        初始化可视化器
        
        Args:
            enable_depth: 是否显示深度图
            enable_obstacle: 是否显示障碍物雷达图
            enable_flow: 是否显示光流
            radar_size: 雷达图大小（像素）
        """
        self.enable_depth = enable_depth
        self.enable_obstacle = enable_obstacle
        self.enable_flow = enable_flow
        self.radar_size = radar_size
        
    def visualize(self, 
                 rgb_image: np.ndarray,
                 depth_map: Optional[np.ndarray] = None,
                 obstacle_frame=None,
                 optical_flow: Optional[np.ndarray] = None,
                 quality_metrics: Optional[Dict] = None) -> np.ndarray:
        """
        生成组合可视化图像
        
        Args:
            rgb_image: RGB原图 [H, W, 3]
            depth_map: 深度图 [H, W]
            obstacle_frame: ObstaclePolarFrame对象
            optical_flow: 光流场 [H, W, 2]
            quality_metrics: 质量指标字典
            
        Returns:
            可视化后的组合图像
        """
        panels = []
        
        # 1. 主图像
        if rgb_image is not None:
            main_panel = rgb_image.copy()
            # 添加信息标注（使用英文避免乱码）
            info_text = []
            if quality_metrics:
                info_text.append(f"Q:{quality_metrics.get('overall_confidence', 0):.2f}")
            if obstacle_frame:
                info_text.append(f"Obs:{obstacle_frame.closest_depth:.1f}m@{obstacle_frame.closest_angle:.0f}d")
            
            if info_text:
                self._add_text_overlay(main_panel, " | ".join(info_text))
            panels.append(main_panel)
        
        # 2. 深度图
        if self.enable_depth and depth_map is not None:
            depth_panel = self.visualize_depth(depth_map)
            depth_panel = self._add_text_overlay(depth_panel, "Depth")
            panels.append(depth_panel)
        
        # 3. 障碍物雷达图
        if self.enable_obstacle and obstacle_frame is not None:
            radar_panel = self.visualize_obstacle_radar(obstacle_frame)
            panels.append(radar_panel)
        
        # 4. 光流图
        if self.enable_flow and optical_flow is not None:
            flow_panel = self.visualize_flow(optical_flow, rgb_image)
            flow_panel = self._add_text_overlay(flow_panel, "光流")
            panels.append(flow_panel)
        
        # 组合所有面板
        if not panels:
            return rgb_image
            
        return self._combine_panels(panels)
    
    def visualize_depth(self, depth_map: np.ndarray, colormap: int = cv2.COLORMAP_JET,
                        max_depth_clip: float = 20.0) -> np.ndarray:
        """
        可视化深度图
        
        Args:
            depth_map: 深度图 [H, W]
            colormap: OpenCV色彩映射类型
            max_depth_clip: 最大深度裁剪值（米），超过此值的显示为最远
            
        Returns:
            彩色深度图 [H, W, 3] BGR
        """
        # 裁剪深度值，过滤掉不合理的深度（如天空的极大值）
        depth_clipped = np.clip(depth_map, 0.1, max_depth_clip)
        
        # 归一化到0-255（近=红色，远=蓝色）
        depth_normalized = (depth_clipped - 0.1) / (max_depth_clip - 0.1)
        depth_normalized = np.clip(depth_normalized, 0, 1)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        
        # 色彩映射
        color_depth = cv2.applyColorMap(depth_uint8, colormap)
        
        return color_depth
    
    def visualize_obstacle_radar(self, obstacle_frame, 
                                  max_distance: Optional[float] = None) -> np.ndarray:
        """
        可视化障碍物极坐标（雷达图样式）
        
        Args:
            obstacle_frame: ObstaclePolarFrame对象
            max_distance: 雷达最大距离（米），默认使用障碍物最大值
            
        Returns:
            雷达图图像
        """
        angles = obstacle_frame.angles  # [-π, π]
        depths = obstacle_frame.depths  # 距离数组
        
        if max_distance is None:
            max_distance = np.max(depths[np.isfinite(depths)]) if np.any(np.isfinite(depths)) else 10.0
        max_distance = max(max_distance, 1.0)  # 至少1米
        
        # 创建雷达图背景
        size = self.radar_size
        center = (size // 2, size // 2)
        radius = size // 2 - 10
        
        radar = np.ones((size, size, 3), dtype=np.uint8) * 30  # 深灰色背景
        radar = cv2.ellipse(radar, center, (radius, radius), 0, 0, 360, (50, 50, 50), 1)
        
        # 画同心圆（距离刻度）
        for r in [0.25, 0.5, 0.75, 1.0]:
            cv2.circle(radar, center, int(radius * r), (80, 80, 80), 1)
        
        # 画方向线
        for angle_deg in [0, 90, 180, 270]:
            angle_rad = np.radians(angle_deg - 90)  # 正上方为0度
            x = int(center[0] + radius * np.cos(angle_rad))
            y = int(center[1] + radius * np.sin(angle_rad))
            cv2.line(radar, center, (x, y), (80, 80, 80), 1)
        
        # 绘制障碍物轮廓线（无角度偏移）
        ANGLE_OFFSET = 0
        
        # 先计算所有点的坐标
        points = []
        for i, (angle, depth) in enumerate(zip(angles, depths)):
            if np.isfinite(depth) and depth > 0.1 and depth <= max_distance:
                angle_deg = np.degrees(angle)
                
                # 添加角度偏移并反转
                radar_angle_deg = -angle_deg + ANGLE_OFFSET
                
                # 归一化到[-180, 180]
                while radar_angle_deg > 180:
                    radar_angle_deg -= 360
                while radar_angle_deg < -180:
                    radar_angle_deg += 360
                
                radar_angle_rad = np.radians(radar_angle_deg)
                
                # 距离归一化
                dist_ratio = depth / max_distance
                dist_ratio = min(dist_ratio, 1.0)
                
                # 有效障碍物的半径放大1.5倍
                if depth < max_distance * 0.95:
                    dist_ratio = min(dist_ratio * 1.5, 1.0)
                
                # 计算像素位置
                px = int(center[0] + radius * dist_ratio * np.sin(radar_angle_rad))
                py = int(center[1] - radius * dist_ratio * np.cos(radar_angle_rad))
                
                if 0 <= px < size and 0 <= py < size:
                    points.append((px, py, depth))
        
        # 画点（增加大小）
        # 使用动态阈值判断是否有障碍物
        no_obstacle_threshold = max_distance * 0.95  # 接近最大距离时视为无障碍
        for px, py, depth in points:
            if depth >= no_obstacle_threshold:
                color = (0, 255, 0)  # 绿色表示无障碍
            else:
                color_ratio = min(depth / (max_distance * 0.7), 1.0)  # 归一化到70%最大距离作为"最近"
                color = (
                    int(255 * color_ratio),
                    int(50),
                    int(255 * (1 - color_ratio))
                )
            cv2.circle(radar, (px, py), 3, color, -1)
        
        # 画线连接相邻点形成轮廓
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(radar, (points[i][0], points[i][1]), 
                        (points[i+1][0], points[i+1][1]), (100, 100, 100), 1)
        
        # 绘制最近障碍物方向（应用角度偏移）
        if obstacle_frame.closest_depth < max_distance and obstacle_frame.closest_depth > 0.1:
            closest_angle_deg = -obstacle_frame.closest_angle + ANGLE_OFFSET
            while closest_angle_deg > 180:
                closest_angle_deg -= 360
            while closest_angle_deg < -180:
                closest_angle_deg += 360
            closest_angle_rad = np.radians(closest_angle_deg)
            cx = int(center[0] + radius * 0.8 * np.sin(closest_angle_rad))
            cy = int(center[1] - radius * 0.8 * np.cos(closest_angle_rad))
            cv2.arrowedLine(radar, center, (cx, cy), (0, 0, 255), 2)
        
        # 添加文字标注（使用英文避免乱码）
        cv2.putText(radar, f"{obstacle_frame.closest_depth:.1f}m", 
                   (5, size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(radar, f"@{obstacle_frame.closest_angle:.0f}d",
                   (5, size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return radar
    
    def visualize_flow(self, flow: np.ndarray, image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        可视化光流
        
        Args:
            flow: 光流场 [H, W, 2]
            image: 背景图像（可选）
            
        Returns:
            光流可视化图像
        """
        # 计算光流幅度
        magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
        angle = np.arctan2(flow[:,:,1], flow[:,:,0])
        
        # 归一化
        mag_max = magnitude.max() if magnitude.max() > 0 else 1.0
        magnitude_norm = magnitude / mag_max
        
        # 转换为HSV
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[:,:,0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
        hsv[:,:,1] = 255
        hsv[:,:,2] = (magnitude_norm * 255).astype(np.uint8)
        
        # 转换到BGR
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 叠加到原图
        if image is not None:
            alpha = 0.6
            flow_bgr = cv2.addWeighted(image, 1-alpha, flow_bgr, alpha, 0)
        
        return flow_bgr
    
    def _add_text_overlay(self, image: np.ndarray, text: str, 
                          position: Tuple[int, int] = (10, 25),
                          font_scale: float = 0.6) -> np.ndarray:
        """在图像上添加文字标注"""
        # 背景框
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(image, 
                     (position[0] - 5, position[1] - text_h - 5),
                     (position[0] + text_w + 5, position[1] + baseline + 5),
                     (0, 0, 0), -1)
        
        # 文字
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        return image
    
    def _combine_panels(self, panels: list, layout: str = 'horizontal') -> np.ndarray:
        """
        组合多个面板
        
        Args:
            panels: 图像列表
            layout: 'horizontal' 或 'vertical'
            
        Returns:
            组合后的图像
        """
        if not panels:
            return None
            
        if layout == 'horizontal':
            # 调整高度一致
            max_height = max(p.shape[0] for p in panels)
            resized = []
            for p in panels:
                if p.shape[0] != max_height:
                    ratio = max_height / p.shape[0]
                    new_width = int(p.shape[1] * ratio)
                    p = cv2.resize(p, (new_width, max_height))
                resized.append(p)
            return np.hstack(resized)
        else:
            # 调整宽度一致
            max_width = max(p.shape[1] for p in panels)
            resized = []
            for p in panels:
                if p.shape[1] != max_width:
                    ratio = max_width / p.shape[1]
                    new_height = int(p.shape[0] * ratio)
                    p = cv2.resize(p, (max_width, new_height))
                resized.append(p)
            return np.vstack(resized)


def create_visualizer(enable_depth: bool = True, enable_obstacle: bool = True,
                      enable_flow: bool = False) -> Visualizer:
    """创建可视化器的便捷函数"""
    return Visualizer(enable_depth, enable_obstacle, enable_flow)
