"""
AirSim API方案主程序

功能：
- 使用AirSim API直接获取数据（不使用神经网络）
- 生成VisualState用于集群控制
- 支持可视化对比

使用方式：
    python -m alternative_pipeline.main_api
    
或：
    python alternative_pipeline/main_api.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

import time
from datetime import datetime
from typing import Optional

import airsim
import cv2
import numpy as np

from alternative_pipeline.api_data_extractor import (
    AirSimDataExtractor,
    MultiCameraAirSimExtractor,
)
from alternative_pipeline.api_visual_adapter import (
    AirSimVisualAdapter,
    MultiCameraAirSimAdapter,
    CameraIntrinsics,
)

try:
    from src.utils.logging_utils import get_logger
except ImportError:
    import logging
    def get_logger(*args, **kwargs):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(args[0] if args else 'AirSimAPI')


class AirSimAPIPipeline:
    """
    AirSim API方案主流程
    
    特点：
    - 使用AirSim API直接获取真实数据
    - 不依赖神经网络
    - 数据更准确（适合集群控制）
    """
    
    def __init__(
        self,
        client: Optional[airsim.MultirotorClient] = None,
        use_multi_camera: bool = False,
        num_angles: int = 72,
        max_distance: float = 20.0,
    ):
        """
        初始化
        
        Args:
            client: AirSim客户端
            use_multi_camera: 是否使用多摄像头
            num_angles: 极坐标角度数量
            max_distance: 最大感知距离
        """
        self.use_multi_camera = use_multi_camera
        self.logger = get_logger('AirSimAPIPipeline')
        
        if client is None:
            self.logger.info("连接AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
        else:
            self.client = client
        
        if use_multi_camera:
            self.extractor = MultiCameraAirSimExtractor(client=self.client)
            self.adapter = MultiCameraAirSimAdapter(
                num_angles=num_angles,
                max_distance=max_distance,
            )
        else:
            self.extractor = AirSimDataExtractor(client=self.client)
            self.adapter = AirSimVisualAdapter(
                num_angles=num_angles,
                max_distance=max_distance,
            )
        
        self.frame_idx = 0
    
    def capture_and_process(self) -> dict:
        """
        捕获并处理当前帧
        
        Returns:
            包含VisualState和处理信息的字典
        """
        start_time = time.time()
        
        if self.use_multi_camera:
            depth_images = self.extractor.get_all_depths()
            position, orientation, linear_vel, angular_vel = self.extractor.get_pose_and_velocity()
            
            visual_state = self.adapter.process_multi_camera(
                depth_images=depth_images,
                position=position,
                linear_velocity=linear_vel,
                timestamp=time.time(),
                frame_idx=self.frame_idx,
            )
        else:
            depth_image = self.extractor.get_depth_image()
            position, orientation = self.extractor.get_pose()
            linear_vel, angular_vel = self.extractor.get_velocity()
            
            visual_state = self.adapter.process_with_pose(
                depth_image=depth_image,
                position=position,
                orientation=orientation,
                linear_velocity=linear_vel,
                angular_velocity=angular_vel,
                timestamp=time.time(),
                frame_idx=self.frame_idx,
            )
        
        process_time = time.time() - start_time
        self.frame_idx += 1
        
        return {
            'visual_state': visual_state,
            'depth_image': depth_image if not self.use_multi_camera else None,
            'depth_images': depth_images if self.use_multi_camera else None,
            'position': position,
            'process_time': process_time,
        }
    
    def run_continuous(self, interval: float = 0.1):
        """
        持续运行
        
        Args:
            interval: 处理间隔（秒）
        """
        self.logger.info("开始持续运行...")
        self.logger.info(f"多摄像头模式: {self.use_multi_camera}")
        
        try:
            while True:
                result = self.capture_and_process()
                
                vs = result['visual_state']
                self.logger.info("=" * 40)
                self.logger.info(f"Frame {vs.frame_idx} | Time: {vs.timestamp:.2f}")
                self.logger.info(f"Position: {result['position']}")
                
                if vs.obstacle_frame:
                    self.logger.info(
                        f"Obstacle: closest={vs.obstacle_frame.closest_depth:.2f}m "
                        f"@ {vs.obstacle_frame.closest_angle:.1f}deg "
                        f"coverage={vs.obstacle_frame.coverage_ratio:.1%}"
                    )
                
                self.logger.info(f"Process time: {result['process_time']*1000:.1f}ms")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("用户中断")
        except Exception as e:
            self.logger.error(f"错误: {e}")
    
    def run_once(self) -> dict:
        """运行一次"""
        self.logger.info("捕获并处理单帧...")
        result = self.capture_and_process()
        
        vs = result['visual_state']
        self.logger.info("=" * 40)
        self.logger.info(f"Frame {vs.frame_idx}")
        self.logger.info(f"Position: {result['position']}")
        
        if vs.obstacle_frame:
            self.logger.info(
                f"Obstacle: closest={vs.obstacle_frame.closest_depth:.2f}m "
                f"@ {vs.obstacle_frame.closest_angle:.1f}deg"
            )
        
        return result


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AirSim API方案主程序')
    parser.add_argument('--multi-camera', action='store_true', help='使用多摄像头')
    parser.add_argument('--continuous', action='store_true', help='持续运行模式')
    parser.add_argument('--interval', type=float, default=0.5, help='处理间隔（秒）')
    parser.add_argument('--angles', type=int, default=72, help='极坐标角度数量')
    parser.add_argument('--max-distance', type=float, default=50.0, help='最大感知距离')
    parser.add_argument('--interactive', action='store_true', help='交互模式（键盘控制）')
    
    args = parser.parse_args()
    
    # 默认为多摄像头模式（根据settings.json配置）
    pipeline = AirSimAPIPipeline(
        use_multi_camera=True,  # 使用多摄像头
        num_angles=args.angles,
        max_distance=args.max_distance,
    )
    
    # 默认进入交互模式（带键盘控制）
    run_interactive(pipeline)


def run_interactive(pipeline: AirSimAPIPipeline):
    """交互模式：带键盘控制的持续运行"""
    import msvcrt
    import time
    
    print("=" * 60)
    print("交互模式启动")
    print("按键说明:")
    print("  w - 前进")
    print("  s - 后退")
    print("  a - 左移")
    print("  d - 右移")
    print("  , - 上升")
    print("  . - 下降")
    print("  空格 - 停止所有")
    print("  e - 捕获并处理")
    print("  v - 切换可视化")
    print("  q - 退出")
    print("=" * 60)
    
    # 初始化可视化器
    try:
        from src.utils.visualization import Visualizer
        visualizer = Visualizer(enable_depth=True, enable_obstacle=True, enable_flow=False)
        has_visualizer = True
    except ImportError:
        has_visualizer = False
        print("警告: 可视化模块未找到")
    
    show_visualization = False
    
    MOVE_SPEED = 10
    move_keys = {
        119: ('x', MOVE_SPEED),   # w
        115: ('x', -MOVE_SPEED),  # s
        97:  ('y', -MOVE_SPEED),  # a
        100: ('y', MOVE_SPEED),   # d
        44:  ('z', -MOVE_SPEED),  # ,
        46:  ('z', MOVE_SPEED),   # .
    }
    
    active_moves = set()
    
    try:
        while True:
            # 检查键盘
            if msvcrt.kbhit():
                key = ord(msvcrt.getch())
                
                if key == 113:  # q 退出
                    print("退出交互模式")
                    break
                
                elif key == 101:  # e 捕获
                    result = pipeline.capture_and_process()
                    vs = result['visual_state']
                    print(f"Position: {result['position']}")
                    if vs.obstacle_frame:
                        print(f"Obstacle: closest={vs.obstacle_frame.closest_depth:.2f}m @ {vs.obstacle_frame.closest_angle:.1f}deg")
                    
                    # 如果开启可视化，显示
                    if show_visualization and has_visualizer and result.get('depth_images'):
                        depth_img = result['depth_images'].get('front')
                        if depth_img is not None:
                            depth_vis = visualizer.visualize_depth(depth_img)
                            radar_vis = visualizer.visualize_obstacle_radar(vs.obstacle_frame)
                            # 调整雷达图大小匹配深度图
                            radar_resized = cv2.resize(radar_vis, (depth_vis.shape[1], depth_vis.shape[0]))
                            combined = np.hstack([depth_vis, radar_resized])
                            cv2.imshow("API Visualization", combined)
                            cv2.waitKey(1)
                
                elif key == 118:  # v 可视化开关
                    show_visualization = not show_visualization
                    status = "开启" if show_visualization else "关闭"
                    print(f"可视化已{status}")
                    if not show_visualization:
                        cv2.destroyWindow("API Visualization")
                
                elif key == 32:  # 空格 停止
                    active_moves.clear()
                    pipeline.client.moveByVelocityAsync(0, 0, 0, 0.1)
                    print("停止所有运动")
                
                elif key in move_keys:
                    axis, value = move_keys[key]
                    if key in active_moves:
                        active_moves.discard(key)
                    else:
                        active_moves.add(key)
            
            # 处理运动
            if active_moves:
                vx, vy, vz = 0, 0, 0
                for k in active_moves:
                    axis, value = move_keys[k]
                    if axis == 'x':
                        vx = value
                    elif axis == 'y':
                        vy = value
                    elif axis == 'z':
                        vz = value
                pipeline.client.moveByVelocityAsync(vx, vy, vz, 0.05)
            
            # 处理cv2窗口事件
            cv2.waitKey(1)
            
            # 短暂休眠
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("用户中断")
    finally:
        pipeline.client.moveByVelocityAsync(0, 0, 0, 0.1)


if __name__ == '__main__':
    main()
