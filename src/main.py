"""有关无人机测试的主程序，用于捕获视频流并响应键盘输入以控制无人机移动和保存图像帧。

多进程架构版本：
- 主进程：处理视觉和显示
- 键盘控制进程：独立监听键盘，不阻塞主进程
- 通过Queue进行进程间通信
"""

# main.py (多进程版本)
import cv2
import time
import sys
import airsim
import numpy as np  # 导入 numpy
import os
from datetime import datetime
from multiprocessing import Process, Queue, Value, Array

# 添加必要的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

sys.path.insert(0, project_root)  # 添加项目根目录
sys.path.insert(0, current_dir)  # 添加 src 目录

from Drone_Interface.rgb_data_extractor import RGBDataExtractor
from Drone_Interface.rgb_data_extractor import FrameBuffer

# 导入神经网络感知模块
from neural_processing.neural_perception import NeuralPerception
from Visual_process.visual_center import VisualPerception

# 导入工具模块（使用绝对导入避免与models/monodepth2/utils.py冲突）
from src.utils.logging_utils import get_logger
from src.utils.keyboard_control import keyboard_control_process
from src.utils.visualization import Visualizer

def visual_processing_process(command_queue, should_stop, velocity_state):
    """
    视觉处理进程（主进程）

    功能：
    - 持续捕获和处理帧
    - 通过command_queue接收键盘命令
    - 显示图像和结果

    Args:
        command_queue: 进程间通信队列
        should_stop: 共享停止标志
        velocity_state: 共享速度状态 [vx, vy, vz, active]
    """
    try:
        logger = get_logger(log_dir='logs', log_file='visual.log', log_level='INFO', log_to_file=True)
        logger.info("=" * 60)
        logger.info("视觉处理进程启动")
        logger.info(f"启动时间: {datetime.now()}")
        logger.info("=" * 60)
        
        # 初始化
        logger.info("开始初始化...")
        logger.info("初始化RGBDataExtractor...")
        extractor = RGBDataExtractor(drone_name="Drone1", save_images=False)
        camera_name = "front_camera"  # 指定摄像头名称
        display_frame = None  # 用于存储要显示的帧
        client = extractor.client  # 获取 airsim client
        logger.info("RGBDataExtractor初始化完成")
        
        # 初始化帧缓冲区
        frame_buffer = FrameBuffer()
        logger.info("帧缓冲区初始化完成")
        
        # 初始化神经网络和视觉处理枢纽
        logger.info("初始化神经网络和视觉处理模块...")
        neural_perception = NeuralPerception()
        visual_perception = VisualPerception()
        logger.info("神经网络和视觉处理模块初始化完成")
        
        # 初始化可视化器
        visualizer = Visualizer(enable_depth=True, enable_obstacle=True, enable_flow=False)
        show_visualization = False  # 可视化开关状态
        logger.info("可视化器初始化完成 (按'v'切换显示)")

        # 创建显示窗口（非阻塞），避免 GUI 无响应
        cv2.namedWindow("Drone View", cv2.WINDOW_NORMAL)
        logger.info("显示窗口已创建")

        # 移动速度
        move_speed = 10  # 单位：米/秒
        logger.info(f"移动速度: {move_speed} m/s")

        # 帧计数器
        frame_counter = 0
        need_process = False  # 是否需要处理两帧

        logger.info("=" * 60)
        logger.info("系统初始化完成，等待键盘输入...")
        logger.info("可用按键:")
        logger.info("  w - 前进")
        logger.info("  s - 后退")
        logger.info("  a - 左移")
        logger.info("  d - 右移")
        logger.info("  , - 上升")
        logger.info("  . - 下降")
        logger.info("  space - 停止所有运动")
        logger.info("  e - 捕获并处理两帧（间隔100ms）")
        logger.info("  v - 切换可视化显示（深度图+障碍物雷达）")
        logger.info("  q - 退出")
        logger.info("=" * 60)

        print("开始视频流捕获...\n多进程架构启动！")
        print("键盘输入决定行为 :\n'w'前进\n's'后退\n'a'左移\n'd'右移\n','上升\n'.'下降\n'空格'停止所有\n'e'捕获并处理两帧\n'v'可视化开关\n'q'退出\n")
        
        # 记录上一帧的速度状态，用于检测变化
        last_velocity = [0.0, 0.0, 0.0]
        
        # 主循环
        while not should_stop.value:
            try:
                # 1. 持续速度控制：检查速度状态变化并发送指令
                current_velocity = [velocity_state[0], velocity_state[1], velocity_state[2]]
                is_active = velocity_state[3] > 0
                
                # 如果速度发生变化，或者需要保持运动，则发送速度指令
                if current_velocity != last_velocity or is_active:
                    try:
                        # 持续发送速度指令（duration很小，实现实时控制）
                        client.moveByVelocityBodyFrameAsync(
                            float(current_velocity[0]),
                            float(current_velocity[1]),
                            float(current_velocity[2]),
                            0.05  # 50ms周期
                        )
                        last_velocity = current_velocity.copy()
                    except Exception as e:
                        logger.error(f"运动控制失败: {e}")
                
                # 2. 检查键盘命令（从队列读取）
                while not command_queue.empty():
                    cmd_type, data = command_queue.get()
                    
                    if cmd_type == 'process':
                        need_process = True
                        logger.info("收到处理指令: 捕获并处理两帧")
                    
                    elif cmd_type == 'visualize':
                        show_visualization = not show_visualization
                        status = "开启" if show_visualization else "关闭"
                        logger.info(f"可视化已{status}")
                    
                    elif cmd_type == 'quit':
                        logger.info("收到退出指令")
                        should_stop.value = True
                        break
                
                # 3. 如果需要处理两帧
                if need_process:
                    need_process = False
                    display_frame = process_two_frames(
                        extractor, 
                        frame_buffer, 
                        camera_name, 
                        neural_perception, 
                        visual_perception, 
                        frame_counter,
                        logger,
                        client,
                        visualizer,
                        show_visualization
                    )
                    frame_counter += 1
                
                # 4. 显示图像（如果已捕获）
                if display_frame is not None:
                    try:
                        # AirSim 返回的是 RGB，OpenCV 使用 BGR，用于正确显示颜色
                        if isinstance(display_frame, np.ndarray) and display_frame.ndim == 3 and display_frame.shape[2] == 3:
                            show_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                        else:
                            show_frame = display_frame
                        cv2.imshow("Drone View", show_frame)
                    except Exception as e:
                        print(f"显示图像时出错: {e}")
                
                # 5. 处理 GUI 事件，防止窗口变为未响应
                cv2.waitKey(1)
                
            except KeyboardInterrupt:
                logger.info("视觉处理进程收到中断信号")
                should_stop.value = True
                break
            except Exception as e:
                logger.error(f"视觉处理进程错误: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("正在清理资源...")
        try:
            extractor.disconnect()
            logger.info("AirSim连接已断开")
        except Exception as e:
            logger.error(f"断开AirSim连接失败: {e}")
        
        try:
            cv2.destroyAllWindows()
            logger.info("显示窗口已关闭")
        except Exception as e:
            logger.error(f"关闭显示窗口失败: {e}")
        
        logger.info("=" * 60)
        logger.info(f"程序正常退出，共处理帧数: {frame_counter}")
        logger.info("=" * 60)

    except Exception as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)


def process_two_frames(extractor, frame_buffer, camera_name, neural_perception, visual_perception, 
                       frame_counter, logger, client, visualizer=None, show_visualization=False):
    """
    处理两帧的函数（从主流程中提取）

    Args:
        extractor: RGB数据提取器
        frame_buffer: 帧缓冲区
        camera_name: 摄像头名称
        neural_perception: 神经网络感知模块
        visual_perception: 视觉处理枢纽
        frame_counter: 帧计数器
        logger: 日志记录器
        client: AirSim客户端
        visualizer: 可视化器（可选）
        show_visualization: 是否显示可视化
    """
    logger.info("开始捕获帧对...")
    start_time = time.time()
    
    print(f"[{frame_counter}] 开始捕获帧对...")
    
    # 捕获第1帧
    timestamp1 = int(time.time() * 1000)
    logger.debug(f"开始捕获第1帧，时间戳: {timestamp1}")
    
    rgb_data1 = extractor.capture_rgb_images(timestamp1)
    
    if camera_name in rgb_data1:
        frame1 = rgb_data1[camera_name]
        if frame1 is not None and isinstance(frame1, np.ndarray) and frame1.size > 0:
            frame_buffer.update(frame1)
            frame_t, frame_t_plus_1 = frame_buffer.get_frames()
            
            if frame_t is not None:
                logger.info(f"[第1帧] 捕获成功 - 分辨率: {frame1.shape}, 时间戳: {timestamp1}")
                
                # 等待100ms后捕获第2帧
                logger.debug("等待100ms后捕获第2帧...")
                time.sleep(0.1)
                
                # 捕获第2帧
                timestamp2 = int(time.time() * 1000)
                rgb_data2 = extractor.capture_rgb_images(timestamp2)
                
                if camera_name in rgb_data2:
                    frame2 = rgb_data2[camera_name]
                    if frame2 is not None and isinstance(frame2, np.ndarray) and frame2.size > 0:
                        frame_buffer.update(frame2)
                        frame_t, frame_t_plus_1 = frame_buffer.get_frames()
                        
                        # 处理两帧
                        if frame_t is not None and frame_t_plus_1 is not None:
                            logger.info(f"[第2帧] 捕获成功 - 分辨率: {frame2.shape}, 时间戳: {timestamp2}")
                            
                            logger.info(f"开始处理帧对 {frame_counter} (NeuralPerception + VisualPerception pipeline)...")
                            
                            try:
                                # 1. 神经网络处理
                                process_start = time.time()
                                neural_output = neural_perception.process_frame_pair(frame_t, frame_t_plus_1)
                                process_time = time.time() - process_start
                                
                                # 2. 视觉处理枢纽
                                visual_state = visual_perception.process(neural_output)
                                
                                # 3. 输出关键信息（分类清晰展示）
                                quality = neural_output.quality_metrics.get('overall_confidence', 0)
                                features = len(neural_output.feature_points.get('points_t', []))
                                segments = len(neural_output.segmentation.get('labels', []))
                                
                                # 分隔线
                                logger.info("─" * 40)
                                logger.info(f"【帧 {frame_counter}】处理完成")
                                logger.info(f"  神经网络: 质量={quality:.3f}, 特征点={features}, 分割={segments}, 耗时={process_time:.2f}s")
                                
                                # 4. 位姿信息（添加速度）
                                if visual_state.ego_motion:
                                    motion = visual_state.ego_motion
                                    t = motion.translation
                                    v = motion.velocity
                                    logger.info(f"  机器人运动: Δx={t[0]:.3f}m, Δy={t[1]:.3f}m, Δz={t[2]:.3f}m | 速度=({v[0]:.2f},{v[1]:.2f},{v[2]:.2f})m/s | 置信度={motion.confidence:.2f}")
                                
                                # 5. 障碍物信息
                                if visual_state.obstacle_frame is not None:
                                    obst = visual_state.obstacle_frame
                                    # 计算覆盖率
                                    valid_ratio = obst.coverage_ratio
                                    logger.info(f"  障碍物检测: 最近={obst.closest_depth:.2f}m @ {obst.closest_angle:.1f}° | 覆盖率={valid_ratio*100:.1f}%")
                                
                                # 6. 警告信息
                                warnings = visual_state.warnings
                                if warnings:
                                    logger.warning(f"  警告: {', '.join(warnings)}")
                                
                                logger.info("─" * 40)
                                
                                # 返回数据用于显示
                                if show_visualization and visualizer is not None:
                                    # 生成可视化图像
                                    vis_frame = visualizer.visualize(
                                        rgb_image=frame2,
                                        depth_map=neural_output.depth_maps.get('depth_t'),
                                        obstacle_frame=visual_state.obstacle_frame,
                                        optical_flow=neural_output.optical_flow.get('flow_field'),
                                        quality_metrics=neural_output.quality_metrics
                                    )
                                    return vis_frame
                                else:
                                    return frame2
                                
                            except Exception as e:
                                logger.error(f"⚠ 视觉处理失败 - 错误: {e}", exc_info=True)
                                return None
                        else:
                            logger.warning("⚠ 帧数据不足，无法处理")
                            return None
                    else:
                        logger.error(f"警告：第2帧从相机 {camera_name} 获取的帧为空或无效")
                        return None
                else:
                    logger.error(f"警告：无法获取相机 {camera_name} 的第2帧")
                    return None
            else:
                logger.warning("⚠ 第1帧捕获成功，但帧缓冲区尚未准备好")
                return None
        else:
            logger.error(f"警告：第1帧从相机 {camera_name} 获取的帧为空或无效")
            return None
    else:
        logger.error(f"警告：无法获取相机 {camera_name} 的第1帧")
        return None


def main():
    """主函数 - 启动两个进程"""
    pass


if __name__ == "__main__":
    # 初始化日志系统
    main_logger = get_logger(log_dir='logs', log_file='main.log', log_level='INFO', log_to_file=True)
    main_logger.info("=" * 60)
    main_logger.info("无人机视觉感知系统启动 (多进程架构)")
    main_logger.info(f"启动时间: {datetime.now()}")
    main_logger.info(f"项目根目录: {project_root}")
    main_logger.info(f"Python版本: {sys.version.split()[0]}")
    main_logger.info("=" * 60)
    
    # 创建共享变量和队列
    should_stop = Value('b', False)
    command_queue = Queue()
    
    # 创建速度状态共享变量 [vx, vy, vz, active]
    # 用于持续速度控制模式
    velocity_state = Array('d', [0.0, 0.0, 0.0, 0.0])
    
    # 启动键盘控制进程
    main_logger.info("启动键盘控制进程...")
    keyboard_proc = Process(
        target=keyboard_control_process,
        args=(command_queue, main_logger, should_stop, velocity_state)
    )
    keyboard_proc.daemon = True  # 设置为守护进程，主进程退出时自动退出
    keyboard_proc.start()
    main_logger.info(f"键盘控制进程PID: {keyboard_proc.pid}")
    
    # 启动视觉处理进程
    main_logger.info("启动视觉处理进程...")
    visual_proc = Process(
        target=visual_processing_process,
        args=(command_queue, should_stop, velocity_state)
    )
    visual_proc.start()
    main_logger.info(f"视觉处理进程PID: {visual_proc.pid}")
    
    # 等待视觉处理进程结束
    visual_proc.join()
    
    # 等待键盘控制进程结束
    keyboard_proc.join()
    
    main_logger.info("所有进程已退出")
    main_logger.info("=" * 60)