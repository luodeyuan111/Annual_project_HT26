"""有关无人机测试的主程序，用于捕获视频流并响应键盘输入以控制无人机移动和保存图像帧。"""
# main.py
import cv2
import time
import sys
import airsim
import numpy as np  # 导入 numpy
import os
import msvcrt
from datetime import datetime
from logging_utils import get_logger

# 添加必要的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

sys.path.insert(0, project_root)  # 添加项目根目录
sys.path.insert(0, current_dir)  # 添加 src 目录

from Drone_Interface.rgb_data_extractor import RGBDataExtractor
from Drone_Interface.rgb_data_extractor import FrameBuffer

# 导入神经网络感知模块
from neural_processing.neural_perception import NeuralPerception
print("✓ 使用真实的 NeuralPerception (torch 可用)")

from Visual_process.visual_center import VisualPerception

def main():
    # 初始化日志系统
    logger = get_logger(log_dir='logs', log_file='drone_vision.log', log_level='INFO', log_to_file=True)
    logger.info("=" * 60)
    logger.info("无人机视觉感知系统启动")
    logger.info(f"启动时间: {datetime.now()}")
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"Python版本: {sys.version.split()[0]}")
    
    # 初始化
    logger.info("开始初始化...")
    extractor = RGBDataExtractor(drone_name="Drone1", save_images=False)
    camera_name = "front_camera"  # 指定摄像头名称
    display_frame = None  # 用于存储要显示的帧
    client = extractor.client  # 获取 airsim client
    
    # 检查AirSim连接
    try:
        connection_state = client.simGetConnectionState()
        logger.info(f"AirSim连接状态: {connection_state}")
        if not connection_state:
            logger.error("无法连接到AirSim，请确保AirSim已启动")
            return
    except Exception as e:
        logger.error(f"检查AirSim连接失败: {e}")
        return
    
    # 初始化帧缓冲区
    frame_buffer = FrameBuffer()
    logger.info("帧缓冲区初始化完成")
    
    # 初始化神经网络和视觉处理枢纽
    logger.info("初始化神经网络和视觉处理模块...")
    neural_perception = NeuralPerception()
    visual_perception = VisualPerception()
    logger.info("神经网络和视觉处理模块初始化完成")

    # 创建显示窗口（非阻塞），避免 GUI 无响应
    cv2.namedWindow("Drone View", cv2.WINDOW_NORMAL)
    logger.info("显示窗口已创建")

    # 移动速度
    move_speed = 10  # 单位：米/秒
    logger.info(f"移动速度: {move_speed} m/s")

    # 帧计数器
    frame_counter = 0
    logger.info("=" * 60)
    logger.info("系统初始化完成，等待键盘输入...")
    logger.info("可用按键:")
    logger.info("  w - 前进")
    logger.info("  s - 后退")
    logger.info("  a - 左移")
    logger.info("  d - 右移")
    logger.info("  , - 上升")
    logger.info("  . - 下降")
    logger.info("  e - 捕获并处理两帧（间隔100ms）")
    logger.info("  q - 退出")
    logger.info("=" * 60)

    try:
        print("开始视频流捕获...\n键盘输入决定行为 :\n'e'键捕获并处理两帧（间隔100ms）\n'q'键退出\n'w'前进\n's'后退\n'a'左移\n'd'右移\n','上升\n'.'下降\n")
        while True:
            if msvcrt.kbhit():  # 检查是否有按键被按下
                key = ord(msvcrt.getch())  # 读取按键的 ASCII 码
                logger.info(f"按键检测: ASCII={key}, 字符={chr(key)}")
                
                # 控制无人机移动 - 使用同步方法避免时序问题
                if key == ord('w'):
                    logger.info("移动指令: 前进")
                    client.moveByVelocityBodyFrameAsync(move_speed, 0, 0, 0.1).join()  # 前进
                elif key == ord('s'):
                    logger.info("移动指令: 后退")
                    client.moveByVelocityBodyFrameAsync(-move_speed, 0, 0, 0.1).join()  # 后退
                elif key == ord('a'):
                    logger.info("移动指令: 左移")
                    client.moveByVelocityBodyFrameAsync(0, -move_speed, 0, 0.1).join()  # 左
                elif key == ord('d'):
                    logger.info("移动指令: 右移")
                    client.moveByVelocityBodyFrameAsync(0, move_speed, 0, 0.1).join()  # 右
                elif key == ord(','):
                    logger.info("移动指令: 上升")
                    client.moveByVelocityBodyFrameAsync(0, 0, -move_speed, 0.1).join()  # 上
                elif key == ord('.'):
                    logger.info("移动指令: 下降")
                    client.moveByVelocityBodyFrameAsync(0, 0, move_speed, 0.1).join()  # 下
                elif key == ord('e'):
                    # 捕获并处理两帧（间隔100ms）- 一次按键完成所有操作
                    logger.info("开始捕获帧对...")
                    start_time = time.time()
                    
                    print("[开始] 捕获第1帧...")
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
                                
                                print("[继续] 捕获第2帧...")
                                timestamp2 = int(time.time() * 1000)
                                rgb_data2 = extractor.capture_rgb_images(timestamp2)
                                
                                if camera_name in rgb_data2:
                                    frame2 = rgb_data2[camera_name]
                                    if frame2 is not None and isinstance(frame2, np.ndarray) and frame2.size > 0:
                                        frame_buffer.update(frame2)
                                        frame_t, frame_t_plus_1 = frame_buffer.get_frames()
                                        
                                        # 处理两帧
                                        if frame_t is not None and frame_t_plus_1 is not None:
                                            display_frame = frame2
                                            logger.info(f"[第2帧] 捕获成功 - 分辨率: {frame2.shape}, 时间戳: {timestamp2}")
                                            
                                            logger.info(f"开始处理帧对 {frame_counter} (NeuralPerception + VisualPerception pipeline)...")
                                            
                                            try:
                                                # 1. 神经网络处理
                                                process_start = time.time()
                                                neural_output = neural_perception.process_frame_pair(frame_t, frame_t_plus_1)
                                                process_time = time.time() - process_start
                                                
                                                # 2. 视觉处理枢纽
                                                visual_state = visual_perception.process(neural_output)
                                                
                                                # 3. 输出关键信息（同时记录到日志）
                                                quality = neural_output.quality_metrics.get('overall_confidence', 0)
                                                features = len(neural_output.feature_points.get('points_t', []))
                                                segments = len(neural_output.segmentation.get('labels', []))
                                                
                                                logger.info(f"✓ 神经网络处理完成 - 质量: {quality:.3f}, 特征点: {features}, 分割区域: {segments}, 时间: {process_time:.3f}s")
                                                
                                                # 4. 位姿信息
                                                if visual_state.ego_motion:
                                                    motion = visual_state.ego_motion
                                                    t = motion.translation
                                                    logger.info(f"✓ 机器人运动 - 位移: dx={t[0]:.2f}, dy={t[1]:.2f}, dz={t[2]:.2f}, 置信度: {motion.confidence:.3f}")
                                                
                                                # 5. 障碍物信息
                                                if visual_state.obstacle_frame is not None:
                                                    logger.info(f"✓ 障碍物检测 - 距离: {visual_state.obstacle_frame.closest_depth:.1f}m, 角度: {visual_state.obstacle_frame.closest_angle:.1f}°")
                                                
                                                # 6. 质量指标
                                                logger.info(f"✓ 质量指标 - 评分: {visual_state.quality}, 警告: {visual_state.warnings}")
                                                
                                                # 7. 序列化输出
                                                payload = visual_state.to_payload()
                                                logger.info(f"✓ 处理完成 - 输出已序列化")
                                                
                                                frame_counter += 1
                                            except Exception as e:
                                                logger.error(f"⚠ 视觉处理失败 - 错误: {e}", exc_info=True)
                                                frame_counter += 1
                                        else:
                                            logger.warning("⚠ 帧数据不足，无法处理")
                                    else:
                                        logger.error(f"警告：第2帧从相机 {camera_name} 获取的帧为空或无效")
                                else:
                                    logger.error(f"警告：无法获取相机 {camera_name} 的第2帧")
                            else:
                                logger.warning("⚠ 第1帧捕获成功，但帧缓冲区尚未准备好")
                        else:
                            logger.error(f"警告：第1帧从相机 {camera_name} 获取的帧为空或无效")
                    else:
                        logger.error(f"警告：无法获取相机 {camera_name} 的第1帧")
                elif key == ord('q'):
                    logger.info("退出程序")
                    break
                else:
                    logger.debug(f"无效键: ASCII={key}, 字符={chr(key)}")
            else:
                time.sleep(0.01)  # 暂停 10 毫秒

            # 显示图像（如果已捕获）
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

            # 处理 GUI 事件，防止窗口变为未响应
            cv2.waitKey(1)

            # 不再在主循环中自动处理，只在按 'e' 键时处理
            
    except Exception as e:
        print(f"发生错误: {e}")

    finally:
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

if __name__ == "__main__":
    main()
