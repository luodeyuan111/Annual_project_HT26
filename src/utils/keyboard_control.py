"""
键盘控制进程模块
独立的键盘监听进程，不阻塞主进程
"""

import msvcrt
import time
import sys
import os
from multiprocessing import Process, Queue, Value
from datetime import datetime

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # utils/src/project_root

def keyboard_control_process(command_queue, logger, should_stop, velocity_state):
    """
    键盘控制进程
    
    功能：
    - 持续监听键盘输入
    - 通过Queue发送按键命令给主进程
    - 支持多键同时按下
    - 支持持续运动模式（按下移动，持续运动；松开停止）
    - 使用共享内存追踪按键状态
    
    Args:
        command_queue: 进程间通信队列
        logger: 日志记录器
        should_stop: 共享停止标志
        velocity_state: 共享速度状态 [vx, vy, vz, active]
    """
    try:
        logger.info("=" * 60)
        logger.info("键盘控制进程启动")
        logger.info(f"启动时间: {datetime.now()}")
        logger.info("开始监听键盘输入...")
        logger.info("=" * 60)
        
        # 速度设置
        MOVE_SPEED = 10  # 米/秒
        
        # 移动命令 - 按下启动，松开停止
        # 使用小写字母的ASCII码
        move_keys = {
            119: {'axis': 0, 'value': MOVE_SPEED, 'key': 'w'},   # w 前进
            115: {'axis': 0, 'value': -MOVE_SPEED, 'key': 's'},  # s 后退
            97:  {'axis': 1, 'value': -MOVE_SPEED, 'key': 'a'},  # a 左移
            100: {'axis': 1, 'value': MOVE_SPEED, 'key': 'd'},   # d 右移
            44:  {'axis': 2, 'value': -MOVE_SPEED, 'key': ','},  # , 上升
            46:  {'axis': 2, 'value': MOVE_SPEED, 'key': '.'},   # . 下降
        }
        
        # 特殊命令
        special_commands = {
            101: {'action': 'process', 'key': 'e'},   # e
            118: {'action': 'visualize', 'key': 'v'}, # v
            113: {'action': 'quit', 'key': 'q'},      # q
            32:  {'action': 'stop_all', 'key': 'space'},  # 空格
        }
        
        # 按键状态追踪
        key_active = {k: False for k in move_keys.keys()}
        
        # 等待队列初始化完成
        time.sleep(0.1)
        
        # 主循环
        while not should_stop.value:
            try:
                # 检查是否有按键（立即返回，不阻塞）
                if msvcrt.kbhit():
                    # 读取按键
                    try:
                        ch = msvcrt.getch()
                    except:
                        continue
                    
                    # 处理扩展键（方向键等）
                    if ch in (b'\xe0', b'\x00'):
                        try:
                            _ = msvcrt.getch()
                        except:
                            pass
                        continue
                    
                    # 获取键值
                    key = ord(ch)
                    
                    # 处理移动按键
                    if key in move_keys:
                        cmd = move_keys[key]
                        axis = cmd['axis']
                        value = cmd['value']
                        
                        # 切换该轴的速度方向
                        current_axis_value = 0
                        for k, v in move_keys.items():
                            if v['axis'] == axis and key_active.get(k, False):
                                current_axis_value = v['value']
                        
                        if current_axis_value == value:
                            key_active[key] = False
                            logger.info(f"停止运动: axis={axis}")
                        else:
                            for k, v in move_keys.items():
                                if v['axis'] == axis:
                                    key_active[k] = (k == key)
                            logger.info(f"开始运动: axis={axis}, value={value}")
                        
                        velocity_state[3] = 1 if any(key_active.values()) else 0
                        
                    # 处理特殊命令
                    elif key in special_commands:
                        cmd = special_commands[key]
                        action = cmd['action']
                        
                        if action == 'stop_all':
                            key_active = {k: False for k in move_keys.keys()}
                            velocity_state[0] = 0
                            velocity_state[1] = 0
                            velocity_state[2] = 0
                            velocity_state[3] = 0
                            logger.info("停止所有运动 (空格)")
                            
                        elif action == 'process':
                            logger.info("按键检测: 处理两帧 (e)")
                            command_queue.put(('process', None))
                        elif action == 'visualize':
                            logger.info("按键检测: 切换可视化 (v)")
                            command_queue.put(('visualize', None))
                        elif action == 'quit':
                            logger.info("按键检测: 退出 (q)")
                            command_queue.put(('quit', None))
                            should_stop.value = True
                            break
                        
                        # 调试信息
                        if key not in move_keys and key not in special_commands:
                            logger.debug(f"未定义按键: ASCII={key}")
                
                # 更新速度状态（基于按键状态）
                # axis 0: x (前进/后退), axis 1: y (左/右), axis 2: z (上升/下降)
                vx, vy, vz = 0.0, 0.0, 0.0
                
                for key, cmd in move_keys.items():
                    if key_active.get(key, False):
                        axis = cmd['axis']
                        value = cmd['value']
                        if axis == 0:
                            vx = value
                        elif axis == 1:
                            vy = value
                        elif axis == 2:
                            vz = value
                
                # 更新共享内存中的速度
                velocity_state[0] = vx
                velocity_state[1] = vy
                velocity_state[2] = vz
                
                # 短暂休眠，降低CPU占用
                time.sleep(0.01)  # 10ms更新一次速度
                
            except KeyboardInterrupt:
                logger.info("键盘控制进程收到中断信号")
                should_stop.value = True
                break
            except Exception as e:
                logger.error(f"键盘控制进程错误: {e}", exc_info=True)
                time.sleep(0.1)
        
        logger.info("键盘控制进程退出")
        logger.info("=" * 60)
        
    except Exception as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    """
    测试键盘控制进程
    """
    import logging
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger('KeyboardControl')
    
    print("测试键盘控制进程...")
    print("可用按键：")
    print("  w - 前进")
    print("  s - 后退")
    print("  a - 左移")
    print("  d - 右移")
    print("  , - 上升")
    print("  . - 下降")
    print("  e - 处理两帧")
    print("  q - 退出")
    print("按任意键开始测试（q键退出）")
    print("=" * 60)
    
    # 创建队列
    command_queue = Queue()
    should_stop = Value('b', False)  # 使用Value而不是简单的bool
    
    # 启动键盘控制进程
    keyboard_proc = Process(
        target=keyboard_control_process,
        args=(command_queue, logger, should_stop)
    )
    keyboard_proc.daemon = True
    keyboard_proc.start()
    
    # 等待进程结束
    try:
        keyboard_proc.join()
    except KeyboardInterrupt:
        print("测试中断")
        should_stop.value = True
        keyboard_proc.join()
    
    print("测试结束")