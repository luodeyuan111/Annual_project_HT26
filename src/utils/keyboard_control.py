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

def keyboard_control_process(command_queue, logger, should_stop):
    """
    键盘控制进程
    
    功能：
    - 持续监听键盘输入
    - 通过Queue发送按键命令给主进程
    - 支持多键同时按下
    
    Args:
        command_queue: 进程间通信队列
        logger: 日志记录器
        should_stop: 共享停止标志
    """
    try:
        logger.info("=" * 60)
        logger.info("键盘控制进程启动")
        logger.info(f"启动时间: {datetime.now()}")
        logger.info("开始监听键盘输入...")
        logger.info("=" * 60)
        
        # 键盘命令映射
        key_commands = {
            ord('w'): {'action': 'move', 'direction': '前进', 'dx': 10, 'dy': 0, 'dz': 0},
            ord('s'): {'action': 'move', 'direction': '后退', 'dx': -10, 'dy': 0, 'dz': 0},
            ord('a'): {'action': 'move', 'direction': '左移', 'dx': 0, 'dy': -10, 'dz': 0},
            ord('d'): {'action': 'move', 'direction': '右移', 'dx': 0, 'dy': 10, 'dz': 0},
            ord(','): {'action': 'move', 'direction': '上升', 'dx': 0, 'dy': 0, 'dz': -10},
            ord('.'): {'action': 'move', 'direction': '下降', 'dx': 0, 'dy': 0, 'dz': 10},
            ord('e'): {'action': 'process', 'direction': '处理两帧'},
            ord('v'): {'action': 'visualize', 'direction': '切换可视化'},
            ord('q'): {'action': 'quit', 'direction': '退出'},
        }
        
        # 等待队列初始化完成
        time.sleep(0.1)
        
        # 主循环
        while not should_stop.value:
            try:
                # 检查是否有按键（立即返回，不阻塞）
                if msvcrt.kbhit():
                    # 读取所有可用的按键
                    keys = []
                    while msvcrt.kbhit():
                        try:
                            key = ord(msvcrt.getch())
                            keys.append(key)
                        except:
                            break
                    
                    # 处理所有按键
                    for key in keys:
                        if key in key_commands:
                            cmd = key_commands[key]
                            action = cmd['action']
                            
                            if action == 'move':
                                logger.info(f"按键检测: {cmd['direction']}")
                                logger.debug(f"移动指令: dx={cmd['dx']}, dy={cmd['dy']}, dz={cmd['dz']}")
                                command_queue.put(('move', cmd))
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
                        
                        # 调试信息（只记录一次）
                        if key not in key_commands and not hasattr(keyboard_control_process, '_debug_printed'):
                            logger.debug(f"未定义按键: ASCII={key}, 字符={chr(key)}")
                            keyboard_control_process._debug_printed = True
                
                # 短暂休眠，降低CPU占用
                time.sleep(0.001)  # 1ms检查一次
                
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