"""交互式终端控制脚本

使用说明:
- 在 PowerShell 中运行（激活 venv）：
  .\venv_raft\Scripts\Activate.ps1
  python scripts\interactive_control.py

按键说明:
- t: 起飞
- l: 降落
- w/a/s/d: 前/左/后/右（短距离，速度模式）
- c: 捕获并处理一次帧对
- n: 连续捕获并处理 N 次（随后输入数字 N）
- q: 退出

每次按键触发移动后，脚本会调用整合器的处理函数进行帧对处理并保存结果。
"""
import time
import sys
import os

# Windows 下读取按键
try:
    import msvcrt
except Exception:
    msvcrt = None

# 确保以脚本方式直接运行时能找到项目根目录下的顶级包
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Drone_Interface.Drone_Interface_AirSim import DroneController
from integration_system.drone_vision_integrator import DroneVisionIntegrator


def read_key():
    if msvcrt:
        ch = msvcrt.getch()
        try:
            return ch.decode('utf-8')
        except Exception:
            return ''
    else:
        # fallback
        return sys.stdin.read(1)


def main():
    print('启动交互式控制（AirSim）')
    integrator = DroneVisionIntegrator()

    # 覆盖为 AirSim 控制器（直接实例化，确保使用 AirSim）
    try:
        airsim_ctrl = DroneController()
    except Exception as e:
        print('无法连接 AirSim:', e)
        return

    integrator.drone_controller = airsim_ctrl

    print('\n按键控制: t=takeoff, l=land, w/a/s/d=move, c=capture once, n=连续N次, q=quit')

    running = True
    while running:
        print('\n等待按键...')
        key = read_key()
        print(f'按键: {key}')

        if key == 't':
            print('起飞')
            airsim_ctrl.takeoff()
        elif key == 'l':
            print('降落')
            airsim_ctrl.land()
        elif key in ('w', 'a', 's', 'd'):
            # 简单速度控制：前为负y（AirSim NED坐标系注意），这里按常见习惯设定
            duration = 0.6
            speed = 1.0
            vx = 0.0
            vy = 0.0
            vz = 0.0
            if key == 'w':
                vy = speed * -1.0
            elif key == 's':
                vy = speed * 1.0
            elif key == 'a':
                vx = speed * -1.0
            elif key == 'd':
                vx = speed * 1.0

            print(f'移动速度 vx={vx}, vy={vy}, duration={duration}')
            try:
                airsim_ctrl.move_by_velocity(vx=vx, vy=vy, vz=0.0, duration=duration)
            except Exception as e:
                print('移动失败:', e)

            # 移动后执行一次处理周期
            print('触发一次处理周期')
            integrator.single_processing_cycle()

        elif key == 'c':
            print('捕获并处理一次帧对')
            integrator.single_processing_cycle()

        elif key == 'n':
            print('输入要连续处理的次数 N，然后回车:')
            try:
                s = sys.stdin.readline().strip()
                N = int(s)
            except Exception:
                print('无效数字')
                continue
            for i in range(N):
                print(f'连续处理 {i+1}/{N} ...')
                integrator.single_processing_cycle()
                time.sleep(0.2)

        elif key == 'q':
            print('退出')
            running = False
        else:
            print('未识别按键')

    print('清理并断开连接')
    integrator.cleanup()


if __name__ == '__main__':
    main()
