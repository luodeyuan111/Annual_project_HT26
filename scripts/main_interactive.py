#!/usr/bin/env python3
"""
交互式主程序 - 终端指令触发无人机视觉处理
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'integration_system'))

def main():
    """主函数"""
    print("无人机视觉处理系统 - 交互模式")
    print("=" * 60)
    
    try:
        # 导入整合系统
        from integration_system.drone_vision_integrator import DroneVisionIntegrator
        from integration_system.command_interface import DroneVisionCommandInterface
        
        # 创建整合器
        config_dir = project_root / 'integration_system/configs'
        integrator = DroneVisionIntegrator(
            drone_config_path=str(config_dir / 'drone_config.yaml'),
            vision_config_path=str(config_dir / 'vision_config.yaml')
        )
        
        # 创建命令行接口
        cli = DroneVisionCommandInterface(integrator)
        
        # 运行交互式命令行
        cli.cmdloop()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n系统运行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())