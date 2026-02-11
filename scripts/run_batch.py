#!/usr/bin/env python3
"""
批量处理脚本 - 执行多次处理周期
"""

import sys
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'integration_system'))

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量处理无人机视觉数据')
    parser.add_argument('--cycles', type=int, default=5, help='处理周期数')
    parser.add_argument('--interval', type=float, default=1.0, help='周期间隔（秒）')
    
    args = parser.parse_args()
    
    print(f"无人机视觉处理系统 - 批量处理模式（{args.cycles}个周期）")
    print("=" * 60)
    
    try:
        # 导入整合系统
        from integration_system.drone_vision_integrator import DroneVisionIntegrator
        
        # 创建整合器
        config_dir = project_root / 'integration_system/configs'
        integrator = DroneVisionIntegrator(
            drone_config_path=str(config_dir / 'drone_config.yaml'),
            vision_config_path=str(config_dir / 'vision_config.yaml')
        )
        
        # 批量处理
        results = []
        for i in range(args.cycles):
            print(f"\n>>> 处理周期 {i+1}/{args.cycles}")
            result = integrator.single_processing_cycle()
            if result:
                results.append(result)
            
            # 周期间隔（最后一个周期不等待）
            if i < args.cycles - 1:
                print(f"等待 {args.interval} 秒...")
                time.sleep(args.interval)
        
        # 显示汇总信息
        print(f"\n批量处理完成！")
        print(f"成功处理: {len(results)}/{args.cycles} 个周期")
        
        # 清理资源
        integrator.cleanup()
        
    except KeyboardInterrupt:
        print("\n批量处理被中断")
    except Exception as e:
        print(f"\n错误: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())