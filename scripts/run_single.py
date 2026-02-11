#!/usr/bin/env python3
"""
单次处理脚本 - 修正路径版本
"""

import sys
import os
import traceback

# 获取项目根目录的正确路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # D:\project_new\scripts
project_root = os.path.dirname(current_dir)  # D:\project_new

print("=" * 60)
print("无人机视觉处理系统 - 单次处理模式")
print(f"脚本目录: {current_dir}")
print(f"项目根目录: {project_root}")
print("=" * 60)

# 将项目根目录添加到系统路径
sys.path.insert(0, project_root)

# 检查关键目录是否存在
print("检查项目结构...")
required_dirs = ['src', 'Drone_Interface', 'integration_system']
for dir_name in required_dirs:
    dir_path = os.path.join(project_root, dir_name)
    if os.path.exists(dir_path):
        print(f"✓ 找到: {dir_name}")
    else:
        print(f"✗ 缺失: {dir_name}")

print("=" * 60)

def main():
    """主函数"""
    
    try:
        # 尝试导入整合系统
        print("导入整合系统模块...")
        try:
            from integration_system.drone_vision_integrator import DroneVisionIntegrator
            print("✓ 导入成功")
        except ImportError as e:
            print(f"导入错误: {e}")
            print("\n尝试从不同路径导入...")
            
            # 尝试直接导入
            integrator_path = os.path.join(project_root, "integration_system", "drone_vision_integrator.py")
            if os.path.exists(integrator_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("drone_vision_integrator", integrator_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                DroneVisionIntegrator = module.DroneVisionIntegrator
                print("✓ 直接导入成功")
            else:
                print(f"✗ 找不到整合器文件: {integrator_path}")
                return 1
        
        # 创建整合器
        print("创建整合器实例...")
        try:
            integrator = DroneVisionIntegrator()
        except Exception as e:
            print(f"✗ 创建整合器失败: {e}")
            traceback.print_exc()
            return 1
        
        # 执行单次处理
        print("执行单次处理周期...")
        result = integrator.single_processing_cycle()
        
        if result:
            print("\n" + "=" * 60)
            print("✓ 处理成功！")
            print(f"结果保存到: {integrator.output_dir}")
            print("=" * 60)
        else:
            print("\n✗ 处理失败")
        
        # 清理资源
        integrator.cleanup()
        return 0
        
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
        return 0
    except Exception as e:
        print(f"\n✗ 运行时错误: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())