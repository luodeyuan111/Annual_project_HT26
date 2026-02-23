#!/usr/bin/env python
"""
测试修复后的AirSim连接问题
"""
import sys
import os

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

from Drone_Interface.rgb_data_extractor import RGBDataExtractor
from neural_processing.neural_perception import NeuralPerception
from Visual_process.visual_center import VisualPerception

def test_imports():
    """测试所有导入是否正常"""
    print("=" * 60)
    print("测试1: 模块导入")
    print("=" * 60)
    
    try:
        print("✓ 导入 airsim 成功")
        import airsim
        print(f"  AirSim版本: {airsim.__version__}")
    except Exception as e:
        print(f"✗ 导入 airsim 失败: {e}")
        return False
    
    try:
        print("✓ 导入 RGBDataExtractor 成功")
    except Exception as e:
        print(f"✗ 导入 RGBDataExtractor 失败: {e}")
        return False
    
    try:
        print("✓ 导入 NeuralPerception 成功")
    except Exception as e:
        print(f"✗ 导入 NeuralPerception 失败: {e}")
        return False
    
    try:
        print("✓ 导入 VisualPerception 成功")
    except Exception as e:
        print(f"✗ 导入 VisualPerception 失败: {e}")
        return False
    
    print()
    return True

def test_rgb_extractor():
    """测试RGBDataExtractor初始化"""
    print("=" * 60)
    print("测试2: RGBDataExtractor初始化")
    print("=" * 60)
    
    try:
        print("正在初始化RGBDataExtractor...")
        extractor = RGBDataExtractor(drone_name="Drone1", save_images=False)
        print("✓ RGBDataExtractor初始化成功")
        print(f"  无人机名称: {extractor.drone_name}")
        print(f"  相机列表: {extractor.camera_names}")
        print(f"  数据目录: {extractor.data_dir}")
        return True
    except Exception as e:
        print(f"✗ RGBDataExtractor初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            extractor.disconnect()
            print("✓ 已断开连接")
        except:
            pass

def test_neural_perception():
    """测试NeuralPerception初始化"""
    print()
    print("=" * 60)
    print("测试3: NeuralPerception初始化")
    print("=" * 60)
    
    try:
        print("正在初始化NeuralPerception...")
        neural_perception = NeuralPerception()
        print("✓ NeuralPerception初始化成功")
        return True
    except Exception as e:
        print(f"✗ NeuralPerception初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visual_perception():
    """测试VisualPerception初始化"""
    print()
    print("=" * 60)
    print("测试4: VisualPerception初始化")
    print("=" * 60)
    
    try:
        print("正在初始化VisualPerception...")
        visual_perception = VisualPerception()
        print("✓ VisualPerception初始化成功")
        return True
    except Exception as e:
        print(f"✗ VisualPerception初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print()
    print("*" * 60)
    print("AirSim连接问题修复测试")
    print("*" * 60)
    print()
    
    results = []
    
    # 测试1: 导入
    results.append(("模块导入", test_imports()))
    
    # 测试2: RGBDataExtractor
    results.append(("RGBDataExtractor", test_rgb_extractor()))
    
    # 测试3: NeuralPerception
    results.append(("NeuralPerception", test_neural_perception()))
    
    # 测试4: VisualPerception
    results.append(("VisualPerception", test_visual_perception()))
    
    # 总结
    print()
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    print()
    print(f"总计: {passed}/{total} 通过")
    
    if passed == total:
        print()
        print("🎉 所有测试通过！修复成功！")
        print()
        print("现在可以运行 main.py 启动完整的视觉系统")
        return 0
    else:
        print()
        print("⚠ 部分测试失败，请检查上述错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())