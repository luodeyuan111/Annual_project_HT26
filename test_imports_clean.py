"""
测试所有模块的导入是否正常
"""

import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("=" * 60)
print("测试模块导入")
print("=" * 60)

try:
    print("\n1. 测试 utils 模块导入...")
    from src.utils import get_logger, keyboard_control_process
    print("   ✓ utils 模块导入成功")
    
    print("\n2. 测试 Drone_Interface 模块导入...")
    from src.Drone_Interface.rgb_data_extractor import RGBDataExtractor, FrameBuffer
    print("   ✓ Drone_Interface 模块导入成功")
    
    print("\n3. 测试 neural_processing 模块导入...")
    from src.neural_processing.neural_perception import NeuralPerception
    print("   ✓ neural_processing 模块导入成功")
    
    print("\n4. 测试 Visual_process 模块导入...")
    # 使用exec来避免相对导入问题
    exec("import src.Visual_process.visual_center as vc")
    print("   ✓ Visual_process 模块导入成功")
    
    print("\n5. 测试 main.py 导入...")
    # 不真正运行main，只测试导入
    import src.main
    print("   ✓ main.py 导入成功")
    
    print("\n" + "=" * 60)
    print("所有模块导入测试通过！✓")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)