"""
测试神经网络模型导入是否正常
"""
import sys
import os

print("=" * 60)
print("测试神经网络模型导入")
print("=" * 60)

# 测试RAFT导入
print("\n1. 测试RAFT导入...")
try:
    raft_path = os.path.join(os.path.dirname(__file__), 'models/RAFT')
    if raft_path not in sys.path:
        sys.path.append(raft_path)
    
    from core.raft import RAFT
    from core.utils.utils import InputPadder
    from core.utils import flow_viz
    print("✓ RAFT导入成功")
except Exception as e:
    print(f"✗ RAFT导入失败: {e}")
    import traceback
    traceback.print_exc()

# 测试MonoDepth2导入
print("\n2. 测试MonoDepth2导入...")
try:
    monodepth_path = os.path.join(os.path.dirname(__file__), 'models/monodepth2')
    if monodepth_path not in sys.path:
        sys.path.insert(0, monodepth_path)
    
    from networks import ResnetEncoder, DepthDecoder
    from layers import disp_to_depth
    from utils import download_model_if_doesnt_exist
    print("✓ MonoDepth2导入成功")
except Exception as e:
    print(f"✗ MonoDepth2导入失败: {e}")
    import traceback
    traceback.print_exc()

# 测试neural_processing模块导入
print("\n3. 测试neural_processing模块导入...")
try:
    from src.neural_processing.flow_processor import FlowProcessor
    from src.neural_processing.depth_estimator import Monodepth2Estimator
    print("✓ neural_processing模块导入成功")
except Exception as e:
    print(f"✗ neural_processing模块导入失败: {e}")
    import traceback
    traceback.print_exc()

# 测试torch是否可用
print("\n4. 测试PyTorch...")
try:
    import torch
    print(f"✓ PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ PyTorch测试失败: {e}")

print("\n" + "=" * 60)
print("导入测试完成")
print("=" * 60)