# project_new\src\modules\__init__.py
"""
视觉处理模块包
"""

# 尝试导入所有模块，如果某些模块不可用则跳过
try:
    from .neural_perception import NeuralPerception, NeuralOutput
    _NEURAL_AVAILABLE = True
except ImportError as e:
    _NEURAL_AVAILABLE = False
    print(f"警告: NeuralPerception 不可用 ({e})")

try:
    from .depth_estimator import Monodepth2Estimator
    _DEPTH_AVAILABLE = True
except ImportError as e:
    _DEPTH_AVAILABLE = False
    print(f"警告: Monodepth2Estimator 不可用 ({e})")

try:
    from .clustering import TraditionalSegmenter
    _CLUSTERING_AVAILABLE = True
except ImportError:
    _CLUSTERING_AVAILABLE = False

try:
    from .flow_processor import FlowProcessor
    _FLOW_AVAILABLE = True
except ImportError:
    _FLOW_AVAILABLE = False

__all__ = [
    'NeuralPerception' if _NEURAL_AVAILABLE else None,
    'NeuralOutput' if _NEURAL_AVAILABLE else None,
    'Monodepth2Estimator' if _DEPTH_AVAILABLE else None,
    'TraditionalSegmenter' if _CLUSTERING_AVAILABLE else None,
    'FlowProcessor' if _FLOW_AVAILABLE else None
]

# 过滤掉 None 值
__all__ = [item for item in __all__ if item is not None]
