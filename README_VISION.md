# AirSim无人机视觉处理系统 - 使用说明

## 项目概述

这是一个基于AirSim的无人机视觉处理项目，集成了深度学习模型和传统计算机视觉算法，用于实时处理无人机捕获的图像数据，输出环境感知信息供集群控制系统使用。

### 主要功能

1. **神经网络感知**：集成RAFT光流估计和MonoDepth2深度估计
2. **视觉处理**：障碍物检测、位姿估计、特征提取
3. **实时处理**：支持连续两帧图像处理
4. **模块化设计**：各模块独立，易于维护和扩展

## 项目结构

```
annual_project/
├── models/                          # 深度学习模型
│   ├── RAFT/                       # 光流估计模型
│   └── monodepth2/                 # 深度估计模型
├── src/                            # 源代码
│   ├── neural_processing/          # 神经网络处理模块
│   │   ├── neural_perception.py    # 神经网络感知主模块
│   │   ├── flow_processor.py       # 光流处理器
│   │   ├── depth_estimator.py     # 深度估计器
│   │   ├── clustering.py          # 传统分割器
│   │   └── __init__.py
│   ├── Visual_process/            # 视觉处理模块
│   │   ├── visual_center.py       # 视觉处理中心
│   │   ├── obstacle_detect.py     # 障碍物检测
│   │   ├── pose_estimator.py      # 位姿估计
│   │   └── types.py               # 数据类型定义
│   ├── Drone_Interface/           # 无人机接口
│   │   ├── Drone_Interface.py     # AirSim接口
│   │   └── rgb_data_extractor.py  # RGB数据提取器
│   ├── main.py                    # 主程序
│   ├── config_manager.py           # 配置管理
│   └── logging_utils.py           # 日志工具
├── sensor_data/                   # 传感器数据存储
├── logs/                          # 日志文件
└── requirements.txt                # 依赖包
```

## 环境配置

### 1. 系统要求

- Python 3.8+
- Windows 10/11 或 Linux
- CUDA 11.0+ (如果使用GPU)
- AirSim仿真环境

### 2. 安装依赖

```bash
# 克隆项目
git clone https://github.com/luodeyuan111/code.git
cd annual_project

# 安装Python依赖
pip install -r requirements.txt
```

### 3. 模型准备

项目使用以下预训练模型：

1. **RAFT光流模型** (`models/RAFT/models/raft-things.pth`)
   - 下载地址：https://github.com/princeton-vl/RAFT
   - 将模型文件放入 `models/RAFT/models/` 目录

2. **MonoDepth2深度模型** (`models/monodepth2/mono+stereo_640x192`)
   - 下载地址：https://github.com/nianticlabs/monodepth2
   - 将模型文件夹放入 `models/monodepth2/` 目录

### 4. AirSim配置

确保AirSim正在运行，并且至少配置了一个摄像头（默认：front_camera）。

## 使用方法

### 基本使用

```bash
# 运行主程序
python src/main.py
```

### 键盘控制

运行程序后，可以使用以下键盘控制：

- `e`: 捕获并处理两帧图像（间隔100ms）
- `w`: 前进
- `s`: 后退
- `a`: 左移
- `d`: 右移
- `,`: 上升
- `.`: 下降
- `q`: 退出程序

### 输出示例

```
处理帧对 1 (NeuralPerception + VisualPerception pipeline)...
帧 2 处理完成:
  特征点: 3416, 分割区域: 3
  质量: 1.000
✓ 神经网络质量: 1.000
✓ 特征点数量: 3416
✓ 分割区域: 3416
✓ 全局运动: dx=0.00, dy=0.00, dz=0.00 (置信度: 0.000)
✓ 最近障碍物: 2.4m
✓ 障碍物角度: 135.0°
✓ 质量指标: {...}
✓ 警告信息: []
✓ 处理完成，输出已序列化
```

## 配置说明

### 使用配置文件

创建YAML配置文件（如 `config/my_config.yaml`）：

```yaml
device:
  device_type: cpu  # 或 cuda
  use_mixed_precision: false

neural:
  raft:
    model_path: models/RAFT/models/raft-things.pth
    iterations: 20
  monodepth2:
    model_path: models/monodepth2/mono+stereo_640x192
    input_width: 640
    input_height: 192

vision:
  camera:
    fx: 320.0  # 焦距
    fy: 320.0
    cx: 320.0  # 主点X
    cy: 240.0  # 主点Y
  obstacle:
    max_distance: 10.0
    fov_horizontal: 90.0
    safety_distance: 2.0

logging:
  level: INFO
  log_to_file: true
  log_dir: logs
```

### 在代码中使用配置

```python
from config_manager import ConfigManager

# 加载配置
config = ConfigManager('config/my_config.yaml')

# 获取配置项
device = config.get('device.device_type')
raft_iterations = config.get('neural.raft.iterations')

# 动态设置配置
config.set('device.device_type', 'cuda')
```

## 日志系统

### 使用日志

```python
from logging_utils import get_logger

# 获取logger
logger = get_logger(
    log_dir='logs',
    log_file='drone_vision.log',
    log_level='INFO'
)

# 记录日志
logger.info("处理开始")
logger.debug("调试信息")
logger.warning("警告信息")
logger.error("错误信息")
```

### 日志级别

- `DEBUG`: 详细的调试信息
- `INFO`: 一般信息
- `WARNING`: 警告信息
- `ERROR`: 错误信息
- `CRITICAL`: 严重错误

## 模块说明

### NeuralPerception (神经网络感知)

位于 `src/neural_processing/neural_perception.py`

**功能**：
- 计算光流（RAFT）
- 估计深度（MonoDepth2）
- 提取特征点
- 场景分割
- 质量评估

**使用示例**：
```python
from neural_processing.neural_perception import NeuralPerception

# 初始化
perception = NeuralPerception()

# 处理图像对
neural_output = perception.process_frame_pair(frame_t, frame_t_plus_1)

# 访问结果
print(f"特征点数量: {neural_output.feature_points['n_features']}")
print(f"深度图形状: {neural_output.depth_maps['depth_t'].shape}")
```

### VisualPerception (视觉处理中心)

位于 `src/Visual_process/visual_center.py`

**功能**：
- 障碍物检测和极坐标映射
- 位姿估计
- 质量评估
- 输出标准化

**使用示例**：
```python
from Visual_process.visual_center import VisualPerception

# 初始化
visual = VisualPerception()

# 处理神经网络输出
visual_state = visual.process(neural_output)

# 访问结果
print(f"最近障碍物: {visual_state.obstacle_frame.closest_depth}m")
print(f"运动: {visual_state.ego_motion.translation}")
```

## 输出数据格式

### VisualState (视觉状态)

```python
@dataclass
class VisualState:
    timestamp: float                    # 时间戳
    frame_idx: int                     # 帧索引
    
    obstacle_frame: ObstaclePolarFrame # 障碍物信息
    ego_motion: Optional[Pose6D]        # 自身运动
    quality: QualityMetrics            # 质量指标
    warnings: List[str]                # 警告列表
    
    def to_payload(self) -> dict:       # 序列化为字典
        ...
```

### ObstaclePolarFrame (障碍物极坐标)

```python
@dataclass
class ObstaclePolarFrame:
    angles: np.ndarray          # 角度数组 [360]
    depths: np.ndarray          # 深度数组 [360]
    safety_mask: np.ndarray     # 安全区域掩码
    forbidden_mask: np.ndarray  # 禁飞区域掩码
    closest_angle: float        # 最近障碍物角度
    closest_depth: float        # 最近障碍物距离
    ...
```

## 性能优化

### CPU优化

- 减少RAFT迭代次数（默认20次，可降到10-15次）
- 降低输入分辨率（640x192可降到320x96）
- 增加特征网格步长（默认8，可增加到16）

### GPU优化

- 使用CUDA加速
- 启用混合精度训练
- 批量处理多帧

### 示例配置

```python
# 高性能配置（需要GPU）
config = {
    'device': 'cuda',
    'raft_iterations': 20,
    'input_width': 640,
    'input_height': 192,
    'feature_grid_step': 8
}

# 低功耗配置（CPU）
config = {
    'device': 'cpu',
    'raft_iterations': 10,
    'input_width': 320,
    'input_height': 96,
    'feature_grid_step': 16
}
```

## 故障排除

### 问题1：模型导入错误

**现象**：
```
导入RAFT失败: No module named 'core'
导入Monodepth2失败: No module named 'networks'
```

**解决方案**：
1. 确保模型目录结构正确：
```
annual_project/
├── models/
│   ├── RAFT/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── raft.py
│   │   │   └── utils/
│   │   │       └── __init__.py
│   │   └── ...
│   └── monodepth2/
│       ├── __init__.py
│       ├── networks/
│       │   ├── __init__.py
│       │   └── ...
│       ├── layers.py
│       └── utils.py
```

2. 运行测试脚本验证：
```bash
python test_imports.py
```

3. 如果仍然失败，检查：
   - Python版本是否为3.8+
   - 是否已安装所有依赖：`pip install -r requirements.txt`
   - 确保从项目根目录运行脚本

### 问题2：找不到torch模块

**解决方案**：
```bash
pip install torch torchvision
```

### 问题2：模型文件不存在

**解决方案**：
1. 检查模型路径是否正确
2. 下载预训练模型并放入正确位置
3. 参考模型准备部分的链接

### 问题3：AirSim连接失败

**解决方案**：
1. 确保AirSim正在运行
2. 检查IP地址和端口（默认：127.0.0.1:41451）
3. 检查防火墙设置

### 问题4：内存不足

**解决方案**：
1. 降低输入分辨率
2. 减少RAFT迭代次数
3. 使用CPU而非GPU
4. 关闭不必要的程序

## 扩展开发

### 添加新的神经网络模型

1. 在 `src/neural_processing/` 创建新模块
2. 在 `NeuralPerception` 中集成
3. 更新配置文件

### 自定义视觉算法

1. 在 `src/Visual_process/` 添加新模块
2. 在 `VisualPerception` 中调用
3. 定义输出数据格式

## 已知问题

1. **Torch警告**：`torch.cuda.amp.autocast` 已弃用
   - 影响：不影响功能，只是警告
   - 解决方案：将在后续版本更新

2. **RuntimeWarning: overflow encountered in scalar add**
   - 影响：轻微，不影响功能
   - 解决方案：将在后续版本修复

## 更新日志

### v1.1.0 (2026-02-15)
- ✅ 重构项目结构，优化目录组织
- ✅ 集成配置管理系统
- ✅ 添加日志系统
- ✅ 移除冗余模块
- ✅ 更新导入路径
- ✅ 完善文档

### v1.0.0 (初始版本)
- 实现基本的视觉处理功能
- 集成RAFT和MonoDepth2模型
- 实现障碍物检测和位姿估计

## 贡献指南

欢迎提交问题和改进建议！

## 许可证

[待补充]

## 联系方式

如有问题，请联系项目维护者。