# 项目结构说明文档

## 概述
本文档描述了无人机视觉感知系统的项目结构、模块组织和导入依赖关系。

## 项目根目录结构

```
annual_project/
├── src/                          # 源代码目录
│   ├── main.py                   # 主程序入口
│   ├── utils/                    # 工具模块（新增）
│   │   ├── __init__.py
│   │   ├── keyboard_control.py   # 键盘控制进程
│   │   └── logging_utils.py      # 日志系统
│   ├── Drone_Interface/          # 无人机接口模块
│   │   ├── Drone_Interface.py
│   │   └── rgb_data_extractor.py # RGB数据提取器
│   ├── neural_processing/        # 神经网络处理模块
│   │   ├── neural_perception.py  # 神经网络感知
│   │   ├── depth_estimator.py    # 深度估计
│   │   └── flow_processor.py     # 光流处理
│   └── Visual_process/           # 视觉处理模块
│       ├── visual_center.py      # 视觉处理中心
│       ├── obstacle_detect.py    # 障碍物检测
│       ├── pose_estimator.py     # 位姿估计
│       └── types.py              # 数据类型定义
├── models/                       # 模型目录
│   ├── monodepth2/              # MonoDepth2深度估计模型
│   └── RAFT/                    # RAFT光流模型
├── integration_system/           # 集成系统
├── docs/                         # 文档目录
└── logs/                         # 日志目录（运行时生成）
```

## 模块说明

### 1. src/main.py
**功能**：主程序入口，多进程架构
- 视觉处理进程（主进程）
- 键盘控制进程（独立进程）
- 通过Queue进行进程间通信

**导入依赖**：
```python
from Drone_Interface.rgb_data_extractor import RGBDataExtractor, FrameBuffer
from neural_processing.neural_perception import NeuralPerception
from Visual_process.visual_center import VisualPerception
from src.utils.logging_utils import get_logger
from src.utils.keyboard_control import keyboard_control_process
```

### 2. src/utils/ (新增工具模块)

#### 2.1 src/utils/__init__.py
**功能**：工具包模块初始化
- 导出 `get_logger` 和 `keyboard_control_process`

#### 2.2 src/utils/logging_utils.py
**功能**：统一的日志管理系统
- 单例模式的日志管理器
- 支持控制台和文件输出
- 提供便捷的日志记录方法

#### 2.3 src/utils/keyboard_control.py
**功能**：键盘控制进程
- 独立监听键盘输入
- 通过Queue发送命令给主进程
- 支持多键同时按下

### 3. src/Drone_Interface/

#### 3.1 Drone_Interface.py
无人机接口主模块

#### 3.2 rgb_data_extractor.py
**功能**：RGB数据提取器
- 从AirSim捕获RGB图像
- 帧缓冲区管理
- 图像保存功能

### 4. src/neural_processing/

#### 4.1 neural_perception.py
**功能**：神经网络感知模块
- 集成深度估计和光流处理
- 特征点提取
- 图像分割

#### 4.2 depth_estimator.py
**功能**：深度估计器
- 使用MonoDepth2模型
- 生成深度图

#### 4.3 flow_processor.py
**功能**：光流处理器
- 使用RAFT模型
- 计算帧间光流

### 5. src/Visual_process/

#### 5.1 visual_center.py
**功能**：视觉处理中心
- 整合障碍物检测和位姿估计
- 生成VisualState输出

#### 5.2 obstacle_detect.py
**功能**：障碍物检测
- 极坐标映射
- 历史数据融合
- 生成障碍物极坐标帧

#### 5.3 pose_estimator.py
**功能**：位姿估计
- 基于背景特征点的运动估计
- 使用RANSAC进行鲁棒估计

#### 5.4 types.py
**功能**：数据类型定义
- VisualState
- EgoMotion
- ObstaclePolarFrame
等数据结构

## 导入规则

### 绝对导入优先
为了与 `models/monodepth2/utils.py` 避免冲突，项目采用以下导入策略：

1. **main.py中的工具模块导入**：
```python
from src.utils.logging_utils import get_logger
from src.utils.keyboard_control import keyboard_control_process
```

2. **模块间的相对导入**：
在src目录下的模块中，使用以下方式：
```python
from neural_processing.neural_perception import NeuralOutput
from Visual_process.types import VisualState
```

3. **避免的导入方式**：
```python
# 不推荐（可能与models/monodepth2/utils.py冲突）
from utils import get_logger
```

### 相对导入问题
Visual_process模块内部的相对导入（`from .obstacle_detect`）在某些测试环境下会失败，因此使用了try-except回退机制：

```python
try:
    from .obstacle_detect import ObstacleProcessor
    from .pose_estimator import BackgroundPoseEstimator
    from .types import VisualState, EgoMotion, ObstaclePolarFrame
except Exception:
    # Fallback imports for local development
    from Visual_process.obstacle_detect import ObstacleProcessor
    from Visual_process.pose_estimator import BackgroundPoseEstimator
    from Visual_process.types import VisualState, EgoMotion, ObstaclePolarFrame
```

## 多进程架构

### 进程1：视觉处理进程
- 捕获和处理帧
- 通过Queue接收键盘命令
- 显示图像和结果

### 进程2：键盘控制进程
- 独立监听键盘输入
- 发送命令到Queue
- 不阻塞主进程

### 进程间通信
使用 `multiprocessing.Queue` 和 `multiprocessing.Value` 进行通信：

```python
command_queue = Queue()  # 命令队列
should_stop = Value('b', False)  # 共享停止标志
```

## 运行方式

### 方式1：直接运行main.py
```bash
cd d:\bian_cheng\code\annual_project
python src/main.py
```

### 方式2：作为模块运行
```bash
cd d:\bian_cheng\code\annual_project
python -m src.main
```

## 键盘控制

- `w` - 前进
- `s` - 后退
- `a` - 左移
- `d` - 右移
- `,` - 上升
- `.` - 下降
- `e` - 捕获并处理两帧（间隔100ms）
- `q` - 退出

## 日志系统

日志文件存储在 `logs/` 目录下：
- `logs/main.log` - 主程序日志
- `logs/visual.log` - 视觉处理日志

日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL

## 依赖关系图

```
main.py
├── utils/logging_utils
├── utils/keyboard_control
├── Drone_Interface/rgb_data_extractor
├── neural_processing/neural_perception
│   ├── depth_estimator (models/monodepth2)
│   └── flow_processor (models/RAFT)
└── Visual_process/visual_center
    ├── obstacle_detect
    ├── pose_estimator
    └── types
```

## 注意事项

1. **路径冲突**：`models/monodepth2/utils.py` 与 `src/utils/` 的命名冲突已解决，使用绝对导入 `from src.utils.xxx`。

2. **相对导入**：Visual_process模块内部使用相对导入，在测试时可能需要绝对导入的回退机制。

3. **AirSim连接**：运行前需要确保AirSim已启动。

4. **模型权重**：深度估计和光流模型需要预先下载权重文件。

## 更新日志

### 2026-02-22
- 创建 `src/utils/` 工具模块
- 移动 `keyboard_control.py` 到 `src/utils/`
- 移动 `logging_utils.py` 到 `src/utils/`
- 修复所有导入路径问题
- 解决与 `models/monodepth2/utils.py` 的命名冲突
- 通过Python语法检查
- **修复Monodepth2导入冲突**：在 `depth_estimator.py` 中将 `models/monodepth2` 路径插入到 `sys.path` 最前面，确保优先级高于 `src/utils/`
