# 项目结构说明

## 最终项目结构

```
ANNUAL_PROJECT/
├── config_manager.py           # ✅ 配置管理模块（已移到根目录）
├── logging_utils.py            # ✅ 日志管理模块（已移到根目录）
├── requirements.txt            # 项目依赖
├── setup.py                    # 项目设置
├── test_imports.py             # 导入测试脚本
├── run_vision.bat              # 快速启动脚本
├── README.md                   # 项目说明
├── README_VISION.md            # 视觉处理说明
│
├── models/                     # 神经网络模型（网盘传输，不提交Git）
│   ├── RAFT/                   # 光流估计模型
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── raft.py
│   │   │   ├── extractor.py
│   │   │   ├── update.py
│   │   │   └── corr.py
│   │   ├── core/utils/
│   │   │   ├── __init__.py
│   │   │   ├── frame_utils.py
│   │   │   ├── augmentor.py
│   │   │   └── flow_viz.py
│   │   └── models/
│   │       └── raft-things.pth  # ⚠️ 大文件（忽略）
│   │
│   └── monodepth2/             # 深度估计模型
│       ├── networks/
│       │   ├── depth_decoder.py
│       │   ├── pose_cnn.py
│       │   ├── pose_decoder.py
│       │   └── resnet_encoder.py
│       ├── mono+stereo_640x192/
│       │   ├── encoder.pth     # ⚠️ 大文件（忽略）
│       │   ├── depth.pth       # ⚠️ 大文件（忽略）
│       │   └── ...
│       └── ...
│
├── src/                        # 源代码（GitHub同步）
│   ├── main.py                 # 主程序
│   ├── config_manager.py       # ⚠️ 不应在此（已移到根目录）
│   ├── logging_utils.py        # ⚠️ 不应在此（已移到根目录）
│   │
│   ├── neural_processing/      # 神经网络处理模块
│   │   ├── neural_perception.py  # 神经感知主模块
│   │   ├── flow_processor.py     # 光流处理器
│   │   ├── depth_estimator.py    # 深度估计器
│   │   ├── clustering.py         # 聚类模块
│   │   └── __init__.py
│   │
│   ├── Visual_process/         # 视觉处理模块
│   │   ├── visual_center.py     # 视觉处理中心
│   │   ├── obstacle_detect.py   # 障碍物检测
│   │   ├── pose_estimator.py    # 位姿估计
│   │   ├── types.py             # 类型定义
│   │   └── __init__.py
│   │
│   └── Drone_Interface/        # 无人机接口
│       ├── Drone_Interface.py  # 无人机接口
│       ├── rgb_data_extractor.py  # RGB数据提取器
│       └── settings.json.txt    # AirSim配置（示例）
│
├── integration_system/         # ⚠️ 需要清理
│   └── ...                     # 这个目录不再需要
│
├── docs/                       # 文档目录
│   ├── PROJECT_STRUCTURE.md    # 本文件
│   ├── ENV_SETUP.md            # 环境配置指南
│   ├── MIGRATION.md            # 迁移指南
│   ├── 项目可迁移性评估报告.md
│   ├── 项目迁移结构优化方案.md
│   └── 导入问题修复报告.md
│
├── venv/                       # 虚拟环境（不提交）
├── sensor_data/                # 传感器数据（不提交）
├── outputs/                    # 输出文件（不提交）
├── logs/                       # 日志文件（不提交）
└── .gitignore                  # Git忽略配置
```

## 文件组织原则

### 根目录文件

**配置和工具模块：**
- `config_manager.py` - 配置管理
- `logging_utils.py` - 日志管理
- `requirements.txt` - 依赖列表
- `test_imports.py` - 导入测试
- `setup.py` - 项目设置
- `run_vision.bat` - 快速启动脚本

**说明文档：**
- `README.md` - 项目总览
- `README_VISION.md` - 视觉处理说明

### src/ 目录（源代码）

**核心处理模块：**
- `neural_processing/` - 神经网络处理
- `Visual_process/` - 视觉处理
- `Drone_Interface/` - 无人机接口

### models/ 目录（模型文件）

**不提交到Git的大文件：**
- 所有 `.pth`, `.npy`, `.pt` 文件
- 所有模型压缩包

### docs/ 目录（文档）

**项目文档：**
- 项目结构说明
- 环境配置指南
- 迁移指南
- 技术评估报告

## 文件移动记录

### 2026-02-15

**从 `src/` 移动到根目录：**
- ✅ `config_manager.py`
- ✅ `logging_utils.py`

**从根目录移动到 `docs/`：**
- ✅ `项目可迁移性评估报告.md`
- ✅ `项目迁移结构优化方案.md`
- ✅ `导入问题修复报告.md`

**新增：**
- ✅ `.gitignore` 更新（忽略模型文件）

## Git管理策略

### 提交到Git的文件

**代码文件：**
- `src/` 目录下所有Python文件
- 根目录的 `requirements.txt`
- 根目录的 `setup.py`
- 根目录的 `test_imports.py`
- 根目录的 `run_vision.bat`
- 根目录的 `README.md`
- 根目录的 `README_VISION.md`
- 根目录的 `config_manager.py`
- 根目录的 `logging_utils.py`
- `.gitignore`

**文档文件：**
- `docs/` 目录下所有文件

**配置文件：**
- `src/Drone_Interface/settings.json.txt`（作为示例配置）

### 忽略的文件

**模型文件（网盘传输）：**
- `models/RAFT/models/*.pth`
- `models/RAFT/models/*.npy`
- `models/monodepth2/mono+stereo_*/encoder.pth`
- `models/monodepth2/mono+stereo_*/depth.pth`
- `*.zip`, `*.tar.gz`, `*.rar`

**临时文件：**
- `venv/`
- `sensor_data/`
- `outputs/`
- `logs/`
- `visualizations/`
- `__pycache__/`
- `*.pyc`
- `*.log`
- `.DS_Store`
- `Thumbs.db`

## 使用流程

### 1. 获取项目文件

```bash
# 网盘下载：models/ 目录
# GitHub拉取：整个项目（不包括models）
```

### 2. 配置环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装基础依赖
pip install -r requirements.txt

# 根据电脑配置安装PyTorch
# 详见 docs/ENV_SETUP.md
```

### 3. 配置AirSim

```bash
# 方式：手动配置
# 打开AirSim -> 配置blocks
```

### 4. 验证安装

```bash
# 测试导入
python test_imports.py

# 运行主程序
python src/main.py
```

## 项目优势

### ✅ 结构清晰

- 根目录：配置、工具、说明文档
- src/：核心代码
- models/：独立管理
- docs/：完整文档

### ✅ Git友好

- 大文件不提交
- 小文件快速同步
- 版本控制清晰

### ✅ 易于维护

- 配置文件独立
- 文档集中管理
- 代码模块化

### ✅ 团队协作

- 明确的文件组织
- 统一的依赖管理
- 清晰的文档指引

## 常见问题

### Q: 为什么config_manager.py和logging_utils.py在根目录？

A: 这两个是工具模块，不属于技术代码，更适合放在根目录作为全局配置和日志工具。

### Q: 为什么models不提交到Git？

A: 模型文件很大（数百MB），提交到Git会导致：
- 同步速度慢
- 存储浪费
- Git历史污染

### Q: 如何保持模型同步？

A: 使用网盘自动同步，或手动传输压缩包。


### Q: docs/目录的作用？

A: 集中存放所有文档，包括技术报告、配置指南、迁移文档等。

## 下一步

1. ✅ 完成文件移动
2. ✅ 更新.gitignore
3. ⏭️ 测试所有导入
4. ⏭️ 创建环境配置文档（ENV_SETUP.md）
5. ⏭️ 测试完整流程