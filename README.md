# 哈尔滨工业大学2025级大一年度项目 - 航天26号

## 项目概述

本项目致力于实现AirSim环境以及实际环境下的无人机集群视觉感知以及集群控制。

## 快速开始

### 详细文档

👉 **完整使用说明请查看 [README_VISION.md](README_VISION.md)**

### 环境要求

- Python 3.8+
- Windows 10/11 或 Linux
- CUDA 11.0+ (如果使用GPU)
- AirSim仿真环境

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/luodeyuan111/code.git
cd annual_project
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **准备模型**
   - 下载RAFT模型：https://github.com/princeton-vl/RAFT
   - 下载MonoDepth2模型：https://github.com/nianticlabs/monodepth2
   - 放置到 `models/` 目录对应位置

4. **运行系统**
```bash
# Windows用户
run_vision.bat

# 或直接运行
python src/main.py
```

## 项目结构

```
annual_project/
├── models/                          # 深度学习模型
│   ├── RAFT/                       # 光流估计模型
│   └── monodepth2/                 # 深度估计模型
├── src/                            # 源代码
│   ├── neural_processing/          # 神经网络处理模块
│   ├── Visual_process/            # 视觉处理模块
│   ├── Drone_Interface/           # 无人机接口
│   ├── main.py                    # 主程序
│   ├── config_manager.py           # 配置管理
│   └── logging_utils.py           # 日志工具
├── README_VISION.md               # 详细使用文档 ⭐
├── requirements.txt                # 依赖包
└── run_vision.bat                 # 快速启动脚本
```

## 开发模式安装

如果需要开发模式安装（便于在其它电脑上复现）：

```powershell
python -m pip install -e .
```

运行测试脚本（推荐使用模块方式）：

```powershell
python -m src.main
```

## 键盘控制

运行程序后：
- `e` - 捕获并处理两帧图像
- `w/s` - 前进/后退
- `a/d` - 左移/右移
- `,/.` - 上升/下降
- `q` - 退出程序

## 主要功能

- ✅ 神经网络感知（RAFT光流 + MonoDepth2深度）
- ✅ 视觉处理（障碍物检测、位姿估计）
- ✅ 实时图像处理
- ✅ 模块化设计

## 联系方式

如有问题，请联系项目维护者。
