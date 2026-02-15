# 环境配置指南

本文档提供详细的环境配置步骤，包括Python环境、依赖安装、PyTorch配置等。

## 目录

1. [系统要求](#系统要求)
2. [创建虚拟环境](#创建虚拟环境)
3. [安装基础依赖](#安装基础依赖)
4. [安装PyTorch](#安装pytorch)
5. [配置AirSim](#配置airsim)
6. [验证安装](#验证安装)
7. [常见问题](#常见问题)

---

## 系统要求

### 硬件要求

**最低配置：**
- CPU: Intel i5 / AMD Ryzen 5 或更高
- 内存: 8GB RAM
- GPU: N/A（使用CPU版本）
- 硬盘: 10GB 可用空间

**推荐配置：**
- CPU: Intel i7 / AMD Ryzen 7 或更高
- 内存: 16GB RAM
- GPU: NVIDIA GTX 1060 或更高（支持CUDA）
- 硬盘: 20GB 可用空间

### 软件要求

**操作系统：**
- Windows 10/11
- macOS 10.15+
- Ubuntu 18.04/20.04/22.04

**软件版本：**
- Python 3.8 - 3.11
- Git 2.0+
- （可选）Visual Studio Code / PyCharm

---

## 创建虚拟环境

### Windows

```powershell
# 在项目根目录执行
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate

# 验证
python --version
```

### macOS/Linux

```bash
# 在项目根目录执行
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 验证
python3 --version
```

### 验证虚拟环境

```bash
# 检查Python版本（应在3.8-3.11之间）
python --version

# 检查pip版本
pip --version

# 检查是否在虚拟环境中
which python  # macOS/Linux
where python  # Windows
```

---

## 安装基础依赖

### 更新pip

```bash
# 升级pip到最新版本
pip install --upgrade pip

# 确保pip版本在20.0以上
pip --version
```

### 安装依赖包

```bash
# 安装项目依赖
pip install -r requirements.txt
```

### 依赖清单

以下是项目所需的主要依赖：

```txt
# 核心框架
numpy>=1.19.0
opencv-python>=4.5.0
Pillow>=8.0.0

# 深度学习框架
torch>=1.9.0
torchvision>=0.10.0

# 神经网络库
PyYAML>=5.4.0

# 视觉处理
scikit-image>=0.18.0

# 数据处理
pandas>=1.2.0

# 进度条
tqdm>=4.50.0

# 可视化
matplotlib>=3.3.0
seaborn>=0.11.0

# 日志
colorlog>=6.2.0

# 配置管理
pyyaml>=5.4.0
```

---

## 安装PyTorch

### 检查CUDA支持

#### Windows

1. **打开NVIDIA控制面板**
2. **检查驱动版本**
3. **确认支持CUDA版本**

建议使用PyTorch CPU版本（更简单），如需GPU加速请继续。

#### macOS

```bash
# 检查GPU支持
python -c "import torch; print(torch.cuda.is_available())"
```

#### Linux

```bash
# 检查GPU支持
nvidia-smi
```

### 安装PyTorch

#### 方式1：使用官方安装脚本（推荐）

访问 [PyTorch官网](https://pytorch.org/get-started/locally/) 获取最新的安装命令。

**示例：CPU版本（Windows）**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**示例：GPU版本（CUDA 11.8）**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**示例：GPU版本（CUDA 12.1）**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 方式2：手动安装

```bash
# CPU版本
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# GPU版本（CUDA 11.8）
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```

### 验证PyTorch安装

```bash
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU设备: {torch.cuda.get_device_name(0)}')
else:
    print('使用CPU版本')
"
```

---

## 配置AirSim

### 下载AirSim

1. **访问AirSim官网**
   https://github.com/microsoft/AirSim

2. **克隆AirSim**

```bash
git clone https://github.com/microsoft/AirSim.git
```

3. **编译AirSim**

```bash
cd AirSim
# Windows: 打开AirSim.sln，使用Visual Studio编译
# macOS: 按照README说明编译
# Linux: 按照README说明编译
```

### 配置AirSim客户端

#### 方式1：自动配置（推荐）

```bash
# 运行自动配置脚本
python src/Drone_Interface/airsim_config.py
```

#### 方式2：手动配置

1. **打开AirSim**

```bash
# Windows
start AirSim\AirSim.exe

# macOS/Linux
./AirSim/build/Release/AirSim
```

2. **配置参数**

在AirSim界面中：
- **Settings** → **Vehicle 1**: 选择 "Drone"
- **Settings** → **Physics**: 确保启用
- **Settings** → **Sensors**: 配置摄像头参数
  - RGB Camera: 分辨率 640x480, 帧率 30fps
  - Depth Camera: 分辨率 640x480, 帧率 30fps
- **Map**: 选择合适的地图

3. **保存配置**

在AirSim中保存当前配置。

### 验证AirSim连接

```bash
python -c "
from AirSim import AirSim
client = AirSimClient()
print(f'连接状态: {client.simGetConnectionState()}')
state = client.simGetVehiclePose()
print(f'无人机位置: {state.position}')
print(f'无人机朝向: {state.orientation}')
"
```

---

## 验证安装

### 运行导入测试

```bash
python test_imports.py
```

预期输出：

```
============================================================
测试神经网络模型导入
============================================================

1. 测试RAFT导入...
✓ RAFT导入成功

2. 测试MonoDepth2导入...
✓ MonoDepth2导入成功

3. 测试neural_processing模块导入...
✓ neural_processing模块导入成功

4. 测试PyTorch...
✓ PyTorch版本: 2.2.2+cpu
  CUDA可用: False
```

### 运行主程序测试

```bash
# 确保虚拟环境已激活
source venv/bin/activate  # Windows: venv\Scripts\activate

# 运行主程序（需要AirSim连接）
python src/main.py
```

### 检查摄像头配置

```bash
python -c "
from src.Drone_Interface.Drone_Interface import AirSimClient
client = AirSimClient()
camera = client.simGetCameraInfo('0')
print(f'摄像头名称: {camera.name}')
print(f'分辨率: {camera.resolution}')
print(f'帧率: {camera.capture_rate}')
"
```

---

## 常见问题

### Q1: pip安装失败，提示"Connection timeout"

**解决方案：**
```bash
# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用阿里云镜像
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### Q2: PyTorch安装后，import torch失败

**解决方案：**
```bash
# 1. 卸载现有PyTorch
pip uninstall torch torchvision torchaudio

# 2. 重新安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. 验证
python -c "import torch; print(torch.__version__)"
```

### Q3: CUDA不可用，但显卡支持CUDA

**解决方案：**
```bash
# 1. 安装正确的CUDA版本
# 检查NVIDIA驱动支持的CUDA版本
nvidia-smi

# 2. 安装对应版本的PyTorch
# 例如：CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Q4: AirSim无法启动

**解决方案：**
1. **检查AirSim安装**
   ```bash
   cd AirSim
   ls build/Release/
   ```

2. **查看AirSim日志**
   - Windows: `AppData\Local\Packages\Microsoft.AirSim_8wekyb3d8bbwe\LocalState\AirSim\logs\`
   - macOS/Linux: `~/.local/share/AirSim/logs/`

3. **重新编译AirSim**
   ```bash
   cd AirSim
   # Windows: 打开AirSim.sln，重新编译
   # macOS/Linux: 按照README说明重新编译
   ```

### Q5: 摄像头数据获取失败

**解决方案：**
```python
from src.Drone_Interface.Drone_Interface import AirSimClient

client = AirSimClient()

# 检查摄像头列表
cameras = client.simGetAllCameraInfo()
print("可用的摄像头:", [cam.name for cam in cameras])

# 检查无人机连接
state = client.simGetVehiclePose()
print("无人机位置:", state.position)
```

### Q6: 内存不足

**解决方案：**
```bash
# 1. 增加虚拟内存
# Windows: 系统属性 -> 高级 -> 性能 -> 高级 -> 虚拟内存
# macOS: 系统设置 -> 通用 -> 關於此Mac -> 更多信息 -> 内存

# 2. 降低图像分辨率
# 修改 src/main.py 中的配置
IMAGE_WIDTH = 640   # 从1280降低
IMAGE_HEIGHT = 480  # 从720降低
```

### Q7: TensorFlow与PyTorch冲突

**解决方案：**
```bash
# 确保只安装PyTorch
pip uninstall tensorflow
pip install torch torchvision torchaudio
```

---

## 性能优化建议

### GPU加速

```python
import torch

# 检查GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'使用设备: {device}')
```

### 内存优化

```python
import torch

# 使用半精度（如果GPU支持）
model.half()

# 启用内存优化
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
```

### 数据加载优化

```python
from torch.utils.data import DataLoader

# 使用DataLoader加速
dataloader = DataLoader(
    dataset,
    batch_size=32,      # 根据GPU内存调整
    shuffle=True,
    num_workers=4,      # 多进程加载
    pin_memory=True     # 加速数据传输
)
```

---

## 更新日志

### 2026-02-15
- 初始版本
- 添加Windows/Mac/Linux配置说明
- 添加常见问题解决方案

---

## 参考资源

- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [AirSim官方文档](https://microsoft.github.io/AirSim/)
- [Ubuntu安装指南](https://docs.conda.io/en/latest/miniconda.html)
- [Python环境管理](https://docs.python-guide.org/)

---

## 获取帮助

如果遇到问题，请：
1. 查阅本文档的常见问题部分
2. 检查GitHub Issues
3. 联系项目维护者