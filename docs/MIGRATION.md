# 迁移指南

本文档提供从旧项目结构迁移到新项目结构的详细步骤。

## 目录

1. [迁移概述](#迁移概述)
2. [迁移前准备](#迁移前准备)
3. [迁移步骤](#迁移步骤)
4. [验证迁移](#验证迁移)
5. [回滚方案](#回滚方案)
6. [迁移检查清单](#迁移检查清单)

---

## 迁移概述

### 迁移原因

**旧项目结构的问题：**
- 配置文件分散，难以维护
- 文件组织不清晰
- Git管理效率低
- 大文件提交到Git导致同步慢

**新项目结构的优势：**
- 配置和工具模块放在根目录
- 代码集中在 `src/`
- 模型独立管理
- 文档集中到 `docs/`

### 迁移范围

**需要迁移的文件：**
1. `src/config_manager.py` → 根目录
2. `src/logging_utils.py` → 根目录
3. `项目可迁移性评估报告.md` → `docs/`
4. `项目迁移结构优化方案.md` → `docs/`
5. `导入问题修复报告.md` → `docs/`

**不需要迁移的文件：**
- `src/` 目录下的所有代码文件
- `models/` 目录下的所有文件（包括网盘传输）
- `src/` 目录下的其他文件

---

## 迁移前准备

### 1. 创建备份

**Git备份：**
```bash
# 查看当前Git状态
git status

# 提交当前更改
git add .
git commit -m "迁移前备份"

# 创建分支
git checkout -b migration
```

**文件备份：**
```bash
# 创建备份文件夹
mkdir backup_2026-02-15
mkdir backup_2026-02-15/src
mkdir backup_2026-02-15/docs

# 复制重要文件
cp src/config_manager.py backup_2026-02-15/
cp src/logging_utils.py backup_2026-02-15/
cp -r docs backup_2026-02-15/
```

### 2. 检查依赖

**确认依赖安装：**
```bash
# 确保虚拟环境已激活
source venv/bin/activate  # Windows: venv\Scripts\activate

# 测试导入
python test_imports.py
```

### 3. 确认AirSim配置

**检查AirSim安装：**
```bash
# 检查AirSim是否安装
ls AirSim/build/Release/
```

**验证AirSim连接：**
```python
from AirSim import AirSim
client = AirSimClient()
print(f'连接状态: {client.simGetConnectionState()}')
```

### 4. 准备文档

**阅读相关文档：**
- [项目结构说明](PROJECT_STRUCTURE.md)
- [环境配置指南](ENV_SETUP.md)

---

## 迁移步骤

### 步骤1：移动配置文件到根目录

**移动 `config_manager.py`：**
```bash
# Windows PowerShell
move src\config_manager.py .

# macOS/Linux
mv src/config_manager.py .
```

**移动 `logging_utils.py`：**
```bash
# Windows PowerShell
move src\logging_utils.py .

# macOS/Linux
mv src/logging_utils.py .
```

### 步骤2：创建docs文件夹（如果不存在）

```bash
# Windows PowerShell
mkdir docs

# macOS/Linux
mkdir docs
```

### 步骤3：移动文档到docs文件夹

**移动技术报告：**
```bash
# Windows PowerShell
move 项目可迁移性评估报告.md docs\
move 项目迁移结构优化方案.md docs\
move 导入问题修复报告.md docs\
```

**或使用git mv（推荐）：**
```bash
git mv 项目可迁移性评估报告.md docs/
git mv 项目迁移结构优化方案.md docs/
git mv 导入问题修复报告.md docs/
```

### 步骤4：更新.gitignore

**如果需要，更新 `.gitignore`：**
```bash
# 确保包含以下内容
echo "# 模型文件（网盘传输，不提交到Git）" >> .gitignore
echo "models/RAFT/models/*.pth" >> .gitignore
echo "models/monodepth2/mono+stereo_*/encoder.pth" >> .gitignore
```

### 步骤5：更新代码中的导入路径

**如果代码中引用了旧路径，需要更新：**

**示例：更新导入路径**

```python
# 旧代码（src/子目录内）
from config_manager import ConfigManager

# 新代码（根目录导入）
from config_manager import ConfigManager
```

**如果从 `src/` 子目录导入，需要更新：**

```python
# 旧代码
from src.config_manager import ConfigManager

# 新代码（使用相对导入或直接导入）
from config_manager import ConfigManager
```

### 步骤6：清理旧的config_manager.py和logging_utils.py

**删除src目录中的旧文件：**
```bash
# Windows PowerShell
Remove-Item src\config_manager.py
Remove-Item src\logging_utils.py

# macOS/Linux
rm src/config_manager.py
rm src/logging_utils.py
```

**使用git rm：**
```bash
git rm src/config_manager.py
git rm src/logging_utils.py
```

### 步骤7：测试导入

```bash
# 确保虚拟环境已激活
source venv/bin/activate

# 运行测试
python test_imports.py
```

---

## 验证迁移

### 1. 检查文件结构

```bash
# Windows PowerShell
dir /b

# macOS/Linux
ls
```

**预期结果：**
```
ANNUAL_PROJECT/
├── config_manager.py           ✅
├── logging_utils.py            ✅
├── requirements.txt            ✅
├── setup.py                    ✅
├── test_imports.py             ✅
├── run_vision.bat              ✅
├── README.md                   ✅
├── README_VISION.md            ✅
├── .gitignore                  ✅
├── docs/                       ✅
│   ├── PROJECT_STRUCTURE.md    ✅
│   ├── ENV_SETUP.md            ✅
│   ├── MIGRATION.md            ✅
│   ├── 项目可迁移性评估报告.md
│   ├── 项目迁移结构优化方案.md
│   └── 导入问题修复报告.md
├── src/                        ✅
├── models/                     ✅
└── venv/                       ✅
```

### 2. 验证Git状态

```bash
git status
```

**预期结果（无重要更改）：**
```
On branch migration
Changes not staged for commit:
  modified:   .gitignore
  deleted:    src/config_manager.py
  deleted:    src/logging_utils.py
  renamed:    项目可迁移性评估报告.md -> docs/项目可迁移性评估报告.md
  renamed:    项目迁移结构优化方案.md -> docs/项目迁移结构优化方案.md
  renamed:    导入问题修复报告.md -> docs/导入问题修复报告.md
```

### 3. 测试功能

```bash
# 测试导入
python test_imports.py

# 测试AirSim连接
python -c "from AirSim import AirSim; client = AirSimClient(); print(client.simGetConnectionState())"
```

### 4. 检查依赖

```bash
# 确保所有依赖都安装
pip list
```

---

## 回滚方案

### 方案1：使用Git回滚

```bash
# 查看提交历史
git log --oneline

# 回滚到迁移前
git checkout main
git reset --hard <迁移前的commit-hash>

# 或回到迁移分支
git checkout migration
```

### 方案2：恢复备份文件

```bash
# 恢复配置文件
cp backup_2026-02-15/config_manager.py src/
cp backup_2026-02-15/logging_utils.py src/

# 恢复文档
cp -r backup_2026-02-15/docs/* .
```

### 方案3：手动还原

1. 从 `docs/` 复制文件到根目录
2. 从备份文件夹复制文件到 `src/`
3. 更新导入路径

---

## 迁移检查清单

### 迁移前

- [ ] 创建Git备份
- [ ] 创建文件备份
- [ ] 检查依赖安装
- [ ] 确认AirSim配置
- [ ] 阅读相关文档

### 迁移中

- [ ] 移动 `config_manager.py` 到根目录
- [ ] 移动 `logging_utils.py` 到根目录
- [ ] 创建 `docs/` 文件夹
- [ ] 移动文档到 `docs/`
- [ ] 更新 `.gitignore`
- [ ] 更新代码中的导入路径
- [ ] 删除src中的旧文件

### 迁移后

- [ ] 验证文件结构
- [ ] 运行导入测试
- [ ] 测试AirSim连接
- [ ] 检查依赖安装
- [ ] 测试功能完整性

---

## 迁移示例

### 完整迁移脚本

```bash
#!/bin/bash
# migration_script.sh

echo "开始迁移..."

# 1. 创建备份
echo "创建备份..."
mkdir -p backup_2026-02-15/src
cp src/config_manager.py backup_2026-02-15/
cp src/logging_utils.py backup_2026-02-15/
cp -r docs backup_2026-02-15/

# 2. 移动配置文件
echo "移动配置文件..."
mv src/config_manager.py .
mv src/logging_utils.py .

# 3. 创建docs文件夹
echo "创建docs文件夹..."
mkdir -p docs

# 4. 移动文档
echo "移动文档..."
git mv 项目可迁移性评估报告.md docs/
git mv 项目迁移结构优化方案.md docs/
git mv 导入问题修复报告.md docs/

# 5. 删除旧文件
echo "删除旧文件..."
rm src/config_manager.py
rm src/logging_utils.py

# 6. 测试导入
echo "测试导入..."
python test_imports.py

echo "迁移完成！"
```

### Windows PowerShell脚本

```powershell
# migration.ps1
Write-Host "开始迁移..." -ForegroundColor Green

# 1. 创建备份
Write-Host "创建备份..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "backup_2026-02-15\src"
Copy-Item "src\config_manager.py" "backup_2026-02-15\"
Copy-Item "src\logging_utils.py" "backup_2026-02-15\"
Copy-Item "docs" "backup_2026-02-15\"

# 2. 移动配置文件
Write-Host "移动配置文件..." -ForegroundColor Yellow
Move-Item "src\config_manager.py" "."
Move-Item "src\logging_utils.py" "."

# 3. 创建docs文件夹
Write-Host "创建docs文件夹..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "docs"

# 4. 移动文档
Write-Host "移动文档..." -ForegroundColor Yellow
git mv "项目可迁移性评估报告.md" "docs\"
git mv "项目迁移结构优化方案.md" "docs\"
git mv "导入问题修复报告.md" "docs\"

# 5. 删除旧文件
Write-Host "删除旧文件..." -ForegroundColor Yellow
Remove-Item "src\config_manager.py"
Remove-Item "src\logging_utils.py"

# 6. 测试导入
Write-Host "测试导入..." -ForegroundColor Yellow
python test_imports.py

Write-Host "迁移完成！" -ForegroundColor Green
```

---

## 常见迁移问题

### Q1: Git无法移动文件

**问题：**
```bash
git mv: 无法移动 '文件名' -> '目标路径'
```

**解决方案：**
```bash
# 使用普通移动，然后添加
mv 文件名 目标路径
git add 目标路径/文件名
```

### Q2: 代码导入错误

**问题：**
```
ModuleNotFoundError: No module named 'config_manager'
```

**解决方案：**
```python
# 确保从根目录导入
from config_manager import ConfigManager
```

### Q3: 文件权限错误

**问题：**
```
PermissionError: [Errno 13] Permission denied
```

**解决方案：**
```bash
# Linux/Mac
chmod 755 文件名

# 或使用管理员权限
sudo mv 文件名 目标路径
```

---

## 迁移后的优化建议

### 1. 代码导入优化

```python
# 在src/目录的模块中，使用相对导入
from ..config_manager import ConfigManager
from ..logging_utils import get_logger
```

### 2. 配置文件优化

```python
# config_manager.py
# 将配置文件路径统一管理
BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "configs"
```

### 3. 文档组织优化

```
docs/
├── USER_GUIDE.md              # 用户指南
├── DEVELOPER_GUIDE.md         # 开发者指南
├── API_REFERENCE.md           # API参考
├── MIGRATION.md               # 迁移指南（本文件）
├── PROJECT_STRUCTURE.md       # 项目结构
├── ENV_SETUP.md               # 环境配置
└── TECHNICAL_REPORTS/         # 技术报告
    ├── project_migration.md
    └── issue_fixes.md
```

---

## 下一步

迁移完成后：

1. ✅ 创建迁移分支
2. ✅ 提交迁移更改
3. ✅ 推送到远程仓库
4. ✅ 合并到主分支
5. ✅ 更新项目文档
6. ✅ 通知团队成员

---

## 获取帮助

如果遇到迁移问题：

1. 查阅本文档的常见问题部分
2. 检查 [项目结构说明](PROJECT_STRUCTURE.md)
3. 查看Git历史记录
4. 联系项目维护者

---

## 更新日志

### 2026-02-15
- 初始版本
- 添加完整迁移步骤
- 添加回滚方案
- 添加检查清单