# 日志系统使用指南

## 概述

本项目的日志系统提供了统一的日志记录功能，所有关键信息都会自动记录到文件和终端。

## 日志文件位置

```
logs/drone_vision.log
```

## 日志级别

### DEBUG (调试信息)
- 用于调试和开发阶段
- 包含详细的处理流程信息
- 默认不显示（使用 INFO 级别）

**示例：**
```
2026-02-15 15:45:30 - DroneVision - DEBUG - 开始捕获第1帧，时间戳: 1707999930000
2026-02-15 15:45:30 - DroneVision - DEBUG - 等待100ms后捕获第2帧...
```

### INFO (信息)
- 标准运行信息
- 程序启动、初始化、按键检测等
- 这是主要显示的级别

**示例：**
```
2026-02-15 15:45:30 - DroneVision - INFO - 无人机视觉感知系统启动
2026-02-15 15:45:30 - DroneVision - INFO - 按键检测: ASCII=119, 字符=w
2026-02-15 15:45:30 - DroneVision - INFO - 移动指令: 前进
2026-02-15 15:45:30 - DroneVision - INFO - [第1帧] 捕获成功 - 分辨率: (480, 640, 3), 时间戳: 1707999930000
2026-02-15 15:45:30 - DroneVision - INFO - ✓ 神经网络处理完成 - 质量: 0.952, 特征点: 156, 分割区域: 3, 时间: 1.234s
```

### WARNING (警告)
- 非关键错误
- 可能影响功能的异常情况

**示例：**
```
2026-02-15 15:45:30 - DroneVision - WARNING - ⚠ 第1帧捕获成功，但帧缓冲区尚未准备好
2026-02-15 15:45:30 - DroneVision - WARNING - ⚠ 帧数据不足，无法处理
```

### ERROR (错误)
- 错误和异常
- 会导致程序功能失败

**示例：**
```
2026-02-15 15:45:30 - DroneVision - ERROR - 警告：无法获取相机 front_camera 的第1帧
2026-02-15 15:45:30 - DroneVision - ERROR - ⚠ 视觉处理失败 - 错误: [异常详细信息]
```

### CRITICAL (严重错误)
- 严重错误
- 需要立即处理

**示例：**
```
2026-02-15 15:45:30 - DroneVision - CRITICAL - 无法连接到AirSim，请确保AirSim已启动
```

## 日志内容分类

### 1. 程序启动
```python
logger.info("无人机视觉感知系统启动")
logger.info(f"启动时间: {datetime.now()}")
logger.info(f"项目根目录: {project_root}")
logger.info(f"Python版本: {sys.version.split()[0]}")
```

### 2. 初始化
```python
logger.info("开始初始化...")
logger.info("帧缓冲区初始化完成")
logger.info("神经网络和视觉处理模块初始化完成")
logger.info("显示窗口已创建")
```

### 3. 按键操作
```python
logger.info(f"按键检测: ASCII={key}, 字符={chr(key)}")
logger.info("移动指令: 前进")
logger.info("移动指令: 后退")
```

### 4. 帧捕获
```python
logger.debug(f"开始捕获第1帧，时间戳: {timestamp1}")
logger.info(f"[第1帧] 捕获成功 - 分辨率: {frame1.shape}, 时间戳: {timestamp1}")
logger.debug("等待100ms后捕获第2帧...")
logger.info(f"[第2帧] 捕获成功 - 分辨率: {frame2.shape}, 时间戳: {timestamp2}")
```

### 5. 视觉处理
```python
logger.info(f"开始处理帧对 {frame_counter} (NeuralPerception + VisualPerception pipeline)...")
logger.info(f"✓ 神经网络处理完成 - 质量: {quality:.3f}, 特征点: {features}, 分割区域: {segments}, 时间: {process_time:.3f}s")
logger.info(f"✓ 机器人运动 - 位移: dx={t[0]:.2f}, dy={t[1]:.2f}, dz={t[2]:.2f}, 置信度: {motion.confidence:.3f}")
logger.info(f"✓ 障碍物检测 - 距离: {visual_state.obstacle_frame.closest_depth:.1f}m, 角度: {visual_state.obstacle_frame.closest_angle:.1f}°")
logger.info(f"✓ 质量指标 - 评分: {visual_state.quality}, 警告: {visual_state.warnings}")
```

### 6. 错误处理
```python
logger.error(f"⚠ 视觉处理失败 - 错误: {e}", exc_info=True)
logger.warning("⚠ 帧数据不足，无法处理")
```

### 7. 程序退出
```python
logger.info("正在清理资源...")
logger.info("AirSim连接已断开")
logger.info("显示窗口已关闭")
logger.info(f"程序正常退出，共处理帧数: {frame_counter}")
```

## 日志格式

```
时间 - 模块名称 - 级别 - 消息内容
```

**示例：**
```
2026-02-15 15:45:30 - DroneVision - INFO - 无人机视觉感知系统启动
```

**时间格式：** `YYYY-MM-DD HH:MM:SS`  
**模块名称：** `DroneVision`  
**级别：** `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`  
**消息内容：** 实际日志信息

## 如何查看日志

### 方法1：查看实时日志
```bash
# Windows PowerShell
Get-Content logs/drone_vision.log -Wait -Tail 20

# macOS/Linux
tail -f logs/drone_vision.log
```

### 方法2：查看日志文件
```bash
# Windows PowerShell
Get-Content logs/drone_vision.log

# macOS/Linux
cat logs/drone_vision.log
```

### 方法3：使用文本编辑器
- 使用 VSCode、Notepad++、Sublime Text 等打开 `logs/drone_vision.log`
- 搜索关键信息：`质量`、`特征点`、`障碍物`

## 调整日志级别

### 临时调整（当前会话）
```python
from logging_utils import get_logger

logger = get_logger(log_dir='logs', log_file='drone_vision.log', log_level='DEBUG', log_to_file=True)
```

### 永久调整（修改配置）
在 `config_manager.py` 中修改：
```python
'logging': {
    'level': 'DEBUG',  # 改为 DEBUG, INFO, WARNING, ERROR, CRITICAL
    'log_to_file': True,
    'log_dir': 'logs',
    'log_file': 'drone_vision.log'
}
```

## 日志示例

### 完整的日志记录示例

```
2026-02-15 15:45:30 - DroneVision - INFO - ============================================================
2026-02-15 15:45:30 - DroneVision - INFO - 无人机视觉感知系统启动
2026-02-15 15:45:30 - DroneVision - INFO - 启动时间: 2026-02-15 15:45:30.123456
2026-02-15 15:45:30 - DroneVision - INFO - 项目根目录: D:\bian_cheng\code\annual_project
2026-02-15 15:45:30 - DroneVision - INFO - Python版本: 3.10.11
2026-02-15 15:45:30 - DroneVision - INFO - 开始初始化...
2026-02-15 15:45:30 - DroneVision - INFO - AirSim连接状态: True
2026-02-15 15:45:30 - DroneVision - INFO - 帧缓冲区初始化完成
2026-02-15 15:45:30 - DroneVision - INFO - 神经网络和视觉处理模块初始化完成
2026-02-15 15:45:30 - DroneVision - INFO - 显示窗口已创建
2026-02-15 15:45:30 - DroneVision - INFO - 移动速度: 10.0 m/s
2026-02-15 15:45:30 - DroneVision - INFO - ============================================================
2026-02-15 15:45:30 - DroneVision - INFO - 系统初始化完成，等待键盘输入...
2026-02-15 15:45:30 - DroneVision - INFO - 可用按键:
2026-02-15 15:45:30 - DroneVision - INFO -   w - 前进
2026-02-15 15:45:30 - DroneVision - INFO -   s - 后退
2026-02-15 15:45:30 - DroneVision - INFO -   a - 左移
2026-02-15 15:45:30 - DroneVision - INFO -   d - 右移
2026-02-15 15:45:30 - DroneVision - INFO -   , - 上升
2026-02-15 15:45:30 - DroneVision - INFO -   . - 下降
2026-02-15 15:45:30 - DroneVision - INFO -   e - 捕获并处理两帧（间隔100ms）
2026-02-15 15:45:30 - DroneVision - INFO -   q - 退出
2026-02-15 15:45:30 - DroneVision - INFO - ============================================================
2026-02-15 15:45:31 - DroneVision - INFO - 按键检测: ASCII=119, 字符=w
2026-02-15 15:45:31 - DroneVision - INFO - 移动指令: 前进
2026-02-15 15:45:31 - DroneVision - INFO - 开始捕获帧对...
2026-02-15 15:45:31 - DroneVision - DEBUG - 开始捕获第1帧，时间戳: 1707999931000
2026-02-15 15:45:31 - DroneVision - INFO - [第1帧] 捕获成功 - 分辨率: (480, 640, 3), 时间戳: 1707999931000
2026-02-15 15:45:31 - DroneVision - DEBUG - 等待100ms后捕获第2帧...
2026-02-15 15:45:31 - DroneVision - INFO - [第2帧] 捕获成功 - 分辨率: (480, 640, 3), 时间戳: 1707999932000
2026-02-15 15:45:31 - DroneVision - INFO - 开始处理帧对 0 (NeuralPerception + VisualPerception pipeline)...
2026-02-15 15:45:32 - DroneVision - INFO - ✓ 神经网络处理完成 - 质量: 0.952, 特征点: 156, 分割区域: 3, 时间: 1.234s
2026-02-15 15:45:32 - DroneVision - INFO - ✓ 机器人运动 - 位移: dx=0.50, dy=0.00, dz=0.00, 置信度: 0.876
2026-02-15 15:45:32 - DroneVision - INFO - ✓ 障碍物检测 - 距离: 3.5m, 角度: 45.0°
2026-02-15 15:45:32 - DroneVision - INFO - ✓ 质量指标 - 评分: High, 警告: []
2026-02-15 15:45:32 - DroneVision - INFO - ✓ 处理完成 - 输出已序列化
2026-02-15 15:45:32 - DroneVision - INFO - 正在清理资源...
2026-02-15 15:45:32 - DroneVision - INFO - AirSim连接已断开
2026-02-15 15:45:32 - DroneVision - INFO - 显示窗口已关闭
2026-02-15 15:45:32 - DroneVision - INFO - ============================================================
2026-02-15 15:45:32 - DroneVision - INFO - 程序正常退出，共处理帧数: 1
2026-02-15 15:45:32 - DroneVision - INFO - ============================================================
```

## 常见问题

### Q1: 日志文件在哪里？
**A:** 在 `logs/drone_vision.log`

### Q2: 如何查看实时日志？
**A:** 使用 `tail -f logs/drone_vision.log` 或 `Get-Content logs/drone_vision.log -Wait -Tail 20`

### Q3: 如何查看所有错误？
**A:** 使用文本编辑器搜索 `ERROR` 或查看 ERROR 级别的日志

### Q4: 如何调整日志级别？
**A:** 修改 `config_manager.py` 中的 `'logging': {'level': 'DEBUG'}`

### Q5: 日志文件会越来越大吗？
**A:** 是的，建议定期清理或设置日志轮转

## 日志轮转建议

当前版本日志会一直追加到文件中。长期运行时，建议：

### 方案1：定期清理
```bash
# Windows
del logs\drone_vision.log

# macOS/Linux
rm logs/drone_vision.log
```

### 方案2：添加日志轮转（需要修改 logging_utils.py）
```python
from logging.handlers import RotatingFileHandler

# 限制日志文件大小（10MB）和保留文件数（3个）
file_handler = RotatingFileHandler(
    log_path,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=3
)
```

## 注意事项

1. **日志会同时输出到终端和文件**
2. **日志级别默认为 INFO**，调试信息不会显示
3. **ERROR 级别会记录完整的异常堆栈**
4. **日志文件可能包含敏感信息**，注意不要泄露
5. **定期清理旧日志**，避免占用过多磁盘空间

---

**更新时间：** 2026-02-15  
**版本：** v1.0