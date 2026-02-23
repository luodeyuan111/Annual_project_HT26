# 多进程架构使用指南

> **更新时间**：2026-02-22  
> **版本**：v1.0

---

## 概述

多进程架构解决了单进程架构中的**键盘输入阻塞**问题。

### 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    主进程                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ 捕获帧       │  │ 处理视觉     │  │ 显示图像     │     │
│  │ (1-2秒)      │  │ (Neural+Vis) │  │ (OpenCV)     │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                          ↓                              │
│                   [Queue通信]                           │
│                     (命令队列)                           │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              键盘控制进程                                │
│  ┌──────────────┐  ┌──────────────┐                     │
│  │ 监听键盘      │  │ 发送命令     │                      │
│  │ (msvcrt)     │  │ (Queue)      │                     │
│  │ (1ms检查)    │  │ (实时响应)    │                      │
│  └──────────────┘  └──────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

---

## 优势

### 1. 键盘输入不再阻塞
- ✅ 键盘进程独立运行，快速响应
- ✅ 处理视觉时，键盘检测无延迟
- ✅ 可以同时操作AirSim窗口和VSCode终端

### 2. 崩溃隔离
- ✅ 键盘进程崩溃，不影响视觉处理
- ✅ 视觉处理崩溃，不影响键盘控制
- ✅ 一个进程崩溃，另一个继续运行

### 3. 易于扩展
- ✅ 可以添加更多进程（如日志记录进程）
- ✅ 可以添加更多功能模块

---

## 文件说明

### 1. `src/keyboard_control.py` - 键盘控制进程

**功能**：
- 独立监听键盘输入
- 通过Queue发送按键命令给主进程
- 支持多键同时按下
- 守护进程（主进程退出时自动退出）

**关键代码**：
```python
def keyboard_control_process(command_queue, logger, should_stop):
    while not should_stop.value:
        if msvcrt.kbhit():
            # 读取所有按键
            keys = []
            while msvcrt.kbhit():
                keys.append(ord(msvcrt.getch()))
            
            # 发送命令给主进程
            for key in keys:
                command_queue.put(('move', {'dx': 10, 'dy': 0, 'dz': 0}))
```

**启动方式**：
```python
from keyboard_control import keyboard_control_process
from multiprocessing import Process, Queue, Value

# 创建队列
command_queue = Queue()
should_stop = Value('b', False)

# 启动键盘控制进程
keyboard_proc = Process(
    target=keyboard_control_process,
    args=(command_queue, logger, should_stop)
)
keyboard_proc.daemon = True
keyboard_proc.start()
```

### 2. `src/main.py` - 主程序（多进程版本）

**架构**：
- 启动键盘控制进程
- 启动视觉处理进程
- 通过Queue通信

**关键代码**：
```python
if __name__ == "__main__":
    # 创建共享变量和队列
    should_stop = Value('b', False)
    command_queue = Queue()
    
    # 启动键盘控制进程
    keyboard_proc = Process(
        target=keyboard_control_process,
        args=(command_queue, logger, should_stop)
    )
    keyboard_proc.daemon = True
    keyboard_proc.start()
    
    # 启动视觉处理进程
    visual_proc = Process(
        target=visual_processing_process,
        args=(command_queue, should_stop)
    )
    visual_proc.start()
    
    # 等待视觉处理进程结束
    visual_proc.join()
    
    # 等待键盘控制进程结束
    keyboard_proc.join()
```

---

## 使用方法

### 1. 测试键盘控制进程

```bash
cd d:\bian_cheng\code\annual_project
python src/keyboard_control.py
```

**按键说明**：
- `w` - 前进
- `s` - 后退
- `a` - 左移
- `d` - 右移
- `,` - 上升
- `.` - 下降
- `e` - 处理两帧
- `q` - 退出

### 2. 运行主程序

```bash
cd d:\bian_cheng\code\annual_project
python src/main.py
```

**先确保**：
- AirSim已启动
- 摄像头已配置

**使用说明**：
- `w/s/a/d` - 移动无人机
- `,/.` - 上升/下降
- `e` - 捕获并处理两帧（间隔100ms）
- `q` - 退出程序

---

## 进程间通信

### 通信方式：Queue（队列）

**生产者（键盘进程）**：
```python
# 放入队列
command_queue.put(('move', {'dx': 10, 'dy': 0, 'dz': 0}))
command_queue.put(('process', None))
command_queue.put(('quit', None))
```

**消费者（主进程）**：
```python
# 从队列读取
if not command_queue.empty():
    cmd_type, data = command_queue.get()
    
    if cmd_type == 'move':
        dx, dy, dz = data['dx'], data['dy'], data['dz']
        # 移动无人机
    elif cmd_type == 'process':
        # 处理两帧
    elif cmd_type == 'quit':
        # 退出程序
```

**特点**：
- FIFO（先进先出）
- 线程安全
- 支持多种数据类型

---

## 调试技巧

### 1. 查看不同进程的日志

- **键盘进程**：`logs/keyboard.log`
- **主进程**：`logs/main.log`
- **视觉处理进程**：`logs/visual.log`

### 2. 检查进程状态

```python
# 检查进程是否活着
print(f"键盘进程状态: {keyboard_proc.is_alive()}")
print(f"视觉处理进程状态: {visual_proc.is_alive()}")
```

### 3. 查看进程PID

```python
# 查看进程ID
print(f"键盘进程PID: {keyboard_proc.pid}")
print(f"视觉处理进程PID: {visual_proc.pid}")
```

### 4. 调试键盘进程

```python
# 在键盘进程的代码中添加调试日志
logger.debug(f"检测到按键: {key}, 字符: {chr(key)}")
logger.debug(f"放入队列: {('move', cmd)}")
```

---

## 常见问题

### Q1: 键盘进程不响应按键

**原因**：
- 键盘进程未启动
- Queue未正确初始化
- 应用了Windows控制台特性

**解决方案**：
```python
# 确保使用msvcrt.kbhit()，不要使用input()
if msvcrt.kbhit():
    key = ord(msvcrt.getch())
```

### Q2: 主进程卡住，无法响应键盘

**原因**：
- 视觉处理时间过长
- Queue阻塞

**解决方案**：
- 检查`logs/main.log`和`logs/keyboard.log`
- 确保视觉处理没有死循环
- 调整`time.sleep()`的值

### Q3: 进程无法正常退出

**原因**：
- 守护进程设置错误
- 共享变量未正确更新

**解决方案**：
```python
# 确保设置守护进程
keyboard_proc.daemon = True

# 确保正确更新共享变量
should_stop.value = True
```

### Q4: ImportError

**原因**：
- 路径配置错误
- 模块未正确导入

**解决方案**：
```python
# 添加项目根目录到路径
sys.path.insert(0, project_root)

# 导入时使用相对路径
from keyboard_control import keyboard_control_process
```

---

## 性能优化

### 1. 降低CPU占用

```python
# 在键盘控制进程中
time.sleep(0.001)  # 1ms检查一次，而不是0ms
```

### 2. 减少日志输出

```python
# 在main.py中
logger.setLevel(logging.WARNING)  # 只输出WARNING及以上
```

### 3. 批量处理命令

```python
# 不建议：每次按键都处理
if msvcrt.kbhit():
    command_queue.put(('move', cmd))

# 建议：批量处理
if msvcrt.kbhit():
    commands = []
    while msvcrt.kbhit():
        commands.append(ord(msvcrt.getch()))
    for key in commands:
        command_queue.put(('move', get_command(key)))
```

---

## 扩展功能

### 1. 添加日志记录进程

```python
def logging_process(queue):
    while True:
        if not queue.empty():
            log_entry = queue.get()
            logger.info(log_entry)
        time.sleep(0.1)

# 启动日志进程
logging_proc = Process(target=logging_process, args=(log_queue,))
logging_proc.start()
```

### 2. 添加性能监控进程

```python
def performance_monitor_process():
    while not should_stop.value:
        fps = calculate_fps()
        logger.info(f"FPS: {fps}")
        time.sleep(1.0)

# 启动性能监控进程
perf_proc = Process(target=performance_monitor_process)
perf_proc.start()
```

### 3. 添加GUI控制面板

```python
import tkinter as tk

def gui_process():
    root = tk.Tk()
    root.title("无人机控制")
    
    # 添加按钮
    btn_forward = tk.Button(root, text="前进", command=lambda: send_command('forward'))
    btn_forward.pack()
    
    root.mainloop()

# 启动GUI进程
gui_proc = Process(target=gui_process)
gui_proc.start()
```

---

## 最佳实践

### 1. 进程设计原则
- ✅ 每个进程做一件事
- ✅ 使用Queue进行通信
- ✅ 设置守护进程
- ✅ 添加详细的错误处理

### 2. 资源管理
- ✅ 及时关闭资源
- ✅ 清理日志文件
- ✅ 释放内存
- ✅ 避免内存泄漏

### 3. 错误处理
- ✅ 每个进程有自己的try-except
- ✅ 记录详细的错误日志
- ✅ 设置合理的超时
- ✅ 提供回退机制

---

## 故障排查清单

- [ ] AirSim已启动
- [ ] 摄像头已配置
- [ ] 所有依赖已安装
- [ ] Python版本正确（3.7+）
- [ ] 日志目录已创建
- [ ] 进程能正常启动
- [ ] 键盘进程能响应按键
- [ ] 主进程能正常显示图像
- [ ] 进程能正常退出

---

## 下一步计划

1. **性能优化**
   - [ ] 添加性能监控
   - [ ] 优化进程间通信
   - [ ] 减少内存占用

2. **功能扩展**
   - [ ] 添加更多传感器数据
   - [ ] 实现GUI控制面板
   - [ ] 添加自动飞行模式

3. **测试完善**
   - [ ] 添加单元测试
   - [ ] 添加集成测试
   - [ ] 压力测试

---

**版本历史**：
- v1.0 (2026-02-22): 初始版本，多进程架构实现

**维护者**：[待填写]