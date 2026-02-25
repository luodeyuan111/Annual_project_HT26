# 版本说明 v1.2

**版本号**: v1.2  
**发布日期**: 2026-02-24  
**代号**: 运动控制与API方案

---

## 更新摘要

本次更新主要解决运动控制问题，并创建AirSim API方案作为集群控制的数据源。

---

## 详细更新内容

### 1. 运动控制改进（持续速度模式）

**问题**: 原方案"按下移动、松开停止"，但AirSim API需要持续发送速度指令才能保持运动。

**解决方案**:
- Toggle模式：第一次按键开始运动，再次按键停止
- 使用共享内存 `velocity_state` 追踪速度状态
- 主循环每帧检查速度变化并持续发送 `moveByVelocityBodyFrameAsync`

**修改文件**:
- `src/utils/keyboard_control.py`
- `src/main.py`

**效果**:
- ✅ 运动与拍摄可同时进行
- ✅ 两帧间有真实运动（光流/位姿估计有效）
- ✅ 游戏式即时响应

---

### 2. 多摄像头整合

**新增文件**: `src/Visual_process/multi_camera_fusion.py`

**功能**:
- `CameraConfig` 类：配置摄像头参数（名称、FOV、角度偏移）
- `MultiCameraFusion` 类：融合多摄像头障碍物信息
- 支持4摄像头方案（前/后/左/右）

```python
# 输入多个**接口**:
ObstaclePolarFrame
camera_frames = {
    "front": front_frame,
    "back": back_frame,
    "left": left_frame,
    "right": right_frame,
}

# 融合为360°覆盖
fusion = MultiCameraFusion.create_quad_camera()
obstacle_frame = fusion.fuse(camera_frames)
```

---

### 3. AirSim API方案

**新增文件夹**: `src/alternative_pipeline/`

**新增文件**:

#### api_data_extractor.py
- `AirSimDataExtractor`: 单摄像头数据获取
- `MultiCameraAirSimExtractor`: 多摄像头数据获取
- 获取：深度图、RGB图、位姿、速度

#### api_visual_adapter.py  
- `AirSimVisualAdapter`: 将API数据转换为VisualState
- `MultiCameraAirSimAdapter`: 多摄像头数据适配
- 复用现有的 `ObstacleDetector` 处理深度图

#### main_api.py
- API方案主程序入口
- 支持单/多摄像头模式
- 支持持续运行和单次运行

---

### 4. 双流程架构

```
┌─────────────────────────────────────────────────────────────┐
│                    双流程架构                                │
├──────────────────────────┬──────────────────────────────────┤
│    流程1: 集群控制        │    流程2: 视觉验证                │
│    (AirSim API)          │    (神经网络+OpenCV)              │
├──────────────────────────┼──────────────────────────────────┤
│  alternative_pipeline/   │  main.py                         │
│  • 真实深度              │  • MonoDepth2深度估计            │
│  • 真实位姿              │  • RAFT光流估计                  │
│  • 多摄像头支持          │  • 图像处理                      │
│  • 直接用于集群控制       │  • 效果展示/验证                 │
└──────────────────────────┴──────────────────────────────────┘
```

---

## 文件变更清单

### 修改的文件
1. `src/utils/keyboard_control.py` - 持续速度控制模式
2. `src/main.py` - 集成持续速度控制

### 新增的文件
1. `src/Visual_process/multi_camera_fusion.py` - 多摄像头融合
2. `src/alternative_pipeline/api_data_extractor.py` - API数据获取
3. `src/alternative_pipeline/api_visual_adapter.py` - API数据适配
4. `src/alternative_pipeline/main_api.py` - API方案主程序

---

## 使用方式

### 神经网络方案（原方案）
```bash
python -m src.main
# 或
cd src && python main.py
```

### AirSim API方案（新方案）
```bash
# 单摄像头模式
python -m src.alternative_pipeline.main_api

# 多摄像头模式
python -m src.alternative_pipeline.main_api --multi-camera

# 持续运行模式
python -m src.alternative_pipeline.main_api --continuous --interval 0.5
```

---

## 后续计划

- [ ] 多无人机支持（等待AirSim setting配置）
- [ ] 集群控制集成测试
- [ ] 准备中期答辩材料

---

# 版本 v1.2.1 更新（2026-02-25）

## 更新摘要

修复障碍物检测和可视化显示问题。

## 详细更新

### 1. 障碍物检测改进

**问题**: 原方案取全列最小值会检测到地面，文献中是取安全高度范围内的最小值

**解决方案**:
- 扫描行从 20%-80%（10行）改为 45%-55%（3行）
- 只检测无人机安全飞行高度范围内的障碍物

**修改文件**: `src/alternative_pipeline/api_visual_adapter.py`

### 2. 可视化优化

**改进**:
- 有效障碍物半径放大1.5倍，使显示更美观
- 移除角度偏移（ANGLE_OFFSET=0）
- 绿点阈值调整到95%最大距离

**修改文件**: `src/utils/visualization.py`

### 3. AirSim深度图处理

**改进**:
- 使用95百分位动态确定"远边界"
- 障碍物阈值设为远边界的80%

---

**维护者**: AI Assistant  
**审核日期**: 2026-02-25
