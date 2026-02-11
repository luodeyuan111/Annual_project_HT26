#!/usr/bin/env python3
"""
基于视觉感知的无人机集群系统 - 完整修正版
修复路径问题和模块导入
"""

import os
import sys
import argparse
import yaml
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import importlib.util
import traceback

# ========== 路径配置 ==========
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))  # src目录

# 项目根目录：src的上级目录
project_root = os.path.abspath(os.path.join(current_dir, ".."))

print("=" * 60)
print("无人机视觉感知系统 - 启动")
print(f"脚本目录: {current_dir}")
print(f"项目根目录: {project_root}")
print("=" * 60)

# 添加基础路径到系统路径
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(project_root, "src", "models", "modules"))
sys.path.insert(0, os.path.join(project_root, "src", "utils"))

# ========== 设置RAFT路径 ==========
raft_dir = os.path.join(project_root, "core_algorithms", "RAFT")
raft_core_dir = os.path.join(raft_dir, "core")
raft_utils_dir = os.path.join(raft_core_dir, "utils")

print("\n设置RAFT路径:")
print(f"  RAFT目录: {raft_dir}")
print(f"  RAFT核心目录: {raft_core_dir}")
print(f"  RAFT utils目录: {raft_utils_dir}")

if os.path.exists(raft_dir):
    if raft_dir not in sys.path:
        sys.path.insert(0, raft_dir)
        print(f"✓ 添加RAFT目录")

    if os.path.exists(raft_core_dir) and raft_core_dir not in sys.path:
        sys.path.insert(0, raft_core_dir)
        print(f"✓ 添加RAFT核心目录")

    # 不要将 raft_core_dir/"utils" 直接加入 sys.path ——
    # 这会把 utils/utils.py 作为顶级模块加载，导致 import utils.utils 失败。
    if os.path.exists(raft_utils_dir):
        print(f"  (info) RAFT utils 目录存在: {raft_utils_dir}")
else:
    print(f"⚠ RAFT目录不存在")

# ========== 设置Monodepth2路径 ==========
monodepth2_dir = os.path.join(project_root, "core_algorithms", "monodepth2")
monodepth2_networks_dir = os.path.join(monodepth2_dir, "networks")

print("\n设置Monodepth2路径:")
print(f"  Monodepth2目录: {monodepth2_dir}")
print(f"  Monodepth2网络目录: {monodepth2_networks_dir}")

if os.path.exists(monodepth2_dir):
    if monodepth2_dir not in sys.path:
        sys.path.insert(0, monodepth2_dir)
        print(f"✓ 添加Monodepth2目录")
    
    if os.path.exists(monodepth2_networks_dir) and monodepth2_networks_dir not in sys.path:
        sys.path.insert(0, monodepth2_networks_dir)
        print(f"✓ 添加Monodepth2网络目录")
    
    # 检查关键文件
    required_files = ['layers.py', 'utils.py']
    for file in required_files:
        file_path = os.path.join(monodepth2_dir, file)
        if os.path.exists(file_path):
            print(f"✓ {file} 存在")
        else:
            print(f"⚠ {file} 缺失")
else:
    print(f"⚠ Monodepth2目录不存在")

print("=" * 60)

# ========== 导入模块 ==========
print("\n导入核心模块...")

try:
    # 尝试从 models.modules 导入
    from models.modules.depth_estimator import Monodepth2Estimator
    from models.modules.clustering import TraditionalSegmenter
    from models.modules.geometry_utils import GeometryProcessor
    # visualization 模块位于 src/utils，可直接作为顶级模块导入
    from visualization import Visualizer
    print("✅ 核心模块导入成功")
    
except ImportError as e:
    print(f"❌ 核心模块导入失败: {e}")
    print("\n尝试动态导入模块...")
    
    try:
        # 动态导入各个模块
        modules_dir = os.path.join(project_root, "src", "models", "modules")
        utils_dir = os.path.join(project_root, "src", "utils")
        
        # 导入 depth_estimator
        depth_estimator_path = os.path.join(modules_dir, "depth_estimator.py")
        if os.path.exists(depth_estimator_path):
            spec = importlib.util.spec_from_file_location("depth_estimator", depth_estimator_path)
            depth_estimator_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(depth_estimator_module)
            Monodepth2Estimator = depth_estimator_module.Monodepth2Estimator
            print("✓ 动态导入 depth_estimator")
        else:
            print("⚠ depth_estimator.py 不存在")
            raise ImportError("depth_estimator.py 不存在")
        
        # 导入 clustering
        clustering_path = os.path.join(modules_dir, "clustering.py")
        if os.path.exists(clustering_path):
            spec = importlib.util.spec_from_file_location("clustering", clustering_path)
            clustering_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(clustering_module)
            TraditionalSegmenter = clustering_module.TraditionalSegmenter
            print("✓ 动态导入 clustering")
        
        # 导入 geometry_utils
        geometry_path = os.path.join(modules_dir, "geometry_utils.py")
        if os.path.exists(geometry_path):
            spec = importlib.util.spec_from_file_location("geometry_utils", geometry_path)
            geometry_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(geometry_module)
            GeometryProcessor = geometry_module.GeometryProcessor
            print("✓ 动态导入 geometry_utils")
        
        # 导入 visualization
        visualization_path = os.path.join(utils_dir, "visualization.py")
        if os.path.exists(visualization_path):
            spec = importlib.util.spec_from_file_location("visualization", visualization_path)
            visualization_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(visualization_module)
            Visualizer = visualization_module.Visualizer
            print("✓ 动态导入 visualization")
        
        print("✅ 动态导入模块成功")
        
    except Exception as e:
        print(f"❌ 动态导入也失败: {e}")
        print("使用模拟模块")
        # 创建模拟模块类
        class MockMonodepth2Estimator:
            def __init__(self, **kwargs):
                print("创建模拟深度估计器")
            def estimate_depth(self, image, input_format='RGB'):
                h, w = image.shape[:2]
                return np.random.rand(h, w) * 10 + 1.0
        
        class MockTraditionalSegmenter:
            def __init__(self, **kwargs):
                print("创建模拟分割器")
            def segment(self, points, flow_vectors):
                n_points = len(points)
                labels = np.random.randint(0, 3, n_points)
                segment_info = {f'region_{i}': {'label': i} for i in range(3)}
                return labels, segment_info
        
        class MockGeometryProcessor:
            def __init__(self, camera_matrix):
                print("创建模拟几何处理器")
        
        class MockVisualizer:
            def __init__(self):
                print("创建模拟可视化器")
        
        Monodepth2Estimator = MockMonodepth2Estimator
        TraditionalSegmenter = MockTraditionalSegmenter
        GeometryProcessor = MockGeometryProcessor
        Visualizer = MockVisualizer

print("=" * 60)

# ========== RAFT处理器 ==========
class RAFTProcessor:
    """RAFT光流处理器 - 修正版"""
    def __init__(self, model_path, device='cuda'):
        print(f"初始化RAFT处理器，模型: {model_path}")
        
        self.device = device
        self.is_virtual = True  # 默认使用虚拟模式
        
        # 检查RAFT目录
        if not os.path.exists(raft_core_dir):
            print(f"⚠ RAFT核心目录不存在: {raft_core_dir}")
            print("使用虚拟光流处理器")
            return
        
        # 检查RAFT核心文件
        required_raft_files = ['raft.py', 'update.py', 'extractor.py', 'corr.py']
        for file in required_raft_files:
            file_path = os.path.join(raft_core_dir, file)
            if not os.path.exists(file_path):
                print(f"⚠ RAFT文件缺失: {file}")
                print("使用虚拟光流处理器")
                return
        
        # 尝试导入RAFT（以包形式导入，避免相对导入问题）
        try:
            import torch

            try:
                from RAFT.core.raft import RAFT

                print("加载RAFT模型...")
                # 为RAFT构造一个简单的 args 对象
                class ArgsObj:
                    def __init__(self):
                        self.small = False
                        self.mixed_precision = False
                        self.alternate_corr = False

                args = ArgsObj()
                self.model = RAFT(args)

                # 如果有预训练权重，加载
                if model_path and os.path.exists(model_path):
                    print(f"加载RAFT权重: {model_path}")
                    checkpoint = torch.load(model_path, map_location=device)
                    # 支持多种 checkpoint 结构
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint

                    def try_load_model(sd, desc, strict=True):
                        try:
                            self.model.load_state_dict(sd, strict=strict)
                            print(f"RAFT: 成功加载权重 ({desc}, strict={strict})")
                            return True
                        except Exception as e:
                            print(f"RAFT: 加载失败 ({desc}, strict={strict}): {e}")
                            return False

                    first_key = next(iter(state_dict.keys())) if isinstance(state_dict, dict) else ''
                    loaded = False

                    # 直接尝试
                    if try_load_model(state_dict, 'original', strict=True):
                        loaded = True

                    # 去掉 module. 前缀后重试
                    if not loaded and first_key.startswith('module.'):
                        stripped = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
                        if try_load_model(stripped, 'stripped_module_prefix', strict=True):
                            loaded = True

                    # 为 key 添加 module. 前缀后重试
                    if not loaded and not first_key.startswith('module.'):
                        prefixed = {'module.' + k: v for k, v in state_dict.items()}
                        if try_load_model(prefixed, 'added_module_prefix', strict=True):
                            loaded = True

                    # 最后尝试 strict=False
                    if not loaded:
                        if try_load_model(state_dict, 'original', strict=False):
                            loaded = True
                        else:
                            if first_key.startswith('module.'):
                                stripped = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
                                if try_load_model(stripped, 'stripped_module_prefix', strict=False):
                                    loaded = True
                            else:
                                prefixed = {'module.' + k: v for k, v in state_dict.items()}
                                if try_load_model(prefixed, 'added_module_prefix', strict=False):
                                    loaded = True

                    if not loaded:
                        raise RuntimeError('无法加载 RAFT 权重：尝试了多种前缀/strict 策略均失败')

                self.model.to(device)
                self.model.eval()
                self.is_virtual = False
                print("✅ RAFT模型加载成功")

            except Exception as e:
                print(f"⚠ 导入RAFT模块失败: {e}")
                print("使用虚拟光流处理器")

        except ImportError as e:
            print(f"⚠ 导入torch失败: {e}")
            print("使用虚拟光流处理器")
    
    def compute_flow(self, img1, img2, iters=20):
        """计算光流"""
        import cv2
        
        # 转换图像格式
        if isinstance(img1, Image.Image):
            img1_np = np.array(img1)
        else:
            img1_np = img1
            
        if isinstance(img2, Image.Image):
            img2_np = np.array(img2)
        else:
            img2_np = img2
        
        h, w = img1_np.shape[:2]
        
        # 如果是虚拟模式，返回虚拟光流
        if self.is_virtual or not hasattr(self, 'model'):
            print("[虚拟模式] 计算光流")
            
            # 生成虚拟光流场
            flow = np.zeros((h, w, 2), dtype=np.float32)
            
            # 创建坐标网格
            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            center_y, center_x = h // 2, w // 2
            
            # 简单的径向流场
            flow_x = (x - center_x) / max(center_x, 1) * 2.0
            flow_y = (y - center_y) / max(center_y, 1) * 2.0
            
            # 添加随机噪声
            flow[:, :, 0] = flow_x + np.random.randn(h, w) * 0.3
            flow[:, :, 1] = flow_y + np.random.randn(h, w) * 0.3
            
            return flow
        
        # 真实RAFT模式
        try:
            import torch
            from torchvision.transforms import ToTensor
            
            # 确保图像格式正确
            if len(img1_np.shape) == 3:
                # 如果是BGR，转换为RGB
                if img1_np.shape[2] == 3:
                    img1_np = cv2.cvtColor(img1_np, cv2.COLOR_BGR2RGB)
                    img2_np = cv2.cvtColor(img2_np, cv2.COLOR_BGR2RGB)
            
            # 转换为tensor
            transform = ToTensor()
            img1_tensor = transform(img1_np).unsqueeze(0).to(self.device)
            img2_tensor = transform(img2_np).unsqueeze(0).to(self.device)
            
            # 计算光流
            with torch.no_grad():
                flow_tensor = self.model(img1_tensor, img2_tensor, iters=iters)[-1]
                flow_np = flow_tensor[0].cpu().numpy().transpose(1, 2, 0)
            
            print(f"✅ RAFT光流计算完成，尺寸: {flow_np.shape}")
            return flow_np
            
        except Exception as e:
            print(f"⚠ RAFT计算失败，使用虚拟光流: {e}")
            # 回退到虚拟光流
            return self.compute_flow(img1, img2)
    
    def visualize_flow(self, flow):
        """可视化光流"""
        import cv2
        
        # 转换为HSV图像
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        
        # 计算幅度和角度
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 归一化
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        # 转换为BGR
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return bgr

# ========== UAVVisionSystem类 ==========
class UAVVisionSystem:
    """无人机视觉感知系统"""
    
    def __init__(self, config_path=None, config_updates=None):
        print("\n初始化无人机视觉感知系统...")
        
        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"加载配置文件: {config_path}")
        else:
            self.config = self._get_default_config()
            print("使用默认配置")
        
        # 应用配置更新
        if config_updates:
            for section, updates in config_updates.items():
                if section in self.config:
                    self.config[section].update(updates)
                    print(f"应用配置更新: {section} -> {updates}")
        
        # 创建输出目录
        self.output_dir = self._create_output_dir()
        
        # 初始化组件
        self._init_components()
        
        print("✅ 系统初始化完成")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'system': {
                'name': 'UAV_Vision_System',
                'version': '1.0.0'
            },
            'optical_flow': {
                'model_path': 'models/raft/raft-things.pth',
                'device': 'cuda',
                'iters': 20
            },
            'depth_estimation': {
                'model_path': 'models/monodepth2/mono+stereo_640x192',
                'input_width': 640,
                'input_height': 192,
                'min_depth': 0.1,
                'max_depth': 100.0,
                'scale_factor': 5.4,
                'device': 'cuda'
            },
            'clustering': {
                'method': 'kmeans',
                'n_clusters': 3,
                'grid_step': 8
            },
            'camera': {
                'intrinsic': {
                    'fx': 600.0,
                    'fy': 600.0,
                    'cx': 320.0,
                    'cy': 240.0
                },
                'resolution': [640, 480]
            },
            'output': {
                'save_results': True,
                'visualize': True,
                'save_format': 'npy'
            }
        }
    
    def _create_output_dir(self):
        """创建输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(project_root, "outputs", f"demo_run_{timestamp}")
        
        # 确保上级目录存在
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        
        # 创建目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建子目录
        subdirs = ['光流', '深度', '分割', '位姿', '可视化', 'logs']
        for subdir in subdirs:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
        
        # 保存配置
        config_save_path = os.path.join(output_dir, 'config.yaml')
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        return output_dir
    
    def _init_components(self):
        """初始化各个组件"""
        print("初始化组件...")
        
        # 1. 光流处理器
        print("  初始化RAFT光流处理器...")
        raft_config = self.config['optical_flow']
        self.flow_processor = RAFTProcessor(
            model_path=raft_config['model_path'],
            device=raft_config.get('device', 'cpu')
        )
        
        # 2. 深度估计器
        print("  初始化Monodepth2深度估计器...")
        depth_config = self.config['depth_estimation']
        try:
            self.depth_estimator = Monodepth2Estimator(
                model_path=depth_config['model_path'],
                input_width=depth_config['input_width'],
                input_height=depth_config['input_height'],
                min_depth=depth_config['min_depth'],
                max_depth=depth_config['max_depth'],
                scale_factor=depth_config['scale_factor'],
                device=depth_config.get('device', 'cpu')
            )
            print("  ✅ 深度估计器初始化成功")
        except Exception as e:
            print(f"  ⚠ 深度估计器初始化失败: {e}")
            print("  使用模拟深度估计器")
            self.depth_estimator = self._create_mock_depth_estimator()
        
        # 3. 聚类分割器
        print("  初始化聚类分割器...")
        cluster_config = self.config['clustering']
        try:
            self.segmenter = TraditionalSegmenter(
                method=cluster_config['method'],
                n_clusters=cluster_config['n_clusters']
            )
            print("  ✅ 分割器初始化成功")
        except Exception as e:
            print(f"  ⚠ 分割器初始化失败: {e}")
            print("  使用模拟分割器")
            self.segmenter = self._create_mock_segmenter()
        
        # 4. 几何处理器
        print("  初始化几何处理器...")
        cam_config = self.config['camera']
        try:
            camera_matrix = np.array([
                [cam_config['intrinsic']['fx'], 0, cam_config['intrinsic']['cx']],
                [0, cam_config['intrinsic']['fy'], cam_config['intrinsic']['cy']],
                [0, 0, 1]
            ])
            self.geometry_processor = GeometryProcessor(camera_matrix)
            print("  ✅ 几何处理器初始化成功")
        except Exception as e:
            print(f"  ⚠ 几何处理器初始化失败: {e}")
            print("  使用模拟几何处理器")
            self.geometry_processor = self._create_mock_geometry_processor(camera_matrix)
        
        # 5. 可视化器
        print("  初始化可视化器...")
        try:
            self.visualizer = Visualizer()
            print("  ✅ 可视化器初始化成功")
        except Exception as e:
            print(f"  ⚠ 可视化器初始化失败: {e}")
            print("  使用模拟可视化器")
            self.visualizer = self._create_mock_visualizer()
        
        self.grid_step = cluster_config.get('grid_step', 8)
        
        print("所有组件初始化完成")
    
    def _create_mock_depth_estimator(self):
        """创建模拟深度估计器"""
        class MockDepthEstimator:
            def __init__(self, **kwargs):
                print("创建模拟深度估计器")
            
            def estimate_depth(self, image, input_format='RGB'):
                """估计深度"""
                if isinstance(image, Image.Image):
                    image_np = np.array(image)
                else:
                    image_np = image
                
                h, w = image_np.shape[:2]
                depth = np.random.rand(h, w) * 10 + 1.0  # 1-11米的随机深度
                return depth
            
            def visualize_depth(self, depth):
                """可视化深度图"""
                depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
                depth_colored = cv2.applyColorMap(
                    (depth_normalized * 255).astype(np.uint8), 
                    cv2.COLORMAP_JET
                )
                return depth_colored
        
        return MockDepthEstimator()
    
    def _create_mock_segmenter(self):
        """创建模拟分割器"""
        class MockSegmenter:
            def __init__(self, method='kmeans', n_clusters=3):
                print(f"创建模拟分割器: {method}, {n_clusters} clusters")
            
            def segment(self, points, flow_vectors):
                """分割"""
                n_points = len(points)
                labels = np.random.randint(0, 3, n_points)
                segment_info = {f'region_{i}': {'label': i, 'n_points': np.sum(labels == i)} 
                               for i in range(3)}
                return labels, segment_info
            
            def visualize_segmentation(self, image, points, labels):
                """可视化分割"""
                if isinstance(image, np.ndarray):
                    img_copy = image.copy()
                else:
                    img_copy = np.array(image).copy()
                
                # 为每个标签分配颜色
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
                for point, label in zip(points, labels):
                    if label < len(colors):
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(img_copy, (x, y), 3, colors[label], -1)
                
                return img_copy
        
        return MockSegmenter()
    
    def _create_mock_geometry_processor(self, camera_matrix):
        """创建模拟几何处理器"""
        class MockGeometryProcessor:
            def __init__(self, camera_matrix):
                print("创建模拟几何处理器")
                self.camera_matrix = camera_matrix
            
            def project_2d_to_3d(self, points, depth):
                """2D到3D投影"""
                n_points = len(points)
                return np.random.rand(n_points, 3) * 5
            
            def estimate_pose_ransac(self, points_3d_t, points_3d_t1, **kwargs):
                """估计位姿"""
                n_points = len(points_3d_t)
                R = np.eye(3)
                t = np.random.randn(3) * 0.1
                inlier_mask = np.random.rand(n_points) > 0.3
                return R, t, inlier_mask
            
            def rotation_matrix_to_euler(self, R):
                """旋转矩阵转欧拉角"""
                return np.array([0.0, 0.0, 0.0])
        
        return MockGeometryProcessor(camera_matrix)
    
    def _create_mock_visualizer(self):
        """创建模拟可视化器"""
        class MockVisualizer:
            def __init__(self):
                print("创建模拟可视化器")
        
        return MockVisualizer()
    
    def _extract_feature_points(self, flow, grid_step=8):
        """
        从光流中提取特征点对
        
        Args:
            flow: 光流场 [H, W, 2]
            grid_step: 网格采样步长
            
        Returns:
            points_t: t时刻点坐标 [N, 2]
            points_t1: t+1时刻点坐标 [N, 2]
            flow_vectors: 光流向量 [N, 2]
        """
        H, W = flow.shape[:2]
        
        # 创建网格
        y_indices = np.arange(0, H, grid_step)
        x_indices = np.arange(0, W, grid_step)
        grid_y, grid_x = np.meshgrid(y_indices, x_indices, indexing='ij')
        
        # 展平
        points_t = np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1).astype(np.float32)
        
        # 提取光流向量
        flow_vectors = flow[grid_y.flatten(), grid_x.flatten()]
        
        # 计算t+1时刻位置
        points_t1 = points_t + flow_vectors
        
        return points_t, points_t1, flow_vectors
    
    def process_frame_pair(self, image1, image2, frame_idx):
        """
        处理一对图像帧
        
        Args:
            image1: t时刻图像
            image2: t+1时刻图像
            frame_idx: 帧索引
            
        Returns:
            result: 处理结果字典
        """
        print(f"\n处理帧对 {frame_idx}")
        print("-" * 40)
        
        result = {
            'frame_idx': frame_idx,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 1. 计算光流
        print("  1. 计算光流...")
        try:
            flow = self.flow_processor.compute_flow(image1, image2, iters=20)
            result['flow'] = flow
            
            # 保存光流
            if self.config['output']['save_results']:
                flow_save_path = os.path.join(self.output_dir, '光流', f'flow_{frame_idx:04d}')
                np.save(f"{flow_save_path}.npy", flow)
                
                # 可视化
                flow_img = self.flow_processor.visualize_flow(flow)
                cv2.imwrite(f"{flow_save_path}.png", cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR))
            
            print(f"    ✓ 光流计算完成: {flow.shape}")
        except Exception as e:
            print(f"    ⚠ 光流计算失败: {e}")
            result['flow_error'] = str(e)
            flow = np.zeros((480, 640, 2), dtype=np.float32)  # 默认尺寸
        
        # 2. 提取特征点
        print("  2. 提取特征点...")
        try:
            points_t, points_t1, flow_vectors = self._extract_feature_points(flow, self.grid_step)
            result['feature_points'] = {
                'points_t': points_t,
                'points_t1': points_t1,
                'flow_vectors': flow_vectors,
                'n_points': len(points_t)
            }
            
            print(f"    ✓ 提取了 {len(points_t)} 个特征点")
        except Exception as e:
            print(f"    ⚠ 特征点提取失败: {e}")
        
        # 3. 估计深度
        print("  3. 估计深度图...")
        try:
            depth1 = self.depth_estimator.estimate_depth(image1, input_format='RGB')
            depth2 = self.depth_estimator.estimate_depth(image2, input_format='RGB')
            result['depth_maps'] = {
                'depth_t': depth1,
                'depth_t1': depth2
            }
            
            # 保存深度图
            if self.config['output']['save_results']:
                depth_save_path = os.path.join(self.output_dir, '深度', f'depth_{frame_idx:04d}')
                np.save(f"{depth_save_path}_t.npy", depth1)
                np.save(f"{depth_save_path}_t1.npy", depth2)
                
                # 可视化
                if hasattr(self.depth_estimator, 'visualize_depth'):
                    depth_img1 = self.depth_estimator.visualize_depth(depth1)
                    depth_img2 = self.depth_estimator.visualize_depth(depth2)
                    cv2.imwrite(f"{depth_save_path}_t.png", depth_img1)
                    cv2.imwrite(f"{depth_save_path}_t1.png", depth_img2)
            
            print(f"    ✓ 深度估计完成: {depth1.shape}")
        except Exception as e:
            print(f"    ⚠ 深度估计失败: {e}")
        
        # 4. 聚类分割
        print("  4. 聚类分割...")
        try:
            if 'feature_points' in result:
                points_t = result['feature_points']['points_t']
                flow_vectors = result['feature_points']['flow_vectors']
                
                labels, segment_info = self.segmenter.segment(points_t, flow_vectors)
                result['segmentation'] = {
                    'labels': labels,
                    'segment_info': segment_info
                }
                
                print(f"    ✓ 分割为 {len(segment_info)} 个区域")
        except Exception as e:
            print(f"    ⚠ 分割失败: {e}")
        
        # 5. 位姿估计
        print("  5. 位姿估计...")
        pose_results = {}
        
        if 'segmentation' in result and 'depth_maps' in result:
            try:
                points_t = result['feature_points']['points_t']
                depth1 = result['depth_maps']['depth_t']
                depth2 = result['depth_maps']['depth_t1']
                labels = result['segmentation']['labels']
                segment_info = result['segmentation']['segment_info']
                
                # 3D投影
                points_3d_t = self.geometry_processor.project_2d_to_3d(points_t, depth1)
                points_3d_t1 = self.geometry_processor.project_2d_to_3d(points_t + flow_vectors, depth2)
                
                for seg_name, seg_data in segment_info.items():
                    mask = labels == seg_data['label']
                    
                    if np.sum(mask) < 10:  # 点数太少跳过
                        continue
                    
                    # 提取该区域的特征点
                    seg_points_3d_t = points_3d_t[mask]
                    seg_points_3d_t1 = points_3d_t1[mask]
                    
                    # 估计刚体变换
                    if hasattr(self.geometry_processor, 'estimate_pose_ransac'):
                        R, t, inlier_mask = self.geometry_processor.estimate_pose_ransac(
                            seg_points_3d_t, seg_points_3d_t1,
                            max_iterations=500, threshold=0.05
                        )
                        
                        if R is not None:
                            # 转换为欧拉角
                            if hasattr(self.geometry_processor, 'rotation_matrix_to_euler'):
                                euler_angles = self.geometry_processor.rotation_matrix_to_euler(R)
                            else:
                                euler_angles = [0.0, 0.0, 0.0]
                            
                            pose_results[seg_name] = {
                                'rotation': R.tolist(),
                                'translation': t.tolist(),
                                'euler_angles': euler_angles,
                                'n_inliers': int(np.sum(inlier_mask)),
                                'n_points': int(np.sum(mask))
                            }
                    
                result['pose_estimates'] = pose_results
                print(f"    ✓ 估计了 {len(pose_results)} 个位姿")
                
            except Exception as e:
                print(f"    ⚠ 位姿估计失败: {e}")
        
        # 保存结果
        if self.config['output']['save_results']:
            try:
                import json
                pose_save_path = os.path.join(self.output_dir, '位姿', f'pose_{frame_idx:04d}.json')
                
                # 将numpy数组转换为列表
                def convert_for_json(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_for_json(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_for_json(item) for item in obj]
                    else:
                        return obj
                
                json_result = convert_for_json(result)
                with open(pose_save_path, 'w', encoding='utf-8') as f:
                    json.dump(json_result, f, indent=2, ensure_ascii=False)
                
                print(f"    ✓ 结果保存到: {pose_save_path}")
            except Exception as e:
                print(f"    ⚠ 保存结果失败: {e}")
        
        print(f"帧对 {frame_idx} 处理完成")
        
        return result

# ========== 主函数 ==========
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='无人机视觉感知系统')
    parser.add_argument('--config', default='configs/default.yaml', 
                       help='配置文件路径')
    parser.add_argument('--input', default='../../data/test', 
                       help='输入图像目录')
    parser.add_argument('--start', type=int, default=0, 
                       help='起始帧索引')
    parser.add_argument('--end', type=int, default=None, 
                       help='结束帧索引')
    parser.add_argument('--output', help='输出目录（覆盖配置）')
    
    # 新增参数
    parser.add_argument('--path', help='输入图像目录（兼容别名）')
    parser.add_argument('--output_dir', help='输出目录（兼容别名）')
    parser.add_argument('--raft_model_path', help='RAFT模型路径')
    parser.add_argument('--monodepth2_model_path', help='Monodepth2模型路径')
    parser.add_argument('--visualize', action='store_true', 
                        help='是否可视化结果')
    parser.add_argument('--test', action='store_true', 
                        help='测试模式：处理单对图像')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("无人机视觉感知系统 - 启动")
    print("=" * 60)
    
    # 参数兼容性处理
    if args.path:
        args.input = args.path
    if args.output_dir:
        args.output = args.output_dir
    
    # 配置更新
    config_updates = {}
    if args.raft_model_path:
        config_updates['optical_flow'] = {'model_path': args.raft_model_path}
    if args.monodepth2_model_path:
        config_updates['depth_estimation'] = {'model_path': args.monodepth2_model_path}
    
    print(f"输入目录: {args.input}")
    if args.output:
        print(f"输出目录: {args.output}")
    
    # 创建系统实例
    print("\n初始化系统...")
    system = UAVVisionSystem(args.config, config_updates=config_updates)
    
    # 如果指定了可视化参数，更新配置
    if args.visualize:
        system.config['output']['visualize'] = True
    
    # 测试模式：处理单对图像
    if args.test:
        print("\n测试模式：处理单对图像")
        
        # 创建测试图像
        width, height = 640, 480
        img1 = np.zeros((height, width, 3), dtype=np.uint8)
        img2 = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 添加一些特征
        cv2.rectangle(img1, (50, 50), (200, 200), (255, 0, 0), 2)
        cv2.rectangle(img2, (60, 60), (210, 210), (255, 0, 0), 2)  # 轻微移动
        
        cv2.circle(img1, (400, 300), 50, (0, 255, 0), -1)
        cv2.circle(img2, (410, 310), 50, (0, 255, 0), -1)  # 轻微移动
        
        # 处理
        result = system.process_frame_pair(img1, img2, 0)
        
        if result:
            print("\n" + "=" * 60)
            print("测试处理完成")
            print(f"输出目录: {system.output_dir}")
            print("=" * 60)
    
    else:
        # 处理图像序列
        print(f"\n处理图像序列: {args.input}")
        
        # 检查输入目录
        if not os.path.exists(args.input):
            print(f"⚠ 输入目录不存在: {args.input}")
            
            # 尝试创建测试目录
            test_dir = os.path.join(project_root, "data", "test")
            os.makedirs(test_dir, exist_ok=True)
            
            # 创建测试图像
            for i in range(5):
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(img, f"Test Image {i}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imwrite(os.path.join(test_dir, f"test_{i:03d}.png"), img)
            
            args.input = test_dir
            print(f"使用测试目录: {test_dir}")
        
        # 获取图像文件
        import glob
        image_files = sorted(glob.glob(os.path.join(args.input, "*.png")) + 
                            glob.glob(os.path.join(args.input, "*.jpg")) +
                            glob.glob(os.path.join(args.input, "*.jpeg")))
        
        if not image_files:
            print(f"❌ 未找到图像文件: {args.input}")
            return
        
        print(f"找到 {len(image_files)} 张图像")
        
        # 处理每个连续帧对
        all_results = []
        for i in range(min(5, len(image_files) - 1)):  # 最多处理5对
            print(f"\n处理帧对 {i}:")
            print(f"  图像1: {os.path.basename(image_files[i])}")
            print(f"  图像2: {os.path.basename(image_files[i+1])}")
            
            try:
                # 加载图像
                img1 = Image.open(image_files[i]).convert('RGB')
                img2 = Image.open(image_files[i+1]).convert('RGB')
                
                # 处理帧对
                result = system.process_frame_pair(img1, img2, i)
                if result:
                    all_results.append(result)
                    
            except Exception as e:
                print(f"❌ 处理失败: {e}")
        
        print(f"\n处理完成: {len(all_results)}/{min(5, len(image_files)-1)} 帧对成功")
    
    print(f"\n所有结果保存到: {system.output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        traceback.print_exc()