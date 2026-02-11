#!/usr/bin/env python3
"""
无人机视觉整合系统 - 完整版（RAFT路径修正版）
根据实际项目结构修正
"""

import os
import sys
import time
import numpy as np
from PIL import Image
from datetime import datetime
import importlib.util
import traceback
import json

# ========== 路径设置 ==========
def get_project_root():
    """获取项目根目录"""
    # 方法1：从当前文件位置推算
    current_file = os.path.abspath(__file__)
    
    # 如果文件在 integration_system 目录中
    if 'integration_system' in current_file:
        # 找到 integration_system 的位置，然后取上级目录
        parts = current_file.split(os.sep)
        integration_index = parts.index('integration_system')
        return os.sep.join(parts[:integration_index])
    
    # 方法2：尝试常见项目结构
    possible_roots = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
        os.getcwd(),  # 当前工作目录
    ]
    
    for root in possible_roots:
        # 检查是否包含关键目录
        if all(os.path.exists(os.path.join(root, dir_name)) 
               for dir_name in ['src', 'Drone_Interface']):
            return root
    
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class DroneVisionIntegrator:
    """无人机视觉处理整合器"""
    
    def __init__(self, drone_config_path=None, vision_config_path=None):
        print("初始化无人机视觉整合系统...")
        
        # 状态变量
        self.frame_count = 0
        self.processing_count = 0
        self.is_processing = False
        
        # 获取项目根目录
        self.project_root = get_project_root()
        print(f"项目根目录: {self.project_root}")
        
        # 检查项目结构
        self._check_project_structure()
        
        # 输出目录
        self.output_dir = self._create_output_dir()
        
        # 初始化现有模块
        self.drone_controller = self._init_drone_interface()
        self.vision_system = self._init_vision_modules()
        
        print("✓ 系统初始化完成")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)
    
    def _check_project_structure(self):
        """检查项目目录结构"""
        print("检查项目结构...")
        
        required_dirs = {
            'src': '视觉处理代码',
            'Drone_Interface': '无人机接口',
            'Visual_process': '视觉处理工具',
        }
        
        for dir_name, description in required_dirs.items():
            dir_path = os.path.join(self.project_root, dir_name)
            if os.path.exists(dir_path):
                print(f"✓ 找到 {description}: {dir_path}")
            else:
                print(f"⚠ 缺失 {description}: {dir_path}")
    
    def _create_output_dir(self):
        """创建输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.project_root, "outputs", f"run_{timestamp}")
        
        # 创建子目录
        subdirs = ['images', 'flow', 'depth', 'segmentation', 'pose', 'logs']
        for subdir in subdirs:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
        
        return output_dir
    
    def _init_drone_interface(self):
        """初始化无人机接口"""
        print("初始化无人机接口...")
        
        try:
            # 尝试从 Drone_Interface 目录导入
            drone_interface_path = os.path.join(self.project_root, "Drone_Interface")
            
            if not os.path.exists(drone_interface_path):
                print(f"⚠ Drone_Interface目录不存在，使用模拟模式")
                return self._create_mock_drone_controller()
            
            # 查找Drone_Interface.py文件
            interface_files = [
                "Drone_Interface.py",
                "drone_interface.py",
                "interface.py",
                "__init__.py"
            ]
            
            for file_name in interface_files:
                file_path = os.path.join(drone_interface_path, file_name)
                if os.path.exists(file_path):
                    print(f"找到无人机接口文件: {file_path}")
                    
                    # 动态导入模块
                    try:
                        spec = importlib.util.spec_from_file_location(
                            "DroneInterfaceModule", 
                            file_path
                        )
                        drone_module = importlib.util.module_from_spec(spec)
                        sys.modules["DroneInterfaceModule"] = drone_module
                        spec.loader.exec_module(drone_module)
                        
                        # 寻找无人机控制器类
                        possible_classes = [
                            'DroneController', 'DroneInterface', 
                            'Drone', 'DroneClient', 'UAVController'
                        ]
                        
                        for class_name in possible_classes:
                            if hasattr(drone_module, class_name):
                                controller_class = getattr(drone_module, class_name)
                                controller = controller_class()
                                print(f"✓ 使用无人机控制类: {class_name}")
                                return controller
                        
                        # 如果没有找到标准类，尝试寻找任何类
                        for attr_name in dir(drone_module):
                            if not attr_name.startswith('_'):
                                attr = getattr(drone_module, attr_name)
                                if isinstance(attr, type) and 'Drone' in attr_name:
                                    controller = attr()
                                    print(f"✓ 使用无人机控制类: {attr_name}")
                                    return controller

                        # 如果存在 AirSim 特定实现文件，优先尝试加载（文件名: Drone_Interface_AirSim.py）
                        airsim_file = os.path.join(drone_interface_path, "Drone_Interface_AirSim.py")
                        if os.path.exists(airsim_file):
                            try:
                                spec_as = importlib.util.spec_from_file_location(
                                    "DroneAirSimModule", airsim_file
                                )
                                airsim_module = importlib.util.module_from_spec(spec_as)
                                sys.modules["DroneAirSimModule"] = airsim_module
                                spec_as.loader.exec_module(airsim_module)
                                if hasattr(airsim_module, 'DroneController'):
                                    controller_class = getattr(airsim_module, 'DroneController')
                                    controller = controller_class()
                                    print("✓ 使用 AirSim DroneController")
                                    return controller
                            except Exception as e:
                                print(f"导入 AirSim 控制器失败: {e}")
                                    
                    except Exception as e:
                        print(f"导入无人机接口失败: {e}")
            
            print("⚠ 未找到合适的无人机控制器，使用模拟模式")
            return self._create_mock_drone_controller()
            
        except Exception as e:
            print(f"无人机接口初始化失败: {e}")
            return self._create_mock_drone_controller()
    
    def _create_mock_drone_controller(self):
        """创建模拟无人机控制器"""
        print("创建模拟无人机控制器...")
        
        class MockDroneController:
            def __init__(self):
                self.connected = True
                self.frame_count = 0
                
            def capture_frame(self):
                """模拟捕获图像"""
                import cv2
                
                # 创建测试图像
                width, height = 640, 480
                img_array = np.zeros((height, width, 3), dtype=np.uint8)
                
                # 添加渐变背景
                for y in range(height):
                    color = int(100 + 100 * y / height)
                    img_array[y, :] = [color, color//2, 255-color]
                
                # 添加一些特征
                center_x, center_y = width//2, height//2
                
                # 圆形
                cv2.circle(img_array, (center_x, center_y), 100, (255, 200, 100), -1)
                # 矩形
                cv2.rectangle(img_array, (100, 100), (200, 200), (100, 200, 255), 2)
                # 文本
                cv2.putText(img_array, f"Frame {self.frame_count}", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img_array, "Mock Drone", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 添加一些随机点模拟特征点
                for _ in range(20):
                    x = np.random.randint(50, width-50)
                    y = np.random.randint(50, height-50)
                    cv2.circle(img_array, (x, y), 3, (0, 255, 0), -1)
                
                self.frame_count += 1
                return Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            
            def disconnect(self):
                print("模拟无人机已断开")
                
            def takeoff(self):
                print("模拟无人机起飞")
                return True
                
            def land(self):
                print("模拟无人机降落")
                return True
        
        return MockDroneController()
    
    def _setup_raft_paths(self):
        """设置RAFT路径 - 新增方法"""
        # RAFT目录结构
        raft_dir = os.path.join(self.project_root, "core_algorithms", "RAFT")
        raft_core_dir = os.path.join(raft_dir, "core")
        raft_utils_dir = os.path.join(raft_core_dir, "utils")
        
        print(f"\n设置RAFT路径:")
        print(f"  RAFT目录: {raft_dir}")
        print(f"  RAFT核心目录: {raft_core_dir}")
        print(f"  RAFT utils目录: {raft_utils_dir}")
        
        # 检查目录是否存在
        if not os.path.exists(raft_dir):
            print("  ⚠ RAFT目录不存在")
            return False
            
        if not os.path.exists(raft_core_dir):
            print("  ⚠ RAFT核心目录不存在")
            return False
            
        # 添加路径到sys.path（不要将 raft_core_dir/"utils" 直接加入 sys.path）
        paths_added = []
        for path in [raft_dir, raft_core_dir]:
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)
                paths_added.append(path)
                print(f"  ✓ 添加路径: {path}")
        if os.path.exists(raft_utils_dir):
            print(f"  (info) RAFT utils 目录存在: {raft_utils_dir}")
        
        # 检查关键文件
        required_files = [
            (raft_core_dir, 'raft.py'),
            (raft_core_dir, 'update.py'),
            (raft_core_dir, 'extractor.py'),
            (raft_core_dir, 'corr.py'),
        ]
        
        all_exist = True
        for dir_path, file_name in required_files:
            file_path = os.path.join(dir_path, file_name)
            if os.path.exists(file_path):
                print(f"  ✓ {file_name}")
            else:
                print(f"  ✗ {file_name} - 缺失")
                all_exist = False
        
        if all_exist:
            print("  ✅ RAFT文件检查通过")
            return True
        else:
            print("  ⚠ RAFT文件不完整，可能影响导入")
            return False
    
    def _init_vision_modules(self):
        """初始化视觉处理模块"""
        print("初始化视觉处理模块...")
        
        # 设置RAFT路径
        raft_path_ok = self._setup_raft_paths()
        
        try:
            # 首先尝试从 src 目录导入
            src_path = os.path.join(self.project_root, "src")
            
            if not os.path.exists(src_path):
                print(f"⚠ src目录不存在: {src_path}")
                return self._create_mock_vision_system()
            
            # 1. 尝试导入 demo.py 中的 UAVVisionSystem
            demo_path = os.path.join(src_path, "demo.py")
            if os.path.exists(demo_path):
                print(f"尝试导入 demo.py: {demo_path}")
                
                try:
                    # 在导入demo.py之前，确保RAFT路径已设置
                    if not raft_path_ok:
                        print("⚠ RAFT路径设置可能有问题，demo.py可能无法正确导入RAFT")
                    
                    # 动态导入demo模块
                    spec = importlib.util.spec_from_file_location("demo", demo_path)
                    demo_module = importlib.util.module_from_spec(spec)
                    
                    # 添加demo.py的目录到sys.path，以便它能找到相关模块
                    demo_dir = os.path.dirname(demo_path)
                    if demo_dir not in sys.path:
                        sys.path.insert(0, demo_dir)
                    
                    # 执行导入
                    spec.loader.exec_module(demo_module)
                    
                    if hasattr(demo_module, 'UAVVisionSystem'):
                        vision_system = demo_module.UAVVisionSystem()
                        print("✓ 使用 UAVVisionSystem")
                        return vision_system
                    else:
                        print("⚠ demo.py 中没有 UAVVisionSystem 类")
                except Exception as e:
                    print(f"导入 demo.py 失败: {e}")
                    traceback.print_exc()
            
            # 2. 尝试导入 Visual_process 模块
            visual_process_path = os.path.join(self.project_root, "Visual_process")
            if os.path.exists(visual_process_path):
                print(f"尝试导入 Visual_process 模块...")
                
                # 查找主要模块文件
                visual_files = [
                    "feature_point_detection.py",
                    "drone_test.py",
                    "visual_processor.py"
                ]
                
                for file_name in visual_files:
                    file_path = os.path.join(visual_process_path, file_name)
                    if os.path.exists(file_path):
                        print(f"找到视觉处理文件: {file_path}")
                        # 可以在这里导入特定的视觉处理函数
                        break
            
            # 3. 尝试从 src/models/modules 导入各个模块
            modules_path = os.path.join(src_path, "models", "modules")
            if os.path.exists(modules_path):
                print(f"尝试导入核心模块: {modules_path}")
                
                # 动态导入核心模块
                modules_imported = self._import_core_modules(modules_path)
                
                if modules_imported:
                    print("✓ 成功导入核心视觉模块")
                    return self._create_unified_vision_system(modules_imported)
            
            # 4. 如果都失败，使用模拟系统
            print("⚠ 无法导入视觉处理模块，使用模拟系统")
            return self._create_mock_vision_system()
            
        except Exception as e:
            print(f"视觉模块初始化失败: {e}")
            traceback.print_exc()
            return self._create_mock_vision_system()
    
    def _import_core_modules(self, modules_path):
        """导入核心模块"""
        modules = {}
        
        # 要导入的模块文件
        module_files = {
            'clustering': 'clustering.py',
            'depth_estimator': 'depth_estimator.py',
            'flow_processor': 'flow_processor.py',
            'geometry_utils': 'geometry_utils.py',
            'pose_estimator': 'pose_estimator.py'
        }
        
        for module_name, file_name in module_files.items():
            file_path = os.path.join(modules_path, file_name)
            if os.path.exists(file_path):
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    modules[module_name] = module
                    print(f"  ✓ 导入: {module_name}")
                except Exception as e:
                    print(f"  ✗ 导入 {module_name} 失败: {e}")
        
        return modules
    
    def _create_unified_vision_system(self, modules):
        """创建统一的视觉系统"""
        class UnifiedVisionSystem:
            def __init__(self, modules, output_dir):
                self.modules = modules
                self.output_dir = output_dir
                self.processors = {}
                
                print("初始化视觉处理器...")
                
                # 初始化深度估计器
                if 'depth_estimator' in modules:
                    try:
                        if hasattr(modules['depth_estimator'], 'Monodepth2Estimator'):
                            self.processors['depth'] = modules['depth_estimator'].Monodepth2Estimator()
                            print("  ✓ 深度估计器")
                    except:
                        pass
                
                # 初始化分割器
                if 'clustering' in modules:
                    try:
                        if hasattr(modules['clustering'], 'TraditionalSegmenter'):
                            self.processors['segmentation'] = modules['clustering'].TraditionalSegmenter(n_clusters=3)
                            print("  ✓ 分割器")
                    except:
                        pass
                
                # 初始化光流处理器
                if 'flow_processor' in modules:
                    try:
                        if hasattr(modules['flow_processor'], 'FlowProcessor'):
                            self.processors['flow'] = modules['flow_processor'].FlowProcessor()
                            print("  ✓ 光流处理器")
                    except:
                        pass
                
                # 初始化几何处理器
                if 'geometry_utils' in modules:
                    try:
                        if hasattr(modules['geometry_utils'], 'GeometryProcessor'):
                            # 假设的相机参数
                            camera_matrix = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]])
                            self.processors['geometry'] = modules['geometry_utils'].GeometryProcessor(camera_matrix)
                            print("  ✓ 几何处理器")
                    except:
                        pass
            
            def process_frame_pair(self, frame1, frame2, frame_idx):
                """处理两帧图像"""
                import cv2
                
                print(f"处理帧对 {frame_idx}")
                
                result = {
                    'frame_idx': frame_idx,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'processors_used': list(self.processors.keys())
                }
                
                # 转换为numpy数组
                img1_np = np.array(frame1)
                img2_np = np.array(frame2)
                
                # 如果有光流处理器，计算光流
                if 'flow' in self.processors:
                    try:
                        flow_result = self.processors['flow'].calculate_flow(img1_np, img2_np)
                        result['flow'] = {
                            'shape': flow_result.shape,
                            'n_features': len(flow_result.get('features', []))
                        }
                        print(f"  ✓ 光流计算完成")
                    except Exception as e:
                        print(f"  ✗ 光流计算失败: {e}")
                
                # 如果有深度估计器，计算深度
                if 'depth' in self.processors:
                    try:
                        depth_map = self.processors['depth'].estimate_depth(img1_np)
                        result['depth'] = {
                            'shape': depth_map.shape,
                            'min': float(np.min(depth_map)),
                            'max': float(np.max(depth_map))
                        }
                        print(f"  ✓ 深度估计完成")
                    except Exception as e:
                        print(f"  ✗ 深度估计失败: {e}")
                
                # 如果有分割器，进行分割
                if 'segmentation' in self.processors and 'flow' in result:
                    try:
                        # 假设光流结果中有特征点
                        if 'features' in flow_result:
                            points = flow_result['features']
                            if len(points) > 0:
                                labels, segment_info = self.processors['segmentation'].segment(
                                    points[:, :2], points[:, 2:]
                                )
                                result['segmentation'] = {
                                    'n_segments': len(segment_info),
                                    'labels_distribution': np.bincount(labels).tolist()
                                }
                                print(f"  ✓ 分割完成: {len(segment_info)} 个区域")
                    except Exception as e:
                        print(f"  ✗ 分割失败: {e}")
                
                return result
        
        return UnifiedVisionSystem(modules, self.output_dir)
    
    def _create_mock_vision_system(self):
        """创建模拟视觉系统"""
        class MockVisionSystem:
            def __init__(self, output_dir):
                self.output_dir = output_dir
                
            def process_frame_pair(self, frame1, frame2, frame_idx):
                """模拟处理流程"""
                print(f"[模拟] 处理帧对 {frame_idx}")
                
                # 转换为numpy数组用于分析
                img1_array = np.array(frame1)
                img2_array = np.array(frame2)
                
                # 生成随机但确定性的结果
                np.random.seed(frame_idx)
                
                result = {
                    'frame_idx': frame_idx,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'status': 'simulated',
                    'image_stats': {
                        'frame1_size': list(img1_array.shape),
                        'frame2_size': list(img2_array.shape),
                        'frame1_mean_color': [float(img1_array[:,:,i].mean()) for i in range(3)],
                        'frame2_mean_color': [float(img2_array[:,:,i].mean()) for i in range(3)],
                    },
                    'feature_points': {
                        'n_points': 150,
                        'points': (np.random.rand(150, 2) * 100).tolist(),
                        'confidence': (np.random.rand(150) * 0.5 + 0.5).tolist()
                    },
                    'segmentation': {
                        'n_segments': 4,
                        'segment_info': {
                            f'region_{i}': {
                                'label': i,
                                'n_points': np.random.randint(20, 60),
                                'center': (np.random.rand(2) * 100).tolist(),
                                'color_mean': (np.random.rand(3) * 255).tolist()
                            }
                            for i in range(4)
                        }
                    },
                    'depth_estimation': {
                        'available': True,
                        'map_shape': [img1_array.shape[0], img1_array.shape[1]],
                        'depth_range': [0.5, 15.0]
                    },
                    'pose_estimates': {
                        'camera_motion': {
                            'translation': (np.random.randn(3) * 0.1).tolist(),
                            'rotation': (np.random.randn(3) * 0.05).tolist(),
                            'confidence': 0.85
                        }
                    }
                }
                
                return result
        
        return MockVisionSystem(self.output_dir)
    
    def capture_two_frames(self):
        """从无人机捕获两帧图像"""
        print("\n捕获两帧图像...")
        
        frames = []
        for i in range(2):
            print(f"  捕获第{i+1}帧...")
            
            try:
                # 使用无人机控制器捕获图像
                frame = self.drone_controller.capture_frame()
                frames.append(frame)
                print(f"    尺寸: {frame.size}, 模式: {frame.mode}")
                
                # 保存原始图像
                frame_path = os.path.join(self.output_dir, 'images', 
                                         f'frame_{self.processing_count}_{i}.png')
                frame.save(frame_path)
                
                # 等待间隔（模拟无人机移动）
                if i < 1:
                    time.sleep(0.2)
                    
            except Exception as e:
                print(f"✗ 第{i+1}帧捕获失败: {e}")
                return None
        
        return frames
    
    def single_processing_cycle(self):
        """执行单次处理周期"""
        print(f"\n开始第{self.processing_count + 1}次处理周期")
        print("=" * 60)
        
        self.is_processing = True
        
        try:
            # 1. 捕获两帧图像
            frames = self.capture_two_frames()
            if not frames or len(frames) < 2:
                print("✗ 图像捕获失败")
                return None
            
            # 2. 处理图像
            result = self.vision_system.process_frame_pair(
                frames[0], frames[1], self.processing_count
            )
            
            # 3. 保存结果
            if result:
                self._save_result(result, frames)
                self._display_result_summary(result)
                print(f"\n✓ 第{self.processing_count + 1}次处理完成")
                self.processing_count += 1
            
            return result
            
        except Exception as e:
            print(f"✗ 处理失败: {e}")
            traceback.print_exc()
            return None
            
        finally:
            self.is_processing = False
    
    def _save_result(self, result, frames=None):
        """保存处理结果"""
        import cv2
        
        result_dir = os.path.join(self.output_dir, f"result_{self.processing_count:04d}")
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存图像
        if frames:
            for i, frame in enumerate(frames):
                # 保存为PNG
                frame_path = os.path.join(result_dir, f"input_frame_{i}.png")
                frame.save(frame_path)
                
                # 同时保存为numpy数组
                np_path = os.path.join(result_dir, f"input_frame_{i}.npy")
                np.save(np_path, np.array(frame))
        
        # 保存JSON结果
        json_path = os.path.join(result_dir, "result.json")
        
        def convert_for_json(obj):
            """递归转换numpy对象为JSON可序列化格式"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(convert_for_json(result), f, indent=2, ensure_ascii=False)
            print(f"✓ 结果保存到: {json_path}")
        except Exception as e:
            print(f"✗ 保存JSON失败: {e}")
    
    def _display_result_summary(self, result):
        """显示处理结果摘要"""
        print("\n" + "-" * 50)
        print("处理结果摘要:")
        print("-" * 50)
        
        print(f"帧索引: {result.get('frame_idx', 'N/A')}")
        print(f"时间戳: {result.get('timestamp', 'N/A')}")
        print(f"状态: {result.get('status', 'N/A')}")
        
        if 'image_stats' in result:
            stats = result['image_stats']
            print(f"图像1尺寸: {stats.get('frame1_size', 'N/A')}")
            print(f"图像2尺寸: {stats.get('frame2_size', 'N/A')}")
        
        if 'processors_used' in result:
            print(f"使用的处理器: {', '.join(result['processors_used'])}")
        
        if 'feature_points' in result:
            n_points = result['feature_points'].get('n_points', 0)
            print(f"特征点数量: {n_points}")
        
        if 'segmentation' in result:
            seg = result['segmentation']
            n_segments = seg.get('n_segments', 0)
            print(f"分割区域: {n_segments}")
        
        if 'depth' in result:
            depth = result['depth']
            print(f"深度图尺寸: {depth.get('shape', 'N/A')}")
            print(f"深度范围: [{depth.get('min', 0):.3f}, {depth.get('max', 0):.3f}]")
        
        if 'pose_estimates' in result:
            poses = result['pose_estimates']
            if 'camera_motion' in poses:
                motion = poses['camera_motion']
                t = motion.get('translation', [0, 0, 0])
                print(f"相机运动估计:")
                print(f"  平移: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
                print(f"  置信度: {motion.get('confidence', 0):.2f}")
        
        print("-" * 50)
    
    def run_multiple_cycles(self, n_cycles=5):
        """运行多个处理周期"""
        print(f"\n开始运行 {n_cycles} 个处理周期")
        print("=" * 60)
        
        results = []
        for i in range(n_cycles):
            print(f"\n[{i+1}/{n_cycles}]")
            result = self.single_processing_cycle()
            if result:
                results.append(result)
            
            # 如果不是最后一个周期，等待一下
            if i < n_cycles - 1:
                time.sleep(1.0)
        
        print(f"\n✓ 完成所有 {len(results)}/{n_cycles} 个处理周期")
        print(f"总结果保存到: {self.output_dir}")
        
        # 生成汇总报告
        self._generate_summary_report(results)
        
        return results
    
    def _generate_summary_report(self, results):
        """生成汇总报告"""
        report_path = os.path.join(self.output_dir, "summary_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 无人机视觉处理汇总报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总处理周期: {len(results)}\n")
            f.write(f"输出目录: {self.output_dir}\n\n")
            
            f.write("## 处理统计\n\n")
            f.write("| 周期 | 特征点数 | 分割区域 | 状态 |\n")
            f.write("|------|----------|----------|------|\n")
            
            for i, result in enumerate(results):
                n_points = result.get('feature_points', {}).get('n_points', 0)
                n_segments = result.get('segmentation', {}).get('n_segments', 0)
                status = result.get('status', 'unknown')
                f.write(f"| {i} | {n_points} | {n_segments} | {status} |\n")
        
        print(f"✓ 汇总报告: {report_path}")
    
    def cleanup(self):
        """清理资源"""
        print("\n清理系统资源...")
        if hasattr(self, 'drone_controller'):
            self.drone_controller.disconnect()
        print("✓ 系统清理完成")


# ========== 测试函数 ==========
def test_single_cycle():
    """测试单次处理"""
    print("测试无人机视觉整合系统 - 单次处理")
    print("=" * 60)
    
    integrator = DroneVisionIntegrator()
    result = integrator.single_processing_cycle()
    
    if result:
        print("\n✓ 单次处理成功")
        print(f"结果目录: {integrator.output_dir}")
    else:
        print("\n✗ 单次处理失败")
    
    integrator.cleanup()

if __name__ == "__main__":
    # 运行单次测试
    test_single_cycle()