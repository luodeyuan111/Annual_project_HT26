import numpy as np
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import cv2
import torch
import os

# Local imports from the same directory
from .flow_processor import FlowProcessor
from .depth_estimator import Monodepth2Estimator
from .clustering import TraditionalSegmenter

@dataclass
class NeuralOutput:
    """神经网络感知输出 - 标准化接口"""
    # 1. 基础信息
    timestamp: float = 0.0
    frame_idx: int = 0
    input_resolution: Tuple[int, int] = (640, 480)
    
    # 2. 光流信息
    optical_flow: Dict = None
    
    # 3. 深度信息
    depth_maps: Dict = None
    
    # 4. 特征点信息
    feature_points: Dict = None
    
    # 5. 分割信息
    segmentation: Dict = None
    
    # 6. 质量指标
    quality_metrics: Dict = None
    
    # 7. 调试信息
    debug: Dict = None

    def __post_init__(self):
        if self.optical_flow is None:
            self.optical_flow = {}
        if self.depth_maps is None:
            self.depth_maps = {}
        if self.feature_points is None:
            self.feature_points = {}
        if self.segmentation is None:
            self.segmentation = {}
        if self.quality_metrics is None:
            self.quality_metrics = {}
        if self.debug is None:
            self.debug = {}

class NeuralPerception:
    """神经网络感知主模块 - 集成现有神经网络函数"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化神经网络感知模块
        
        Args:
            config: 配置字典，包含模型路径、参数等
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        self.frame_idx = 0
        self.current_timestamp = 0.0
        
        # 初始化子模块
        self._init_submodules()
        
        print("NeuralPerception模块初始化完成")
        print(f"配置: 设备={self.config['device']}, 特征网格步长={self.config['feature_grid_step']}")
    
    def _get_default_config(self) -> Dict:
        """默认配置"""
        return {
            # 设备和通用
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            
            # 光流配置
            'raft_model_path': 'models/RAFT/models/raft-things.pth',
            'raft_iterations': 20,
            
            # 深度配置
            'monodepth2_model_path': 'models/monodepth2/mono+stereo_640x192',
            'depth_min': 0.1,
            'depth_max': 100.0,
            'depth_scale_factor': 5.4,
            'input_width': 640,
            'input_height': 192,
            
            # 特征提取
            'feature_grid_step': 8,
            'flow_magnitude_threshold': 0.5,  # 最小光流幅度
            
            # 分割配置
            'segmentation_method': 'kmeans',
            'n_clusters': 3,
            'min_region_size': 50,
            
            # 质量阈值
            'min_features': 50,
            'min_valid_depth_ratio': 0.3,
            'flow_confidence_threshold': 0.5
        }
    
    def _init_submodules(self):
        """初始化子模块"""
        # 光流处理器
        flow_config = {
            'model_path': self.config['raft_model_path'],
            'iterations': self.config['raft_iterations'],
            'use_gpu': self.config['device'] == 'cuda'
        }
        self.flow_processor = FlowProcessor(flow_config)
        
        # 深度估计器
        depth_config = {
            'model_path': self.config['monodepth2_model_path'],
            'input_width': self.config['input_width'],
            'input_height': self.config['input_height'],
            'min_depth': self.config['depth_min'],
            'max_depth': self.config['depth_max'],
            'scale_factor': self.config['depth_scale_factor'],
            'device': self.config['device']
        }
        self.depth_estimator = Monodepth2Estimator(**depth_config)
        
        # 分割器
        seg_config = {
            'method': self.config['segmentation_method'],
            'n_clusters': self.config['n_clusters']
        }
        self.segmenter = TraditionalSegmenter(**seg_config)
    
    def process_frame_pair(self, frame_t: np.ndarray, frame_t_plus_1: np.ndarray) -> NeuralOutput:
        """
        处理图像对，返回标准化神经网络输出
        
        Args:
            frame_t: t时刻图像 [H, W, 3] numpy uint8 (BGR or RGB)
            frame_t_plus_1: t+1时刻图像 [H, W, 3] numpy uint8
            
        Returns:
            neural_output: NeuralOutput对象，包含所有感知信息
        """
        start_time = time.time()
        self.current_timestamp = time.time()
        
        # 确保输入格式一致 (转换为RGB)
        if len(frame_t.shape) == 3 and frame_t.shape[2] == 3:
            if frame_t[0, 0, 0] + frame_t[0, 0, 1] + frame_t[0, 0, 2] < 200:  # 粗略判断BGR
                frame_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2RGB)
                frame_t_plus_1 = cv2.cvtColor(frame_t_plus_1, cv2.COLOR_BGR2RGB)
        
        h, w = frame_t.shape[:2]
        self.frame_idx += 1
        
        try:
            # 1. 计算光流
            flow_start = time.time()
            flow_field = self.flow_processor.compute_flow(frame_t, frame_t_plus_1, return_type='numpy')
            flow_time = time.time() - flow_start
            
            # 计算光流统计
            flow_stats = self.flow_processor.compute_flow_statistics(flow_field)
            flow_magnitude = flow_stats['mean_magnitude']
            flow_confidence = np.mean((flow_stats['mean_magnitude'] > 0.1) & (flow_stats['mean_magnitude'] < 20.0))
            
            # 2. 提取特征点
            feature_start = time.time()
            points_t, points_t1, flow_vectors, valid_mask = self.flow_processor.extract_feature_points(
                flow_field, 
                grid_step=self.config['feature_grid_step'],
                threshold=self.config['flow_magnitude_threshold']
            )
            feature_time = time.time() - feature_start
            
            n_features = len(points_t)
            
            # 3. 估计深度
            depth_start = time.time()
            depth_t = self.depth_estimator.estimate_depth(frame_t, input_format='RGB')
            depth_t1 = self.depth_estimator.estimate_depth(frame_t_plus_1, input_format='RGB')
            depth_time = time.time() - depth_start
            
            # 深度有效性掩码
            valid_depth_t = depth_t > self.config['depth_min']
            valid_depth_t1 = depth_t1 > self.config['depth_min']
            valid_depth_ratio = np.mean(valid_depth_t & valid_depth_t1)
            depth_confidence = min(valid_depth_ratio * 2, 1.0)  # 线性映射
            
            # 4. 分割
            seg_start = time.time()
            if n_features > 10:  # 至少需要10个点进行分割
                labels, segment_info = self.segmenter.segment(points_t, flow_vectors)
                n_segments = len(segment_info)
                segmentation_method = self.config['segmentation_method']
            else:
                # 点数不足，使用简单分割
                labels = np.zeros(n_features, dtype=int)
                segment_info = {'fallback': {'label': 0, 'n_points': n_features}}
                n_segments = 1
                segmentation_method = 'fallback'
            seg_time = time.time() - seg_start
            
            # 5. 质量评估
            quality_start = time.time()
            quality_metrics = self._assess_quality(
                flow_confidence, depth_confidence, n_features, n_segments,
                flow_magnitude, valid_depth_ratio
            )
            quality_time = time.time() - quality_start
            
            # 6. 组装输出
            neural_output = NeuralOutput(
                timestamp=self.current_timestamp,
                frame_idx=self.frame_idx,
                input_resolution=(w, h)
            )
            
            # 填充光流信息
            neural_output.optical_flow = {
                'flow_field': flow_field,
                'flow_magnitude': float(flow_magnitude),
                'flow_confidence': float(flow_confidence),
                'valid_mask': valid_mask,
                'processing_time': float(flow_time),
                **{k: v for k, v in flow_stats.items() if k not in ['mean_magnitude']}
            }
            
            # 填充深度信息
            neural_output.depth_maps = {
                'depth_t': depth_t,
                'depth_t_plus_1': depth_t1,
                'depth_range': [self.config['depth_min'], self.config['depth_max']],
                'depth_confidence': float(depth_confidence),
                'valid_depth_mask_t': valid_depth_t,
                'valid_depth_mask_t_plus_1': valid_depth_t1,
                'processing_time': float(depth_time)
            }
            
            # 填充特征点信息
            neural_output.feature_points = {
                'points_t': points_t,
                'points_t_plus_1': points_t1,
                'flow_vectors': flow_vectors,
                'n_features': n_features,
                'n_valid': int(np.sum(valid_mask)),
                'extraction_method': 'grid_flow',
                'processing_time': float(feature_time)
            }
            
            # 填充分割信息
            neural_output.segmentation = {
                'labels': labels,
                'segments': segment_info,
                'n_segments': n_segments,
                'method': segmentation_method,
                'processing_time': float(seg_time)
            }
            
            # 填充质量指标
            neural_output.quality_metrics = quality_metrics
            
            # 填充调试信息
            neural_output.debug = {
                'total_processing_time': float(time.time() - start_time),
                'flow_time': float(flow_time),
                'depth_time': float(depth_time),
                'feature_time': float(feature_time),
                'seg_time': float(seg_time),
                'quality_time': float(quality_time),
                'raft_iterations': self.config['raft_iterations'],
                'feature_grid_step': self.config['feature_grid_step'],
                'kmeans_n_clusters': self.config['n_clusters']
            }
            
            # 警告信息
            warnings = []
            if flow_confidence < self.config['flow_confidence_threshold']:
                warnings.append('low_flow_confidence')
            if valid_depth_ratio < self.config['min_valid_depth_ratio']:
                warnings.append('insufficient_depth_coverage')
            if n_features < self.config['min_features']:
                warnings.append('insufficient_features')
            if n_segments < 2:
                warnings.append('limited_segmentation')
            
            neural_output.quality_metrics['warnings'] = warnings
            
            print(f"帧 {self.frame_idx} 处理完成:")
            print(f"  特征点: {n_features}, 分割区域: {n_segments}")
            print(f"  质量: {quality_metrics['overall_confidence']:.3f}")
            if warnings:
                print(f"  警告: {', '.join(warnings)}")
            
            return neural_output
            
        except Exception as e:
            print(f"神经网络处理失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回降级输出
            return self._create_fallback_output(h, w)
    
    def _assess_quality(self, flow_conf, depth_conf, n_features, n_segments, 
                       flow_mag, valid_depth_ratio) -> Dict:
        """评估感知质量"""
        # 各组件置信度
        feature_conf = min(n_features / 200.0, 1.0)  # 假设200个点为理想
        seg_conf = min(n_segments / 3.0, 1.0) if n_segments > 0 else 0.5
        
        # 运动合理性
        motion_reasonable = 1.0 if (flow_mag > 0.1 and flow_mag < 10.0) else 0.5
        
        # 整体质量 (加权平均)
        overall_conf = (
            flow_conf * 0.3 +
            depth_conf * 0.3 +
            feature_conf * 0.2 +
            seg_conf * 0.1 +
            motion_reasonable * 0.1
        )
        
        # 场景复杂度 (基于特征点密度和运动变化)
        scene_complexity = min((n_features / 100.0 + flow_mag / 5.0) / 2, 1.0)
        
        return {
            'overall_confidence': float(overall_conf),
            'flow_confidence': float(flow_conf),
            'depth_confidence': float(depth_conf),
            'feature_confidence': float(feature_conf),
            'segmentation_confidence': float(seg_conf),
            'motion_reasonable': float(motion_reasonable),
            'scene_complexity': float(scene_complexity),
            'warnings': []
        }
    
    def _create_fallback_output(self, h: int, w: int) -> NeuralOutput:
        """创建降级输出"""
        # 创建虚拟数据
        fallback_flow = np.zeros((h, w, 2), dtype=np.float32)
        fallback_depth = np.ones((h, w), dtype=np.float32) * 10.0  # 10米默认深度
        
        neural_output = NeuralOutput(
            timestamp=self.current_timestamp,
            frame_idx=self.frame_idx,
            input_resolution=(w, h)
        )
        
        neural_output.optical_flow = {
            'flow_field': fallback_flow,
            'flow_magnitude': 0.0,
            'flow_confidence': 0.0,
            'valid_mask': np.zeros((0,), dtype=bool),
            'processing_time': 0.0,
            'error': 'processing_failed'
        }
        
        neural_output.depth_maps = {
            'depth_t': fallback_depth,
            'depth_t_plus_1': fallback_depth,
            'depth_range': [self.config['depth_min'], self.config['depth_max']],
            'depth_confidence': 0.0,
            'valid_depth_mask_t': np.ones((h, w), dtype=bool),
            'valid_depth_mask_t_plus_1': np.ones((h, w), dtype=bool),
            'processing_time': 0.0,
            'error': 'processing_failed'
        }
        
        neural_output.feature_points = {
            'points_t': np.empty((0, 2), dtype=np.float32),
            'points_t_plus_1': np.empty((0, 2), dtype=np.float32),
            'flow_vectors': np.empty((0, 2), dtype=np.float32),
            'n_features': 0,
            'n_valid': 0,
            'extraction_method': 'fallback',
            'processing_time': 0.0,
            'error': 'insufficient_data'
        }
        
        neural_output.segmentation = {
            'labels': np.array([], dtype=int),
            'segments': {},
            'n_segments': 0,
            'method': 'fallback',
            'processing_time': 0.0,
            'error': 'insufficient_features'
        }
        
        neural_output.quality_metrics = {
            'overall_confidence': 0.0,
            'warnings': ['processing_failed', 'fallback_mode']
        }
        
        neural_output.debug = {
            'total_processing_time': 0.0,
            'error': 'processing_failed'
        }
        
        return neural_output
    
    def get_status(self) -> Dict:
        """获取模块状态"""
        return {
            'initialized': True,
            'frame_count': self.frame_idx,
            'device': self.config['device'],
            'config': {k: v for k, v in self.config.items() if k not in ['model_path']},
            'submodules': {
                'flow_processor': 'ready',
                'depth_estimator': 'ready',
                'segmenter': 'ready'
            }
        }
    
    def visualize_results(self, neural_output: NeuralOutput, save_dir: str = None) -> Dict:
        """
        可视化感知结果
        
        Args:
            neural_output: NeuralOutput对象
            save_dir: 保存目录（可选）
            
        Returns:
            vis_paths: 可视化图像路径字典
        """
        vis_paths = {}
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 光流可视化
        if neural_output.optical_flow.get('flow_field') is not None:
            flow_vis = self.flow_processor.visualize_flow(
                neural_output.optical_flow['flow_field']
            )
            flow_path = os.path.join(save_dir, f"flow_{neural_output.frame_idx:04d}.png") if save_dir else None
            if flow_path:
                cv2.imwrite(flow_path, cv2.cvtColor(flow_vis, cv2.COLOR_RGB2BGR))
            vis_paths['flow'] = flow_path
        
        # 深度可视化
        if neural_output.depth_maps.get('depth_t') is not None:
            depth_vis_t = self.depth_estimator.visualize_depth(neural_output.depth_maps['depth_t'])
            depth_path_t = os.path.join(save_dir, f"depth_t_{neural_output.frame_idx:04d}.png") if save_dir else None
            if depth_path_t:
                cv2.imwrite(depth_path_t, depth_vis_t)
            vis_paths['depth_t'] = depth_path_t
            
            depth_vis_t1 = self.depth_estimator.visualize_depth(neural_output.depth_maps['depth_t_plus_1'])
            depth_path_t1 = os.path.join(save_dir, f"depth_t1_{neural_output.frame_idx:04d}.png") if save_dir else None
            if depth_path_t1:
                cv2.imwrite(depth_path_t1, depth_vis_t1)
            vis_paths['depth_t1'] = depth_path_t1
        
        # 分割可视化 (需要图像背景，这里简化)
        if neural_output.segmentation.get('labels') is not None and len(neural_output.segmentation['labels']) > 0:
            # 创建简单可视化 (点云形式)
            h, w = neural_output.input_resolution
            seg_vis = np.zeros((h, w, 3), dtype=np.uint8)
            
            points_t = neural_output.feature_points['points_t']
            labels = neural_output.segmentation['labels']
            
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            for i, (pt, label) in enumerate(zip(points_t, labels)):
                if label >= 0 and label < len(colors):
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= y < h and 0 <= x < w:
                        cv2.circle(seg_vis, (x, y), 3, colors[label], -1)
            
            seg_path = os.path.join(save_dir, f"segmentation_{neural_output.frame_idx:04d}.png") if save_dir else None
            if seg_path:
                cv2.imwrite(seg_path, seg_vis)
            vis_paths['segmentation'] = seg_path
        
        return vis_paths

# 测试函数
def test_neural_perception():
    """测试NeuralPerception模块"""
    import cv2
    import matplotlib.pyplot as plt
    
    print("测试NeuralPerception模块...")
    
    # 创建测试图像
    H, W = 240, 320  # 小尺寸测试
    frame_t = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    frame_t_plus_1 = np.roll(frame_t, shift=3, axis=(0, 1))  # 简单平移
    
    # 添加一些结构
    cv2.rectangle(frame_t, (50, 50), (100, 100), (255, 0, 0), -1)
    cv2.rectangle(frame_t_plus_1, (53, 53), (103, 103), (255, 0, 0), -1)
    
    # 初始化模块
    config = {
        'device': 'cpu',  # 测试用CPU
        'raft_model_path': 'models/raft/raft-things.pth',  # 确保路径正确
        'feature_grid_step': 16  # 稀疏网格
    }
    
    try:
        perception = NeuralPerception(config)
        
        # 处理帧对
        neural_output = perception.process_frame_pair(frame_t, frame_t_plus_1)
        
        print(f"\n处理结果:")
        print(f"时间戳: {neural_output.timestamp:.3f}")
        print(f"帧索引: {neural_output.frame_idx}")
        print(f"输入分辨率: {neural_output.input_resolution}")
        
        print(f"\n光流信息:")
        print(f"  光流场形状: {neural_output.optical_flow['flow_field'].shape}")
        print(f"  平均幅度: {neural_output.optical_flow['flow_magnitude']:.3f}")
        print(f"  置信度: {neural_output.optical_flow['flow_confidence']:.3f}")
        
        print(f"\n深度信息:")
        print(f"  深度图形状: {neural_output.depth_maps['depth_t'].shape}")
        print(f"  深度置信度: {neural_output.depth_maps['depth_confidence']:.3f}")
        
        print(f"\n特征点信息:")
        print(f"  特征点数量: {neural_output.feature_points['n_features']}")
        print(f"  有效匹配: {neural_output.feature_points['n_valid']}")
        
        print(f"\n分割信息:")
        print(f"  分割区域数: {neural_output.segmentation['n_segments']}")
        print(f"  方法: {neural_output.segmentation['method']}")
        
        print(f"\n质量指标:")
        print(f"  整体置信度: {neural_output.quality_metrics['overall_confidence']:.3f}")
        if neural_output.quality_metrics['warnings']:
            print(f"  警告: {', '.join(neural_output.quality_metrics['warnings'])}")
        
        print(f"\n调试信息:")
        print(f"  总处理时间: {neural_output.debug['total_processing_time']:.3f}s")
        
        # 可视化 (简化)
        vis_paths = perception.visualize_results(neural_output, save_dir='./test_output')
        print(f"\n可视化保存到: {vis_paths}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_neural_perception()
    if success:
        print("\nNeuralPerception测试通过！")
    else:
        print("\nNeuralPerception测试失败！")
