"""
命令行接口 - 简化版
"""

import cmd
import sys
import os

class DroneVisionCommandInterface(cmd.Cmd):
    """无人机视觉系统命令行接口"""
    
    intro = """
==================================================
无人机视觉处理系统 - 命令行接口
==================================================
输入 help 或 ? 查看命令列表
输入 q 或 quit 退出系统
==================================================
"""
    
    prompt = '(drone-vision) > '
    
    def __init__(self, integrator):
        super().__init__()
        self.integrator = integrator
    
    def do_process(self, arg):
        """执行一次处理周期: process"""
        print("\n执行处理周期...")
        self.integrator.single_processing_cycle()
    
    def do_batch(self, arg):
        """批量处理多个周期: batch [次数]"""
        try:
            n_cycles = int(arg) if arg else 5
            print(f"\n批量处理 {n_cycles} 个周期...")
            
            for i in range(n_cycles):
                print(f"\n>>> 周期 {i+1}/{n_cycles}")
                self.integrator.single_processing_cycle()
            
            print(f"\n批量处理完成")
            
        except ValueError:
            print("错误: 请输入有效的数字")
    
    def do_status(self, arg):
        """显示系统状态: status"""
        print("\n系统状态:")
        print("-" * 40)
        print(f"处理次数: {self.integrator.processing_count}")
        print(f"正在处理: {'是' if self.integrator.is_processing else '否'}")
        print(f"输出目录: {self.integrator.output_dir}")
        print("-" * 40)
    
    def do_test(self, arg):
        """测试系统组件: test"""
        print("\n测试系统组件...")
        
        # 测试无人机
        if hasattr(self.integrator, 'drone_controller'):
            print("✓ 无人机控制器: 已连接")
        else:
            print("✗ 无人机控制器: 未连接")
        
        # 测试视觉系统
        if hasattr(self.integrator, 'vision_system'):
            print("✓ 视觉系统: 已初始化")
        else:
            print("✗ 视觉系统: 未初始化")
    
    def do_q(self, arg):
        """退出系统: q, quit, exit"""
        print("\n正在退出系统...")
        self.integrator.cleanup()
        return True
    
    do_quit = do_q
    do_exit = do_q
    
    def default(self, line):
        """处理未知命令"""
        print(f"未知命令: {line}")
        print("可用命令: process, batch, status, test, quit")