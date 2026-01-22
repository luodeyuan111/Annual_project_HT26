"""
无人机动力学模型实现 - 对应论文第二章2.1节
用于模拟单架无人机的运动特性
"""

import numpy as np
import math


class Drone:
    """
    无人机类 - 模拟单架无人机的动力学行为
    对应论文公式(2-1)和(2-2)
    """

    def __init__(self, drone_id, initial_position=None, initial_velocity=None):
        """
        初始化无人机对象

        参数:
            drone_id: 无人机唯一标识符
            initial_position: 初始位置 [x, y] (单位: 米)
            initial_velocity: 初始速度 [vx, vy] (单位: 米/秒)
        """
        self.id = drone_id  # 无人机ID

        # 位置状态（论文中的Q^i）
        if initial_position is None:
            self.position = np.array([0.0, 0.0])  # 默认起始位置
        else:
            self.position = np.array(initial_position, dtype=float)

        # 速度状态（论文中的V^i）
        if initial_velocity is None:
            self.velocity = np.array([0.0, 0.0])  # 默认零速度
        else:
            self.velocity = np.array(initial_velocity, dtype=float)

        # 偏航角（论文中的φ^i）- 速度方向角
        self.yaw = 0.0  # 初始偏航角（弧度）

        # 控制输入（论文中的u^i）- 这里修正为加速度
        self.control_input = np.array([0.0, 0.0])

        # 无人机参数（根据论文表2-1和公式2-2）
        self.min_speed = 5.0  # 最小水平空速 V_xy^min (m/s)
        self.max_speed = 15.0  # 最大水平空速 V_xy^max (m/s)
        self.max_lateral_load = 10.0  # 最大侧向负载 n^max (g)
        self.gravity = 10.0  # 重力加速度 g (m/s^2)

        # 历史轨迹记录（用于可视化）
        self.trajectory = [self.position.copy()]

    def update_state(self, dt=0.1):
        """
        更新无人机状态

        对应论文公式:
        dQ^i/dt = V^i  (位置变化率 = 速度)
        dV^i/dt = u^i  (速度变化率 = 控制输入，即加速度)

        修正说明：控制输入 u^i 直接作为加速度
        """
        # 1. 控制输入作为加速度
        acceleration = self.control_input  # 控制输入就是加速度

        # 2. 更新速度：v_new = v_old + a * dt
        self.velocity += acceleration * dt

        # 3. 限制速度大小在[min_speed, max_speed]范围内（公式2-2约束）
        speed = np.linalg.norm(self.velocity)
        if speed > 0:
            if speed > self.max_speed:
                # 如果速度超过最大值，按比例缩小
                self.velocity = self.velocity / speed * self.max_speed
            elif speed < self.min_speed:
                # 如果速度低于最小值，按比例增大
                self.velocity = self.velocity / speed * self.min_speed

        # 4. 更新位置：p_new = p_old + v * dt
        self.position += self.velocity * dt

        # 5. 更新偏航角（从速度方向计算）
        if np.linalg.norm(self.velocity) > 0.01:  # 避免除零
            self.yaw = math.atan2(self.velocity[1], self.velocity[0])

        # 6. 记录轨迹
        self.trajectory.append(self.position.copy())

        # 7. 重置控制输入为0，为下一周期做准备
        self.control_input = np.array([0.0, 0.0])

    def set_control_input(self, control_input):
        """
        设置控制输入（论文中的u^i）

        参数:
            control_input: 控制输入向量 [ux, uy] (单位: m/s²，即加速度)
        """
        self.control_input = np.array(control_input, dtype=float)

    def get_position(self):
        """获取当前位置"""
        return self.position.copy()

    def get_velocity(self):
        """获取当前速度"""
        return self.velocity.copy()

    def get_speed(self):
        """获取当前速度大小（水平空速）"""
        return np.linalg.norm(self.velocity)

    def get_yaw(self):
        """获取当前偏航角（弧度）"""
        return self.yaw

    def get_yaw_degrees(self):
        """获取当前偏航角（度）"""
        return math.degrees(self.yaw)

    def distance_to(self, other_drone):
        """
        计算与另一架无人机的距离 - 对应论文中的d^ij

        参数:
            other_drone: 另一架无人机对象

        返回:
            两架无人机之间的欧氏距离（米）
        """
        # 计算位置差
        delta = self.position - other_drone.position
        # 计算欧氏距离
        distance = np.linalg.norm(delta)
        return distance

    def distance_to_point(self, point):
        """
        计算到指定点的距离

        参数:
            point: 目标点坐标 [x, y]

        返回:
            到目标点的欧氏距离
        """
        return np.linalg.norm(self.position - np.array(point))

    def get_state_vector(self):
        """
        获取无人机状态向量 [x, y, vx, vy]

        用于后续的神经网络输入（对应论文2.5.2节）
        """
        return np.concatenate([self.position, self.velocity])

    def get_trajectory(self):
        """获取历史轨迹"""
        return np.array(self.trajectory)

    def reset(self, position=None, velocity=None):
        """
        重置无人机状态

        参数:
            position: 新的位置
            velocity: 新的速度
        """
        if position is not None:
            self.position = np.array(position, dtype=float)
        if velocity is not None:
            self.velocity = np.array(velocity, dtype=float)
        self.trajectory = [self.position.copy()]
        self.control_input = np.array([0.0, 0.0])

    def __str__(self):
        """字符串表示，用于调试"""
        return (
            f"Drone {self.id}: "
            f"Position: [{self.position[0]:.2f}, {self.position[1]:.2f}], "
            f"Velocity: [{self.velocity[0]:.2f}, {self.velocity[1]:.2f}], "
            f"Speed: {self.get_speed():.2f} m/s, "
            f"Yaw: {self.get_yaw_degrees():.2f}°"
        )


# 测试函数
def test_drone():
    """测试无人机类的功能"""
    print("=== 测试无人机动力学模型（修正版）===")

    # 1. 创建一架无人机
    print("1. 创建无人机对象...")
    drone = Drone(drone_id=1, initial_position=[0, 0], initial_velocity=[1, 0])
    print(f"初始状态: {drone}")
    print()

    # 2. 测试设置控制输入（加速度）
    print("2. 设置控制输入为加速度...")
    drone.set_control_input([2, 1])  # 设置x方向加速度为2m/s²，y方向为1m/s²
    print(f"控制输入（加速度）已设置: {drone.control_input}")
    print()

    # 3. 测试状态更新
    print("3. 更新无人机状态...")
    print("理论计算:")
    print("  - 初始速度: [1, 0] m/s")
    print("  - 加速度: [2, 1] m/s²")
    print("  - 时间步长: 0.5 s")
    print("  - 速度变化: Δv = a * dt = [1, 0.5] m/s")
    print("  - 预期第1步速度: [2, 0.5] m/s")
    print()

    for i in range(5):
        drone.update_state(dt=0.5)  # 时间步长0.5秒
        print(f"时间步 {i+1}: {drone}")
    print()

    # 4. 测试距离计算
    print("4. 测试距离计算...")
    drone2 = Drone(drone_id=2, initial_position=[10, 5])
    distance = drone.distance_to(drone2)
    print(f"无人机{drone.id}和无人机{drone2.id}的距离: {distance:.2f} 米")
    print()

    # 5. 测试获取状态向量
    print("5. 获取状态向量...")
    state_vector = drone.get_state_vector()
    print(f"状态向量: {state_vector}")
    print()

    print("=== 测试完成 ===")


if __name__ == "__main__":
    # 如果直接运行此文件，执行测试
    test_drone()
