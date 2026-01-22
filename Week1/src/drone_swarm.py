"""
无人机集群类 - 对应论文第二章2.2节集群模型
用于管理多架无人机的集群行为
"""

import numpy as np
import math
from src.drone import Drone  # 导入之前的Drone类


class DroneSwarm:
    """
    无人机集群类 - 管理多架无人机的集群行为
    对应论文2.2节集群模型
    """

    def __init__(self, num_drones=5, max_communication_distance=20.0):
        """
        初始化无人机集群

        参数:
            num_drones: 无人机数量
            max_communication_distance: 最大通信距离，用于邻居检测
        """
        self.num_drones = num_drones
        self.max_comm_distance = max_communication_distance

        # 创建无人机列表
        self.drones = []
        self.initialize_drones()

        # 集群参数（对应论文参数）
        self.cohesion_strength = 0.1  # 聚集强度
        self.separation_strength = 0.2  # 分离强度
        self.alignment_strength = 0.15  # 对齐强度

        # 期望距离和最小距离
        self.desired_distance = 10.0  # 期望距离 (D_d)
        self.min_separation_distance = 2.0  # 最小分离距离 (D_l1)

        # 集群历史记录（用于可视化）
        self.history = {
            "positions": [],  # 每帧所有无人机的位置
            "velocities": [],  # 每帧所有无人机的速度
            "neighbors": [],  # 每帧邻居关系
        }

    def initialize_drones(self):
        """初始化所有无人机，随机分布在指定区域内"""
        np.random.seed(42)  # 固定随机种子，确保结果可重现

        for i in range(self.num_drones):
            # 随机位置（在100x100区域内）
            pos_x = np.random.uniform(0, 100)
            pos_y = np.random.uniform(0, 100)

            # 随机速度方向，速度大小在5-10之间
            speed = np.random.uniform(5, 10)
            angle = np.random.uniform(0, 2 * math.pi)
            vel_x = speed * math.cos(angle)
            vel_y = speed * math.sin(angle)

            # 创建无人机
            drone = Drone(
                drone_id=i + 1,
                initial_position=[pos_x, pos_y],
                initial_velocity=[vel_x, vel_y],
            )
            self.drones.append(drone)

    def get_neighbors(self, drone_id):
        """
        获取指定无人机的邻居

        参数:
            drone_id: 无人机ID（1-based）

        返回:
            邻居无人机ID列表
        """
        neighbors = []
        current_drone = self.drones[drone_id - 1]  # ID-1转为0-based索引

        for other_id in range(1, self.num_drones + 1):
            if other_id == drone_id:
                continue  # 跳过自己

            other_drone = self.drones[other_id - 1]
            distance = current_drone.distance_to(other_drone)

            # 如果距离小于最大通信距离，认为是邻居
            if distance <= self.max_comm_distance:
                neighbors.append(other_id)

        return neighbors

    def calculate_cohesion(self, drone_id, neighbors):
        """
        计算聚集力 - 使无人机向邻居中心移动

        参数:
            drone_id: 当前无人机ID
            neighbors: 邻居ID列表

        返回:
            聚集加速度 [ax, ay]
        """
        if not neighbors:
            return np.array([0.0, 0.0])

        current_drone = self.drones[drone_id - 1]

        # 计算邻居平均位置
        avg_position = np.array([0.0, 0.0])
        for neighbor_id in neighbors:
            neighbor_drone = self.drones[neighbor_id - 1]
            avg_position += neighbor_drone.get_position()

        avg_position /= len(neighbors)

        # 计算指向平均位置的方向向量
        direction = avg_position - current_drone.get_position()
        distance = np.linalg.norm(direction)

        if distance > 0:
            # 归一化并乘以强度系数
            direction = direction / distance
            cohesion_force = direction * self.cohesion_strength
        else:
            cohesion_force = np.array([0.0, 0.0])

        return cohesion_force

    def calculate_separation(self, drone_id, neighbors):
        """
        计算分离力 - 使无人机远离太近的邻居

        参数:
            drone_id: 当前无人机ID
            neighbors: 邻居ID列表

        返回:
            分离加速度 [ax, ay]
        """
        if not neighbors:
            return np.array([0.0, 0.0])

        current_drone = self.drones[drone_id - 1]
        separation_force = np.array([0.0, 0.0])

        for neighbor_id in neighbors:
            neighbor_drone = self.drones[neighbor_id - 1]
            distance = current_drone.distance_to(neighbor_drone)

            if distance < self.min_separation_distance:
                # 计算远离邻居的方向
                direction = current_drone.get_position() - neighbor_drone.get_position()
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    # 距离越近，排斥力越大
                    strength = self.separation_strength * (
                        1.0 - distance / self.min_separation_distance
                    )
                    separation_force += direction * strength

        return separation_force

    def calculate_alignment(self, drone_id, neighbors):
        """
        计算对齐力 - 使无人机速度与邻居平均速度对齐

        参数:
            drone_id: 当前无人机ID
            neighbors: 邻居ID列表

        返回:
            对齐加速度 [ax, ay]
        """
        if not neighbors:
            return np.array([0.0, 0.0])

        current_drone = self.drones[drone_id - 1]

        # 计算邻居平均速度
        avg_velocity = np.array([0.0, 0.0])
        for neighbor_id in neighbors:
            neighbor_drone = self.drones[neighbor_id - 1]
            avg_velocity += neighbor_drone.get_velocity()

        avg_velocity /= len(neighbors)

        # 计算速度差
        velocity_diff = avg_velocity - current_drone.get_velocity()

        # 乘以对齐强度系数
        alignment_force = velocity_diff * self.alignment_strength

        return alignment_force

    def update_swarm(self, dt=0.1):
        """
        更新整个集群状态

        参数:
            dt: 时间步长
        """
        # 记录当前状态
        current_positions = []
        current_velocities = []
        current_neighbors = []

        # 第一遍：检测邻居
        neighbors_list = []
        for drone in self.drones:
            neighbors = self.get_neighbors(drone.id)
            neighbors_list.append(neighbors)
            current_neighbors.append(neighbors)

        # 第二遍：计算控制输入并更新
        for i, drone in enumerate(self.drones):
            drone_id = drone.id
            neighbors = neighbors_list[i]

            # 计算三种力
            cohesion_force = self.calculate_cohesion(drone_id, neighbors)
            separation_force = self.calculate_separation(drone_id, neighbors)
            alignment_force = self.calculate_alignment(drone_id, neighbors)

            # 合成总控制输入（加速度）
            total_force = cohesion_force + separation_force + alignment_force

            # 设置控制输入
            drone.set_control_input(total_force)

            # 更新无人机状态
            drone.update_state(dt)

            # 记录状态
            current_positions.append(drone.get_position())
            current_velocities.append(drone.get_velocity())

        # 保存历史记录
        self.history["positions"].append(np.array(current_positions))
        self.history["velocities"].append(np.array(current_velocities))
        self.history["neighbors"].append(current_neighbors)

    def get_swarm_state(self):
        """
        获取集群当前状态

        返回:
            包含所有无人机状态的字典
        """
        state = {
            "positions": np.array([d.get_position() for d in self.drones]),
            "velocities": np.array([d.get_velocity() for d in self.drones]),
            "speeds": np.array([d.get_speed() for d in self.drones]),
            "yaws": np.array([d.get_yaw_degrees() for d in self.drones]),
        }
        return state

    def get_history(self):
        """获取历史记录"""
        return self.history

    def reset(self):
        """重置集群到初始状态"""
        self.drones = []
        self.initialize_drones()
        self.history = {"positions": [], "velocities": [], "neighbors": []}

    def __str__(self):
        """字符串表示"""
        output = f"无人机集群 ({self.num_drones}架):\n"
        for drone in self.drones:
            output += f"  {drone}\n"
        return output


# 测试函数
def test_swarm():
    """测试无人机集群功能"""
    print("=== 测试无人机集群 ===")

    # 创建5架无人机的集群
    swarm = DroneSwarm(num_drones=5)
    print(swarm)

    # 测试邻居检测
    print("\n1. 测试邻居检测:")
    for i in range(1, 6):
        neighbors = swarm.get_neighbors(i)
        print(f"  无人机{i}的邻居: {neighbors}")

    # 测试更新集群
    print("\n2. 测试集群更新:")
    for step in range(3):
        swarm.update_swarm(dt=0.5)
        state = swarm.get_swarm_state()
        avg_speed = np.mean(state["speeds"])
        print(f"  时间步 {step+1}: 平均速度 = {avg_speed:.2f} m/s")

    print("\n3. 查看更新后状态:")
    print(swarm)

    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_swarm()
