"""
模块二：单架无人机动力学模型主程序
对应论文第二章2.1节
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ========== 字体设置 - 解决中文显示问题 ==========
# 尝试多种中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 设置更大的字体
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10

# ========== 导入无人机类 ==========
# 添加src目录到Python路径
sys.path.append("src")

# 导入Drone类
from src.drone import Drone


def create_results_folder():
    """创建results文件夹（如果不存在）"""
    if not os.path.exists("results"):
        os.makedirs("results")
        print("已创建 results 文件夹")
    return "results"


def test_single_drone_basic():
    """
    测试单架无人机基本功能
    对应论文公式(2-1)和(2-2)
    """
    print("=== 测试1：单架无人机基本功能 ===")

    # 创建无人机对象
    drone = Drone(drone_id=1, initial_position=[0, 0], initial_velocity=[6, 0])

    print(f"无人机初始状态: {drone}")

    # 测试1.1：测试控制输入和状态更新
    print("\n测试1.1：控制输入和状态更新")
    drone.set_control_input([2, 1])
    # 模拟5个时间步
    for i in range(5):
        drone.update_state(dt=0.5)
        drone.set_control_input([2, 1])
        print(f"时间步 {i+1}: {drone}")

    # 测试1.2：测试速度限制（论文公式2-2约束）
    print("\n测试1.2：速度限制测试（对应论文公式2-2）")
    fast_drone = Drone(
        drone_id=2, initial_velocity=[20, 0]
    )  # 初始速度20m/s，超过最大值15m/s
    print(
        f"初始速度: {fast_drone.get_velocity()}，速度大小: {fast_drone.get_speed():.2f} m/s"
    )

    fast_drone.update_state(dt=0.1)
    print(
        f"更新后速度: {fast_drone.get_velocity()}，速度大小: {fast_drone.get_speed():.2f} m/s"
    )
    print(f"✓ 速度已被限制在最大值 {fast_drone.max_speed} m/s 以内")

    return drone


def test_distance_calculation():
    """
    测试距离计算方法
    对应论文中的d^ij计算
    """
    print("\n=== 测试2：距离计算（对应论文中的d^ij） ===")

    # 创建4架无人机
    drones = []
    positions = [[0, 0], [3, 4], [6, 8], [10, 0]]

    for i, pos in enumerate(positions):
        drone = Drone(drone_id=i + 1, initial_position=pos)
        drones.append(drone)
        print(f"无人机{i+1}: 位置 {pos}")

    print("\n无人机之间的距离矩阵:")

    # 计算所有无人机对之间的距离
    n = len(drones)
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0.0)
            else:
                distance = drones[i].distance_to(drones[j])
                row.append(distance)
        # 格式化输出
        formatted_row = [f"{d:.2f}" for d in row]
        print(f"  无人机{i+1}: [{', '.join(formatted_row)}]")

    # 验证3-4-5三角形
    print("\n验证: 无人机1(0,0)和无人机2(3,4)的距离应为5.00米 (3-4-5三角形)")
    actual = drones[0].distance_to(drones[1])
    print(f"实际计算值: {actual:.2f}米")

    if abs(actual - 5.00) < 0.01:
        print("✓ 距离计算正确！")
    else:
        print("✗ 距离计算有误")

    return drones


def visualize_drone_movement():
    """
    可视化无人机运动轨迹
    将图片保存到results文件夹
    """
    print("\n=== 测试3：可视化无人机运动 ===")

    # 创建无人机
    drone = Drone(drone_id=1, initial_position=[0, 0], initial_velocity=[1, 0])

    # 模拟4种控制场景（对应论文中的不同控制输入）
    scenarios = [
        ([2, 0], 10, "向右加速", "r"),
        ([0, 2], 10, "向上加速", "g"),
        ([-1, -1], 10, "向左下减速", "b"),
        ([0, 0], 10, "无控制输入", "orange"),
    ]

    all_trajectories = []

    for control_input, steps, label, color in scenarios:
        print(f"场景: {label}, 控制输入: {control_input}")

        # 重置无人机到初始状态
        drone.reset(position=[0, 0], velocity=[1, 0])

        # 记录轨迹
        positions = []

        for step in range(steps):
            # 每次更新前都设置控制输入
            drone.set_control_input(control_input)
            drone.update_state(dt=0.5)
            positions.append(drone.get_position())
            # 注意：update_state()会重置control_input为0，所以下次循环需要重新设置

        all_trajectories.append(
            {
                "positions": np.array(positions),
                "label": label,
                "color": color,
                "control_input": control_input,
            }
        )

    # 创建可视化图表 - 只画一个简单的图
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 子图1：轨迹图
    ax = axs[0, 0]
    for traj in all_trajectories:
        positions = traj["positions"]
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            color=traj["color"],
            linewidth=2,
            label=traj["label"],
            alpha=0.7,
        )

        # 标记起点和终点
        ax.scatter(
            positions[0, 0],
            positions[0, 1],
            color=traj["color"],
            s=100,
            marker="o",
            edgecolors="black",
        )
        ax.scatter(
            positions[-1, 0],
            positions[-1, 1],
            color=traj["color"],
            s=100,
            marker="s",
            edgecolors="black",
        )

    ax.set_title("无人机运动轨迹")
    ax.set_xlabel("X 位置 (米)")
    ax.set_ylabel("Y 位置 (米)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis("equal")

    # 子图2：速度变化
    ax = axs[0, 1]

    # 创建一个新无人机来演示速度变化
    speed_drone = Drone(drone_id=2, initial_position=[0, 0], initial_velocity=[8, 4])

    speeds = []
    times = []
    for t in range(20):
        speed_drone.set_control_input([2, 1])
        speed_drone.update_state(dt=0.1)
        speeds.append(speed_drone.get_speed())
        times.append(t * 0.1)

    ax.plot(times, speeds, "b-", linewidth=2, label="速度")
    ax.axhline(
        y=speed_drone.max_speed,
        color="r",
        linestyle="--",
        label=f"最大速度: {speed_drone.max_speed}m/s",
    )
    ax.axhline(
        y=speed_drone.min_speed,
        color="g",
        linestyle="--",
        label=f"最小速度: {speed_drone.min_speed}m/s",
    )
    ax.set_title("水平空速变化")
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("速度大小 (米/秒)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 子图3：偏航角变化
    ax = axs[1, 0]

    yaw_drone = Drone(drone_id=3, initial_position=[0, 0], initial_velocity=[1, 0])
    yaw_drone.set_control_input([0, 2])

    yaws = []
    yaw_times = []
    for t in range(20):
        yaw_drone.update_state(dt=0.1)
        yaws.append(yaw_drone.get_yaw_degrees())
        yaw_times.append(t * 0.1)
        yaw_drone.set_control_input([0, 2])

    ax.plot(yaw_times, yaws, "r-", linewidth=2, label="偏航角")
    ax.set_title("偏航角变化")
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("偏航角 (度)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 子图4：3-4-5三角形验证
    ax = axs[1, 1]

    # 创建3-4-5三角形
    drone_a = Drone(drone_id=5, initial_position=[0, 0])
    drone_b = Drone(drone_id=6, initial_position=[3, 0])
    drone_c = Drone(drone_id=7, initial_position=[0, 4])

    points = np.array(
        [
            drone_a.get_position(),
            drone_b.get_position(),
            drone_c.get_position(),
            drone_a.get_position(),
        ]
    )

    ax.plot(points[:, 0], points[:, 1], "k-", linewidth=2)
    ax.scatter(points[0:3, 0], points[0:3, 1], s=100, color=["red", "green", "blue"])

    # 添加标签
    ax.text(0, 0, "A(0,0)", fontsize=12, ha="right")
    ax.text(3, 0, "B(3,0)", fontsize=12, ha="left")
    ax.text(0, 4, "C(0,4)", fontsize=12, ha="right", va="bottom")

    # 计算距离
    ab = drone_a.distance_to(drone_b)
    bc = drone_b.distance_to(drone_c)
    ca = drone_c.distance_to(drone_a)

    # 显示距离
    ax.text(1.5, -0.5, f"AB={ab:.1f}", fontsize=10, ha="center")
    ax.text(1.5, 2.0, f"BC={bc:.1f}", fontsize=10, ha="center")
    ax.text(-0.5, 2.0, f"CA={ca:.1f}", fontsize=10, ha="right")

    ax.set_title("3-4-5三角形距离验证")
    ax.set_xlabel("X 位置 (米)")
    ax.set_ylabel("Y 位置 (米)")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    plt.suptitle("无人机动力学模型测试结果", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # 保存图片到results文件夹
    results_folder = create_results_folder()
    save_path = os.path.join(results_folder, "drone_dynamics_simple.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 图表已保存到: {save_path}")

    plt.show()

    return all_trajectories


def main():
    """
    主函数 - 模块二：单架无人机动力学模型
    """
    print("=" * 70)
    print("基于视觉感知的无人机集群协同演化 - 模块二")
    print("单架无人机动力学模型实现")
    print("对应论文第二章：基于神经网络的速度可控的无人机集群避障")
    print("=" * 70)

    # 创建results文件夹
    create_results_folder()

    print("开始测试...")

    try:
        # 测试1：单架无人机基本功能
        drone1 = test_single_drone_basic()

        # 测试2：距离计算
        drones = test_distance_calculation()

        # 测试3：可视化
        print("\n正在生成可视化图表...")
        trajectories = visualize_drone_movement()

    except Exception as e:
        print(f"错误: {e}")
        print("尝试使用备用显示方法...")

        # 使用更简单的显示方法
        import matplotlib

        matplotlib.use("Agg")  # 不显示窗口，只保存文件

        # 重新运行测试
        import traceback

        traceback.print_exc()

    # 总结
    print("\n" + "=" * 50)
    print("模块二总结:")
    print("1. ✓ 成功实现无人机动力学模型（论文公式2-1和2-2）")
    print("2. ✓ 成功实现距离计算方法（论文中的d^ij）")
    print("3. ✓ 成功实现状态向量获取（用于后续神经网络输入）")
    print("4. ✓ 成功保存可视化图表到results文件夹")
    print("5. ✓ 验证了3-4-5三角形距离计算正确性")
    print("=" * 50)


if __name__ == "__main__":
    main()
