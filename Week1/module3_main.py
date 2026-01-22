"""
模块三：5架无人机集群框架主程序
对应论文第二章2.2节集群模型
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os
import sys
import time

# 设置中文字体，解决中文显示为方块的问题
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 添加src目录到Python路径
sys.path.append("src")

# 导入集群类
from src.drone_swarm import DroneSwarm


def create_results_folder():
    """创建results文件夹（如果不存在）"""
    if not os.path.exists("results"):
        os.makedirs("results")
        print("已创建 results 文件夹")
    return "results"


def test_swarm_initialization():
    """
    测试集群初始化
    """
    print("=== 测试1：集群初始化 ===")

    # 创建5架无人机的集群
    swarm = DroneSwarm(num_drones=5)

    print(f"创建了 {swarm.num_drones} 架无人机的集群")
    print(f"最大通信距离: {swarm.max_comm_distance} 米")
    print(f"期望距离: {swarm.desired_distance} 米")
    print(f"最小分离距离: {swarm.min_separation_distance} 米")

    # 显示每架无人机的初始状态
    print("\n无人机初始状态:")
    for i, drone in enumerate(swarm.drones):
        print(
            f"  无人机{drone.id}: 位置={drone.get_position()}, "
            f"速度={drone.get_velocity()}, 速度大小={drone.get_speed():.2f} m/s"
        )

    # 检查初始距离
    print("\n初始距离矩阵:")
    for i in range(5):
        distances = []
        for j in range(5):
            if i == j:
                distances.append(0.0)
            else:
                dist = swarm.drones[i].distance_to(swarm.drones[j])
                distances.append(dist)
        formatted_dists = [f"{d:.1f}" for d in distances]
        print(f"  无人机{i+1}: [{', '.join(formatted_dists)}]")

    return swarm


def test_neighbor_detection(swarm):
    """
    测试邻居检测功能
    """
    print("\n=== 测试2：邻居检测 ===")

    for i in range(1, swarm.num_drones + 1):
        neighbors = swarm.get_neighbors(i)
        print(f"无人机{i}的邻居: {neighbors}")

        # 验证邻居检测
        current_drone = swarm.drones[i - 1]
        neighbor_drones = [swarm.drones[n - 1] for n in neighbors]

        print(f"  验证: 所有邻居距离都应 ≤ {swarm.max_comm_distance} 米")
        for neighbor_id in neighbors:
            neighbor_drone = swarm.drones[neighbor_id - 1]
            distance = current_drone.distance_to(neighbor_drone)
            print(f"    到无人机{neighbor_id}: {distance:.1f} 米")

    return swarm


def test_flocking_forces(swarm):
    """
    测试集群力的计算
    """
    print("\n=== 测试3：集群力计算 ===")

    # 测试第一架无人机的力
    drone_id = 1
    neighbors = swarm.get_neighbors(drone_id)

    if neighbors:
        print(f"无人机{drone_id}的邻居: {neighbors}")

        # 计算各种力
        cohesion = swarm.calculate_cohesion(drone_id, neighbors)
        separation = swarm.calculate_separation(drone_id, neighbors)
        alignment = swarm.calculate_alignment(drone_id, neighbors)

        print(f"  聚集力: {cohesion}")
        print(f"  分离力: {separation}")
        print(f"  对齐力: {alignment}")
        print(f"  总力: {cohesion + separation + alignment}")
    else:
        print(f"无人机{drone_id}没有邻居")

    return swarm


def simulate_swarm_movement():
    """
    模拟集群运动
    """
    print("\n=== 测试4：集群运动模拟 ===")

    # 创建集群
    swarm = DroneSwarm(num_drones=5)

    # 模拟多个时间步
    num_steps = 50
    print(f"模拟 {num_steps} 个时间步，时间步长 0.1秒")

    # 记录一些统计信息
    avg_speeds = []
    avg_distances = []
    cohesion_measures = []

    for step in range(num_steps):
        swarm.update_swarm(dt=0.1)
        state = swarm.get_swarm_state()

        # 计算平均速度
        avg_speed = np.mean(state["speeds"])
        avg_speeds.append(avg_speed)

        # 计算平均距离
        positions = state["positions"]
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        avg_distance = np.mean(distances) if distances else 0
        avg_distances.append(avg_distance)

        # 计算聚集度（位置的标准差）
        if len(positions) > 0:
            position_std = np.std(positions, axis=0)
            cohesion_measure = np.mean(position_std)
            cohesion_measures.append(cohesion_measure)

        # 每10步打印一次进度
        if (step + 1) % 10 == 0:
            print(
                f"  步数 {step+1}: 平均速度={avg_speed:.2f} m/s, "
                f"平均距离={avg_distance:.2f} m"
            )

    # 最终状态
    print(f"\n最终状态:")
    state = swarm.get_swarm_state()
    print(f"  平均速度: {np.mean(state['speeds']):.2f} m/s")
    print(f"  速度标准差: {np.std(state['speeds']):.2f} m/s")

    # 计算最终平均距离
    positions = state["positions"]
    final_distances = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            final_distances.append(dist)

    if final_distances:
        print(f"  平均距离: {np.mean(final_distances):.2f} m")
        print(f"  最小距离: {np.min(final_distances):.2f} m")
        print(f"  最大距离: {np.max(final_distances):.2f} m")

    return swarm, avg_speeds, avg_distances, cohesion_measures


def visualize_swarm_2d(swarm, save_path=None):
    """
    2D可视化集群运动

    参数:
        swarm: 无人机集群对象
        save_path: 保存路径，如果为None则不保存
    """
    print("\n=== 2D集群可视化 ===")

    # 获取历史数据
    history = swarm.get_history()
    positions_history = history["positions"]

    if not positions_history:
        print("错误: 没有历史数据")
        return

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 子图1: 最终状态
    ax1 = axes[0, 0]
    final_positions = positions_history[-1]

    # 绘制无人机位置
    for i, pos in enumerate(final_positions):
        ax1.scatter(pos[0], pos[1], s=200, label=f"无人机 {i+1}", alpha=0.7)

    # 绘制通信范围
    for i, pos in enumerate(final_positions):
        circle = Circle(
            pos,
            swarm.max_comm_distance,
            fill=False,
            color="gray",
            alpha=0.3,
            linestyle="--",
        )
        ax1.add_patch(circle)

    # 绘制连接线（邻居）
    for i in range(len(final_positions)):
        for j in range(i + 1, len(final_positions)):
            dist = np.linalg.norm(final_positions[i] - final_positions[j])
            if dist <= swarm.max_comm_distance:
                x_vals = [final_positions[i][0], final_positions[j][0]]
                y_vals = [final_positions[i][1], final_positions[j][1]]
                ax1.plot(x_vals, y_vals, "gray", alpha=0.5, linewidth=0.5)

    ax1.set_title("最终状态")
    ax1.set_xlabel("X (米)")
    ax1.set_ylabel("Y (米)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis("equal")

    # 子图2: 初始状态
    ax2 = axes[0, 1]
    if positions_history:
        initial_positions = positions_history[0]

        for i, pos in enumerate(initial_positions):
            ax2.scatter(pos[0], pos[1], s=200, label=f"无人机 {i+1}", alpha=0.7)

        ax2.set_title("初始状态")
        ax2.set_xlabel("X (米)")
        ax2.set_ylabel("Y (米)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.axis("equal")

    # 子图3: 运动轨迹
    ax3 = axes[0, 2]
    colors = ["red", "green", "blue", "orange", "purple"]

    for i in range(len(positions_history[0])):
        # 收集第i架无人机的所有位置
        drone_trajectory = [pos[i] for pos in positions_history]
        drone_trajectory = np.array(drone_trajectory)

        ax3.plot(
            drone_trajectory[:, 0],
            drone_trajectory[:, 1],
            color=colors[i % len(colors)],
            linewidth=2,
            alpha=0.7,
            label=f"无人机 {i+1}",
        )

        # 标记起点和终点
        ax3.scatter(
            drone_trajectory[0, 0],
            drone_trajectory[0, 1],
            color=colors[i % len(colors)],
            s=100,
            marker="o",
        )
        ax3.scatter(
            drone_trajectory[-1, 0],
            drone_trajectory[-1, 1],
            color=colors[i % len(colors)],
            s=100,
            marker="s",
        )

    ax3.set_title("运动轨迹")
    ax3.set_xlabel("X (米)")
    ax3.set_ylabel("Y (米)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")
    ax3.axis("equal")

    # 子图4: 平均速度变化
    ax4 = axes[1, 0]
    if len(positions_history) > 1:
        # 计算每步的平均速度
        avg_speeds = []
        for step in range(len(positions_history)):
            if step < len(history["velocities"]):
                velocities = history["velocities"][step]
                speeds = np.linalg.norm(velocities, axis=1)
                avg_speeds.append(np.mean(speeds))

        ax4.plot(range(len(avg_speeds)), avg_speeds, "b-", linewidth=2)
        ax4.set_title("平均速度变化")
        ax4.set_xlabel("时间步")
        ax4.set_ylabel("平均速度 (m/s)")
        ax4.grid(True, alpha=0.3)

    # 子图5: 距离变化
    ax5 = axes[1, 1]
    if len(positions_history) > 1:
        # 计算每步的平均距离
        avg_distances = []
        for step in range(len(positions_history)):
            positions = positions_history[step]
            distances = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append(dist)
            if distances:
                avg_distances.append(np.mean(distances))

        if avg_distances:
            ax5.plot(range(len(avg_distances)), avg_distances, "g-", linewidth=2)
            # 添加期望距离参考线
            ax5.axhline(
                y=swarm.desired_distance,
                color="r",
                linestyle="--",
                label=f"期望距离: {swarm.desired_distance}m",
            )
            ax5.axhline(
                y=swarm.min_separation_distance,
                color="orange",
                linestyle="--",
                label=f"最小距离: {swarm.min_separation_distance}m",
            )

            ax5.set_title("平均距离变化")
            ax5.set_xlabel("时间步")
            ax5.set_ylabel("平均距离 (m)")
            ax5.grid(True, alpha=0.3)
            ax5.legend()

    # 子图6: 集群参数
    ax6 = axes[1, 2]
    ax6.axis("off")  # 关闭坐标轴

    # 显示集群参数信息
    params_text = (
        f"集群参数:\n"
        f"无人机数量: {swarm.num_drones}\n"
        f"最大通信距离: {swarm.max_comm_distance} m\n"
        f"期望距离: {swarm.desired_distance} m\n"
        f"最小分离距离: {swarm.min_separation_distance} m\n"
        f"聚集强度: {swarm.cohesion_strength:.2f}\n"
        f"分离强度: {swarm.separation_strength:.2f}\n"
        f"对齐强度: {swarm.alignment_strength:.2f}\n"
        f"模拟步数: {len(positions_history)}\n"
        f"时间步长: 0.1 s"
    )

    ax6.text(
        0.1,
        0.9,
        params_text,
        fontsize=10,
        verticalalignment="top",
        transform=ax6.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax6.set_title("集群参数")

    plt.suptitle("无人机集群仿真结果", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图表已保存到: {save_path}")

    plt.show()

    return fig


def create_animation(swarm, save_path=None):
    """
    创建集群运动的动画

    参数:
        swarm: 无人机集群对象
        save_path: 保存路径，如果为None则不保存
    """
    print("\n=== 创建集群动画 ===")

    history = swarm.get_history()
    positions_history = history["positions"]

    if not positions_history:
        print("错误: 没有历史数据")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # 设置坐标轴范围
    all_positions = np.vstack(positions_history)
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()

    # 添加一些边距
    margin = max((x_max - x_min), (y_max - y_min)) * 0.1
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X (米)")
    ax.set_ylabel("Y (米)")
    ax.set_title("无人机集群运动动画")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # 颜色
    colors = ["red", "green", "blue", "orange", "purple"]

    # 创建初始散点图
    scat = ax.scatter([], [], s=100, alpha=0.7)

    # 存储通信范围圆
    circles = []
    for i in range(swarm.num_drones):
        circle = Circle(
            (0, 0),
            swarm.max_comm_distance,
            fill=False,
            color="gray",
            alpha=0.3,
            linestyle="--",
        )
        ax.add_patch(circle)
        circles.append(circle)

    # 存储连接线
    lines = []
    for i in range(swarm.num_drones):
        for j in range(i + 1, swarm.num_drones):
            (line,) = ax.plot([], [], "gray", alpha=0.5, linewidth=0.5)
            lines.append((line, i, j))

    # 添加文本显示时间步
    time_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, fontsize=12, verticalalignment="top"
    )

    def init():
        """初始化动画"""
        scat.set_offsets(np.empty((0, 2)))
        for circle in circles:
            circle.center = (0, 0)
        for line, _, _ in lines:
            line.set_data([], [])
        time_text.set_text("")
        return [scat] + circles + [line for line, _, _ in lines] + [time_text]

    def update(frame):
        """更新动画帧"""
        positions = positions_history[frame]

        # 更新散点位置
        scat.set_offsets(positions)

        # 更新颜色
        scat.set_color(colors[: len(positions)])

        # 更新通信范围圆
        for i, circle in enumerate(circles):
            if i < len(positions):
                circle.center = positions[i]

        # 更新连接线
        line_index = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if line_index < len(lines):
                    line, line_i, line_j = lines[line_index]
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist <= swarm.max_comm_distance:
                        line.set_data(
                            [positions[i][0], positions[j][0]],
                            [positions[i][1], positions[j][1]],
                        )
                        line.set_alpha(0.5)
                    else:
                        line.set_data([], [])
                    line_index += 1

        # 更新时间文本
        time_text.set_text(f"时间步: {frame}")

        return [scat] + circles + [line for line, _, _ in lines] + [time_text]

    # 创建动画
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(positions_history),
        init_func=init,
        blit=True,
        interval=100,
    )

    if save_path:
        print(f"正在保存动画到 {save_path}...")
        try:
            # 保存为GIF（需要安装pillow）
            ani.save(save_path, writer="pillow", fps=10)
            print(f"动画已保存到: {save_path}")
        except Exception as e:
            print(f"保存动画时出错: {e}")
            print("尝试保存为MP4...")
            try:
                ani.save(save_path.replace(".gif", ".mp4"), writer="ffmpeg", fps=10)
                print(f"动画已保存为MP4格式")
            except Exception as e2:
                print(f"保存MP4时也出错: {e2}")

    plt.show()

    return ani


def main():
    """
    主函数 - 模块三：5架无人机集群框架
    """
    print("=" * 70)
    print("基于视觉感知的无人机集群协同演化 - 模块三")
    print("5架无人机集群框架实现")
    print("对应论文第二章2.2节：集群模型")
    print("=" * 70)

    # 创建results文件夹
    results_folder = create_results_folder()

    # 测试1：集群初始化
    print("\n开始测试集群初始化...")
    swarm = test_swarm_initialization()

    # 测试2：邻居检测
    print("\n开始测试邻居检测...")
    swarm = test_neighbor_detection(swarm)

    # 测试3：集群力计算
    print("\n开始测试集群力计算...")
    swarm = test_flocking_forces(swarm)

    # 测试4：集群运动模拟
    print("\n开始模拟集群运动...")
    swarm, avg_speeds, avg_distances, cohesion_measures = simulate_swarm_movement()

    # 2D可视化
    print("\n生成2D可视化图表...")
    save_path_2d = os.path.join(results_folder, "module3_swarm_2d.png")
    fig = visualize_swarm_2d(swarm, save_path_2d)

    # 创建动画
    print("\n创建集群运动动画...")
    save_path_animation = os.path.join(results_folder, "module3_swarm_animation.gif")
    ani = create_animation(swarm, save_path_animation)

    # 总结
    print("\n" + "=" * 70)
    print("模块三总结:")
    print("1. ✓ 成功实现5架无人机集群初始化")
    print("2. ✓ 成功实现邻居检测功能")
    print("3. ✓ 成功实现聚集、分离、对齐规则")
    print("4. ✓ 成功模拟集群运动")
    print("5. ✓ 成功生成可视化图表和动画")
    print(f"6. ✓ 结果已保存到 {results_folder} 文件夹")
    print("\n下一模块（模块四）：集群控制器实现")
    print("将实现论文中的4个控制组件和参数优化")
    print("=" * 70)


if __name__ == "__main__":
    main()
