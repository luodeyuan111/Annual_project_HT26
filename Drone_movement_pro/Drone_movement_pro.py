#多无人机不同路径下的路径运行
import airsim
import json
import threading

# 1. 连接AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

# 2. 加载路径映射和个体路径
def load_path_mapping(mapping_file):#路径映射文件在此以“mapping_file”为例
    """加载路径映射文件"""
    with open(mapping_file, 'r') as f:
        return json.load(f) # 返回字典：{无人机名: 路径文件}

def load_individual_paths(mapping):
    """为每架无人机加载其路径数据"""
    paths = {}
    for drone, path_file in mapping.items():
        with open(path_file, 'r') as f:
            paths[drone] = json.load(f)# 存储为字典：drone -> 航点列表
    return paths
# 加载映射
path_mapping = load_path_mapping("path_mapping.json")
# 加载所有路径
individual_paths = load_individual_paths(path_mapping)

# 3. 初始化所有无人机
drones = list(path_mapping.keys())  # 从映射获取无人机列表
for drone in drones:
    client.enableApiControl(True, drone)
    client.armDisarm(True, drone)
    client.takeoffAsync(vehicle_name=drone).join()
    print(f"{drone} 初始化完成")

# 4. 多线程执行不同路径
def execute_path_for_drone(client, drone_name, waypoints, speed=3.0):
    """为单架无人机执行路径（线程安全）"""
    for i, wp in enumerate(waypoints):
        target = airsim.Vector3r(wp['x'], wp['y'], wp['z'])
        client.moveToPositionAsync(
            target.x_val, target.y_val, target.z_val, speed, vehicle_name=drone_name
        ).join()# 等待到达航点
    print(f"{drone_name} 路径完成")
# 为每架无人机创建独立线程
threads = []
for drone, path in individual_paths.items():
    thread = threading.Thread(target=execute_path_for_drone, args=(client, drone, path))
    thread.start()
    threads.append(thread)
# 等待所有线程完成
for thread in threads:
    thread.join()

# 5. 降落
for drone in drones:
    client.landAsync(vehicle_name=drone).join()
    client.armDisarm(False, drone)
