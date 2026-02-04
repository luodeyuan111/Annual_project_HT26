import airsim
import json

# 1. 连接AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

# 2. 加载路径
def load_path(file_path):#file_path即文件路径，文件名以"path.json"为例，文件格式为json格式
    with open(file_path, 'r') as f:
        return json.load(f)

waypoints = load_path("path.json")  # 航点格式: [{"x": 10, "y": 20, "z": -5}, ...]

# 3. 初始化所有无人机
drones = ["Drone1", "Drone2", "Drone3", "Drone4"]#以四架无人机为例
for drone in drones:
    client.enableApiControl(True, drone)
    client.armDisarm(True, drone)
    client.takeoffAsync(vehicle_name=drone).join()

# 4. 执行路径跟踪
for drone in drones:
    for i, wp in enumerate(waypoints):
        target = airsim.Vector3r(wp['x'], wp['y'], wp['z'])
        client.moveToPositionAsync(
            target.x_val, target.y_val, target.z_val,
            speed=3.0, vehicle_name=drone
        ).join()
    print(f"{drone} 路径执行完毕！")

# 5. 降落并清理
for drone in drones:
    client.landAsync(vehicle_name=drone).join()
    client.armDisarm(False, drone)

