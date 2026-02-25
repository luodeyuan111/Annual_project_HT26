import airsim
import os
import time
from airsim import ImageType, VehicleCamera

def main():
    # 【需填充】设置深度图保存文件夹路径：默认值为 "depth_images"，可修改为任何有效路径
    # 例如：output_dir = "C:/实验数据/深度图" 或 output_dir = "./results"
    output_dir = "depth_images"  # 可修改路径，确保有写入权限
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出文件夹: {output_dir}")
    else:
        print(f"使用现有文件夹: {output_dir}")

    # 连接 AirSim 客户端（默认本地连接，如需远程连接可修改参数）
    # 【需注意】如果 AirSim 运行在远程服务器，需指定 IP 和端口，例如：
    # client = airsim.MultirotorClient("192.168.1.100")  # 替换为实际 IP
    client = airsim.MultirotorClient()  # 默认本地连接
    try:
        client.confirmConnection()
        print("成功连接 AirSim 仿真环境")
    except Exception as e:
        print(f"连接失败: {e}")
        return

    # 获取所有无人机名称（动态处理，无需修改）
    drones = client.listVehicles()
    if not drones:
        print("未检测到无人机，请检查 settings.json 配置")
        return
    print(f"检测到 {len(drones)} 架无人机: {drones}")

    # 【需填充】定义相机名称列表：必须与 settings.json 中 Camera 配置的键完全匹配
    # 如果添加了新相机（如 "top" 或 "bottom"），需在此扩展列表
    cameras = ["front", "right", "back", "left"]  # 可修改相机名称或数量

    # 循环处理每架无人机和每个相机
    for drone_name in drones:
        print(f"\n处理无人机: {drone_name}")
        for camera_name in cameras:
            try:
                # 【需注意】图像请求参数：ImageType.DepthPerspective 为深度图，可改为其他类型
                # 例如：ImageType.Scene 获取场景图，ImageType.Segmentation 获取分割图
                responses = client.simGetImages([
                    airsim.ImageRequest(camera_name, ImageType.DepthPerspective, pixels_as_float=False, compress=False)
                    # pixels_as_float=True 可获取浮点数据（需调整保存逻辑）
                ], vehicle_name=drone_name)
                
                if responses and len(responses) > 0 and responses[0].height > 0:
                    depth_image = responses[0]
                    # 【需填充】文件名格式：可根据需要修改，例如添加时间戳
                    # 例如：filename = f"{drone_name}_{camera_name}_{time.time()}.png"
                    filename = os.path.join(output_dir, f"{drone_name}_{camera_name}_depth.png")
                    airsim.write_file(filename, depth_image.image_data_uint8)
                    print(f"  保存 {camera_name} 相机深度图: {filename}")
                else:
                    print(f"  警告: {drone_name} 的 {camera_name} 相机未返回有效数据")
            except Exception as e:
                print(f"  错误: 获取 {drone_name} 的 {camera_name} 相机图像失败 - {str(e)}")

    print(f"\n深度图获取完成！所有图像保存至: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()
