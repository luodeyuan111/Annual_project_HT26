#RGB图像提取功能
import airsim
import numpy as np
import cv2
import os
import time

class RGBDataExtractor:
    def __init__(self, drone_name="Drone1"):
        """初始化RGB数据提取器"""
        self.drone_name = drone_name
        self.client = airsim.MultirotorClient()  # 连接AirSim
        self.client.confirmConnection()  # 确认连接
        self.client.enableApiControl(True, vehicle_name=drone_name)  # 启用API控制
        self.client.armDisarm(True, vehicle_name=drone_name)  # 解锁无人机

        # 相机名称列表（需与settings.json中的配置一致）
        self.camera_names = ["front_camera", "right_camera", "back_camera", "left_camera"]

        # 创建数据保存目录（例如：sensor_data/Drone1/）
        self.data_dir = os.path.join("sensor_data", drone_name)
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"数据保存目录：{self.data_dir}")

    def capture_rgb_images(self, timestamp):
        """从所有相机捕获RGB图像"""
        rgb_data = {}  # 存储所有相机的RGB数据

        for cam_name in self.camera_names:
            try:
                # 发送图像请求（仅RGB，不压缩）
                responses = self.client.simGetImages([
                    airsim.ImageRequest(
                        cam_name,          # 相机名称
                        airsim.ImageType.Scene,  # 图像类型：Scene=RGB
                        pixels_as_float=False,  # 不返回浮点数（返回字节流）
                        compress=False         # 不压缩（便于后续处理）
                    )
                ], vehicle_name=self.drone_name)

                # 检查响应是否有效
                if not responses or len(responses) == 0:
                    print(f"警告：相机{cam_name}未返回数据")
                    continue

                # 处理RGB图像
                response = responses[0]
                img_rgb = self._process_rgb_image(response)

                # 保存图像到本地
                self._save_rgb_image(img_rgb, cam_name, timestamp)

                # 存储数据（可选，供实时处理）
                rgb_data[cam_name] = img_rgb

                print(f"成功捕获相机{cam_name}的RGB图像")
            except Exception as e:
                print(f"错误：捕获相机{cam_name}数据失败 - {str(e)}")

        return rgb_data

    def _process_rgb_image(self, response):
        """将AirSim返回的字节流转换为OpenCV格式的RGB图像"""
        # 将字节流转换为numpy数组
        img_bytes = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        # 重塑数组为图像形状（高度×宽度×3通道）
        img_rgb = img_bytes.reshape(response.height, response.width, 3)
        # 转换BGR为RGB（OpenCV默认读取为BGR格式）
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        return img_rgb

    def _save_rgb_image(self, img_rgb, camera_name, timestamp):
        """保存RGB图像到本地"""
        # 生成文件名（例如：front_camera_rgb_1695000000.png）
        filename = f"{camera_name}_rgb_{timestamp}.png"
        filepath = os.path.join(self.data_dir, filename)
        # 保存图像（使用OpenCV的imwrite）
        cv2.imwrite(filepath, img_rgb)
        print(f"保存图像：{filepath}")

    def disconnect(self):
        """断开与AirSim的连接"""
        self.client.armDisarm(False, vehicle_name=self.drone_name)
        self.client.enableApiControl(False, vehicle_name=self.drone_name)
        print("已断开与AirSim的连接")
    

def main():
    # 初始化RGB数据提取器（无人机名称为Drone1）
    extractor = RGBDataExtractor(drone_name="Drone1")

    try:
        # 测试：捕获一次RGB图像
        timestamp = int(time.time() * 1000)  # 生成时间戳（毫秒级）
        print(f"开始捕获RGB图像（时间戳：{timestamp}）...")
        rgb_data = extractor.capture_rgb_images(timestamp)

        # 打印数据信息
        print("\n捕获结果：")
        for cam_name, img in rgb_data.items():
            print(f"相机{cam_name}：图像形状{img.shape}（高度×宽度×通道）")

    finally:
        # 断开连接
        extractor.disconnect()

if __name__ == "__main__":
    main()
#RGBDataExtractor类：封装了RGB数据提取的核心功能，包括连接AirSim、捕获图像、处理和保存数据。

#capture_rgb_images方法：遍历所有相机，发送图像请求，处理并返回RGB数据。

#process_rgb_image方法：将AirSim返回的字节流转换为OpenCV可处理的RGB格式（解决BGR转RGB的问题）。

#_save_rgb_image方法：将RGB图像保存为PNG文件（文件名包含相机名称和时间戳，便于后续检索）。