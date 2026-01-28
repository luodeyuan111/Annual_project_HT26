import airsim
import numpy as np
import cv2
import os
import time

from Drone_Interface.rgb_data_extractor_utf8 import RGBDataExtractor


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

            # 显示图像 (可选)
            cv2.imshow(f"RGB Image from {cam_name}", img)
            cv2.waitKey(1)  # 暂停一毫秒，确保图像显示

        cv2.waitKey(0)  # 等待用户按下按键
        cv2.destroyAllWindows() # 关闭所有窗口

    finally:
        # 断开连接
        extractor.disconnect()

if __name__ == "__main__":
    main()
