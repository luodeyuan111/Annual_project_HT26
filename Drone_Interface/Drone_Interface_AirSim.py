"""AirSim Drone controller implementation.

Provides a simple `DroneController` wrapper around the AirSim Python
client with methods used by the integrator: `capture_frame`,
`takeoff`, `land`, `move_by_velocity`, `move_to_position`, `disconnect`.
"""
import time
import numpy as np
from PIL import Image

try:
    import airsim
except Exception:
    airsim = None


class DroneController:
    def __init__(self, ip="127.0.0.1", port=41451, camera_name="0"):
        if airsim is None:
            raise ImportError("airsim is required in this environment; pip install airsim")
        self.client = airsim.MultirotorClient(ip=ip, port=port)
        self.client.confirmConnection()
        try:
            self.client.enableApiControl(True)
        except Exception:
            pass
        try:
            self.client.armDisarm(True)
        except Exception:
            pass
        self.camera_name = camera_name
        self.frame_count = 0

    def capture_frame(self, image_type=airsim.ImageType.Scene, compress=True):
        req = airsim.ImageRequest(self.camera_name, image_type, False, compress)
        try:
            res = self.client.simGetImages([req])
        except Exception as e:
            raise RuntimeError(f"simGetImages failed: {e}")

        if not res or res[0] is None:
            raise RuntimeError("no image returned from AirSim")

        resp = res[0]
        # Diagnostic (console) to help debugging different server variants
        try:
            print("[AirSim] ImageResponse attrs:", [a for a in dir(resp) if not a.startswith('_')])
        except Exception:
            pass

        # Try compressed/uint8 first
        img_bytes = getattr(resp, 'image_data_uint8', None)
        if img_bytes:
            try:
                import cv2
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(img)
                    self.frame_count += 1
                    return pil
            except Exception:
                pass

        # Try image_data_float
        img_f = getattr(resp, 'image_data_float', None)
        h = getattr(resp, 'height', None)
        w = getattr(resp, 'width', None)
        if img_f is not None and h and w:
            try:
                arr = np.array(img_f, dtype=np.float32).reshape(h, w, -1)
                if arr.shape[2] == 1:
                    arr = np.repeat(arr, 3, axis=2)
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                pil = Image.fromarray(arr)
                self.frame_count += 1
                return pil
            except Exception:
                pass

        # Try raw image_data
        raw = getattr(resp, 'image_data', None)
        if raw is not None and h and w:
            try:
                arr = np.frombuffer(raw, dtype=np.uint8)
                if arr.size in (h * w * 3, h * w * 4):
                    try:
                        img = arr.reshape((h, w, 3))
                    except Exception:
                        img = arr.reshape((h, w, 4))[:, :, :3]
                    pil = Image.fromarray(img)
                    self.frame_count += 1
                    return pil
            except Exception:
                pass

        raise RuntimeError('Could not decode AirSim ImageResponse; check server variant')

    def takeoff(self):
        try:
            self.client.takeoffAsync().join()
            return True
        except Exception:
            return False

    def land(self):
        try:
            self.client.landAsync().join()
            return True
        except Exception:
            return False

    def move_by_velocity(self, vx=0.0, vy=0.0, vz=0.0, duration=1.0, drivetrain=None, yaw_mode=None):
        try:
            if drivetrain is None:
                drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom
            if yaw_mode is None:
                yaw_mode = airsim.YawMode(False, 0)
            self.client.moveByVelocityAsync(vx, vy, vz, duration, drivetrain=drivetrain, yaw_mode=yaw_mode).join()
            return True
        except Exception as e:
            raise RuntimeError(f"move_by_velocity failed: {e}")

    def move_to_position(self, x, y, z, velocity=2.0):
        try:
            self.client.moveToPositionAsync(x, y, z, velocity).join()
            return True
        except Exception as e:
            raise RuntimeError(f"move_to_position failed: {e}")

    def disconnect(self):
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except Exception:
            pass


__all__ = ['DroneController']
