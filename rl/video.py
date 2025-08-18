from typing import Tuple
import numpy as np
import dxcam
import cv2

# downscaling the frame and post processing
def process_frame(frame: np.ndarray, downscale_factor: float = 0.1) -> np.ndarray:
    height, width = frame.shape[:2]
    new_height = int(height * downscale_factor)
    new_width = int(width * downscale_factor)   

    downscaled = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # shift colors
    downscaled = cv2.cvtColor(downscaled, cv2.COLOR_BGR2RGB)

    return downscaled

class GameCapture:
    def __init__(self, target_fps=120, device_idx=0, output_idx=0):
        self.target_fps = target_fps
        self.device_idx = device_idx
        self.output_idx = output_idx
        self.camera = dxcam.create(device_idx=self.device_idx, output_idx=self.output_idx)

    def start(self, region: None | Tuple[int, int, int, int] = None):
        if region is None:
            self.camera.start(target_fps=self.target_fps, video_mode=True)
        else:
            self.camera.start(region=region, target_fps=self.target_fps, video_mode=True)

    def get_frame(self):
        frame = self.camera.get_latest_frame()
        return process_frame(frame, downscale_factor=0.2)

    def stop(self):
        self.camera.stop()
