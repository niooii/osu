import numpy as np
import cv2
import mouse
from video import GameCapture
from osu import on_frame

mouse.move(-800, 100)
mouse.click()

game_capture = GameCapture(target_fps=120, device_idx=0, output_idx=1)

game_capture.start()

try:
    while True:
        image = game_capture.get_frame()
        on_frame(image)
finally:
    game_capture.stop()