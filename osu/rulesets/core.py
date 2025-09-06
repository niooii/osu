SCREEN_WIDTH = 512
SCREEN_HEIGHT = 384
REPLAY_SAMPLING_RATE = 12

#CIRCLE_FADEOUT = 50
#SLIDER_FADEOUT = 100

import os

APP_DATA = os.getenv("APPDATA")
OSU_PATH = f'{APP_DATA}/../Local/osu!'
print(f'osu! path: {OSU_PATH}')


def osu_to_screen_pixel(x, y) -> (int, int):
    import pyautogui
    w, h = pyautogui.size()

    # screen/osu ratio
    # https://osu.ppy.sh/wiki/en/Client/Playfield
    osr_x = w / 640 * 0.75  # for some reason this constant just works. misleading wiki?
    osr_y = h / 480

    x = x * osr_x
    y = y * osr_y

    # osu width and height
    ow = SCREEN_WIDTH * osr_x
    oh = SCREEN_HEIGHT * osr_y

    offset_x = 0.5 * (w - ow)
    # "The playfield is slightly shifted vertically, placed 8 game pixels lower than the window's centre."
    offset_y = 0.5 * (h - oh) + (8 * osr_y)

    return int(x + offset_x), int(y + offset_y)