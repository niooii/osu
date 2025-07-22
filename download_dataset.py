import os
import re
import math
import glob
from importlib import reload

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import osu.rulesets.beatmap as bm
import osu.rulesets.replay as rp
import osu.rulesets.hitobjects as hitobjects
import osu.dataset as dataset

import osu.preview.preview as preview

import osu.downloader as downloader

#
# download only S ranks from top 50 on each map
# mapsets = [
#     '1263383 Niji no Conquistador - Zutto Summer de Koishiteru [no video]',
#     '977552 umu - humanly [no video]',
#     '848400 Haywyre - Insight',
#     '675779 Camellia - NUCLEAR-STAR',
#     '936698 sana - Packet Hero',
#     '889855 GALNERYUS - RAISE MY SWORD',
#     '651507 a_hisa - Logical Stimulus',
#     '914242 Kano - Walk This Way!',
#     '463548 Nightcore - Flower Dance',
#     '685822 That Poppy - Altar',
#     '890438 Ata - Euphoria',
#     '431135 Imagine Dragons - Warriors',
#     '1125370 Sheena Ringo - Marunouchi Sadistic (neetskills remix)',
#     '327557 nameless - Milk Crown on Sonnetica',
#     '478405 Omoi - Snow Drive(0123) [no video]'
# ]

mapsets = downloader.get_all_mapset_folders_on_disk()


def filter_beatmap(beatmap: bm.Beatmap) -> bool:
    if beatmap.ar_raw() <= 9:
        return False

    return True


downloader.download_mapsets(mapsets, filter=filter_beatmap, max=25, only='S')
