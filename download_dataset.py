import glob
import math
import os
import re
from importlib import reload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import osu.dataset as dataset
import osu.downloader as downloader
import osu.preview.preview as preview
import osu.rulesets.beatmap as bm
import osu.rulesets.hitobjects as hitobjects
import osu.rulesets.replay as rp

# test user-based downloading using mrekk hehehehaw
user_id = 7562902

# download only S ranks from top 100 on each map only from mrekk tho
# (remove the by_user_id param to make it for everyone in top 100
# mapsets = [
#    "166062 Lime - Renai Syndrome(1)"
# ]
mapsets = downloader.get_all_mapset_folders_on_disk()

def filter_beatmap(beatmap: bm.Beatmap) -> bool:
    # include all beatmaps
    # if beatmap.ar_raw() <= 9:
    #     return False
    return True


# downloader.download_mapsets(mapsets, filter=filter_beatmap, max=100, only="ALL", verbose=False, by_user_id=user_id)

replays = downloader.download_user_scores(
    user_id=user_id, score_type="ALL", only="ALL", max=1000
)
