import lzma
import time
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as mpatches

from enum import IntFlag

from . import beatmap as osu_beatmap
from ._util.bsearch import bsearch
from ._util.binfile import *

from functools import reduce

class Mod(IntFlag):
    DT = 0x40
    HR = 0x10
    HT = 0x100
    EZ = 0x2

class Replay:
    def __init__(self, file):
        self.game_mode = read_byte(file)

        assert self.game_mode == 0, "Not a osu!std replay"

        self.osu_version = read_int(file)
        self.map_md5 = read_binary_string(file)
        
        self.player = read_binary_string(file)

        self.replay_md5 = read_binary_string(file)

        self.n_300s = read_short(file)
        self.n_100s = read_short(file)
        self.n_50s = read_short(file)
        self.n_geki = read_short(file)
        self.n_katu = read_short(file)
        self.n_misses = read_short(file)

        self.score = read_int(file)
        self.max_combo = read_short(file)
        self.perfect = read_byte(file)

        total = self.n_300s + self.n_100s + self.n_50s + self.n_misses
        self.accuracy = (self.n_300s + self.n_100s / 3 + self.n_50s / 6) / total

        self.mods = read_int(file)

        life_graph = read_binary_string(file)
        self.life_graph = [t.split('|') for t in life_graph.split(',')[:-1]]

        self.timestamp = read_long(file)
        
        replay_length = read_int(file)
        replay_data = lzma.decompress(file.read(replay_length)).decode('utf8')

        data = [t.split("|") for t in replay_data.split(',')[:-1]]
        data = [(int(w), float(x), float(y), int(z)) for w, x, y, z in data]

        self.data = []
        offset = 0
        for w, x, y, z in data:
            offset += w
            self.data.append((offset, x, y, z))

        self.data = list(sorted(self.data))

        _ = read_long(file)

    def has_mods(self, *mods):
        mask = reduce(lambda x, y: x|y, mods)
        return bool(self.mods & mask)

    def frame(self, time):
        index = bsearch(self.data, time, lambda f: f[0])
        
        offset, _, _, _ = self.data[index]
        if offset > time:
            if index > 0:
                return self.data[index - 1][1:]
            else:
                return (0, 0, 0)
        elif index >= len(self.data):
            index = -1

        return self.data[index][1:]

def load(filename):
    with open(filename, "rb") as file:
        replay = Replay(file)
        return replay