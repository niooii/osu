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
    EZ = 0x1

class Replay:
    def __init__(self, file):
        self.game_mode = read_byte(file)

        # Sem minigames
        assert self.game_mode == 0, "Not a osu!std replay"

        # Versão do osu! e hash do mapa. A gente ignora.
        self.osu_version = read_int(file)
        self.map_md5 = read_binary_string(file)
        
        # Nome do jogador.
        self.player = read_binary_string(file)

        # Hash do replay. Dava pra usar, mas é meio desnecessário.
        self.replay_md5 = read_binary_string(file)

        # Acertos
        self.n_300s = read_short(file)
        self.n_100s = read_short(file)
        self.n_50s = read_short(file)
        self.n_geki = read_short(file)
        self.n_katu = read_short(file)
        self.n_misses = read_short(file)

        # Score e combo
        self.score = read_int(file)
        self.max_combo = read_short(file)
        self.perfect = read_byte(file)

        # Acurácia
        total = self.n_300s + self.n_100s + self.n_50s + self.n_misses
        self.accuracy = (self.n_300s + self.n_100s / 3 + self.n_50s / 6) / total

        # Mods (ignora)
        self.mods = Mod(read_int(file))

        # Gráfico de vida. Vide site para o formato.
        life_graph = read_binary_string(file)
        self.life_graph = [t.split('|') for t in life_graph.split(',')[:-1]]

        # Timestamp do replay (ignora)
        self.timestamp = read_long(file)
        
        # Informações do replay em si
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

        # Não usado
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

# Try to import Rust acceleration once at module level
try:
    import osu_fast
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

def _convert_rust_replay(rust_data):
    """Convert Rust replay data to Python Replay object"""
    # Create a new Replay object and populate it with Rust data
    class FastReplay:
        def __init__(self, data):
            # Set attributes directly from structured data
            self.game_mode = data.get('game_mode', 0)
            self.osu_version = data.get('osu_version', 0)
            self.map_md5 = data.get('map_md5', '')
            self.player = data.get('player', '')
            self.replay_md5 = data.get('replay_md5', '')
            
            # Score data
            self.n_300s = data.get('n_300s', 0)
            self.n_100s = data.get('n_100s', 0)
            self.n_50s = data.get('n_50s', 0)
            self.n_geki = data.get('n_geki', 0)
            self.n_katu = data.get('n_katu', 0)
            self.n_misses = data.get('n_misses', 0)
            self.score = data.get('score', 0)
            self.max_combo = data.get('max_combo', 0)
            self.perfect = data.get('perfect', False)
            
            # Calculate accuracy like in Python constructor
            total = self.n_300s + self.n_100s + self.n_50s + self.n_misses
            if total > 0:
                self.accuracy = (self.n_300s + self.n_100s / 3 + self.n_50s / 6) / total
            else:
                self.accuracy = 0.0
            
            # Convert mods to Mod enum
            self.mods = Mod(data.get('mods_int', 0))
            
            # Life graph and timestamp
            self.life_graph = data.get('life_graph', '').split(',')[:-1] if data.get('life_graph') else []
            if self.life_graph:
                self.life_graph = [t.split('|') for t in self.life_graph]
            self.timestamp = data.get('timestamp', 0)
            
            # Frame data - already in the correct format
            self.data = data.get('data', [])
            # Ensure data is sorted by time
            self.data = list(sorted(self.data))
            
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
    
    return FastReplay(rust_data)

def load(filename):
    # Try to use Rust acceleration if available
    if _RUST_AVAILABLE:
        try:
            rust_replay = osu_fast.parse_replay_fast(filename)
            # Convert Rust result to Python Replay object
            replay = _convert_rust_replay(rust_replay)
            replay._file_path = filename
            return replay
        except Exception:
            # Fall back to Python implementation on any error
            pass

    with open(filename, "rb") as file:
        replay = Replay(file)
        replay._file_path = filename  # Store file path for Rust acceleration
        return replay