import math
import os
import pickle
import re
from glob import escape as glob_escape, glob

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import tqdm

import osu.rulesets.beatmap as osu_beatmap
import osu.rulesets.core as osu_core
import osu.rulesets.hitobjects as hitobjects
import osu.rulesets.replay as osu_replay

# Constants
BATCH_LENGTH    = 2048
FRAME_RATE      = 24

# Feature index
INPUT_FEATURES  = ['x', 'y', 'visible', 'is_slider', 'is_spinner']
OUTPUT_FEATURES = ['x', 'y']

# Default beatmap frame information
_DEFAULT_BEATMAP_FRAME = (
    osu_core.SCREEN_WIDTH / 2, osu_core.SCREEN_HEIGHT / 2, # x, y
    float("inf"), False, False # time_left, is_slider, is_spinner
)

def pt_pad_sequences(data, maxlen, value=0):
    if not data:
        return np.array([])

    # Convert to list of tensors
    tensor_list = [torch.tensor(seq, dtype=torch.float) for seq in data]

    # Pad sequences
    padded = pad_sequence(tensor_list, batch_first=True, padding_value=value)

    # Trim or pad to exact maxlen
    if padded.size(1) > maxlen:
        padded = padded[:, :maxlen]
    elif padded.size(1) < maxlen:
        padding_tuple = (0, maxlen - padded.size(1))
        padded = torch.nn.functional.pad(padded, padding_tuple, value=value)

    return padded.numpy()

def all_files(osu_path, limit=0, verbose=False):
    """Return a pandas DataFrame mapping replay files to beatmap files"""

    replays = _list_all_replays(osu_path)
    if limit > 0:
        replays = replays[:limit]

    beatmaps = []
    for i in tqdm.tqdm(range(len(replays)-1, -1, -1), disable=not verbose):

        beatmap = _get_replay_beatmap_file(osu_path, replays[i])

        if beatmap is None:
            replays.pop(i)
        else:
            beatmaps.insert(0, beatmap)

    global _beatmap_cache
    with open('../.data/beatmap_cache.dat', 'wb') as f:
        pickle.dump(_beatmap_cache, f)

    if verbose:
        print()
        print()

    files = list(zip(replays, beatmaps))
    return pd.DataFrame(files, columns=['replay', 'beatmap'])


def load(files, verbose=0) -> pd.DataFrame:
    """Map the replay and beatmap files into osu! ruleset objects"""

    replays = []
    beatmaps = []

    for index, row in tqdm.tqdm(files.iterrows(), disable=not verbose):
        try:
            replay = osu_replay.load(row['replay'])
            # assert not replay.has_mods(osu_replay.Mod.DT, osu_replay.Mod.HR),\
            #         "DT and HR are not supported yet"
            beatmap = osu_beatmap.load(row['beatmap'])

        except Exception as e:
            if verbose:
                print()
                print("\tFailed:", e)
            continue

        replays.append(replay)
        beatmaps.append(beatmap)

    return pd.DataFrame(list(zip(replays, beatmaps)), columns=['replay', 'beatmap'])

_memo = {}

def get_beatmap_time_data(beatmap) -> []:
    if beatmap in _memo:
        return _memo[beatmap]

    if len(beatmap.hit_objects) == 0:
        return None

    _memo[beatmap] = []
    data = []
    chunk = []
    preempt, _ = beatmap.approach_rate()
    last_ok_frame = None  # Last frame with at least one visible object

    for time in range(beatmap.start_offset(), beatmap.length(), FRAME_RATE):
        frame = _beatmap_frame(beatmap, time)

        if frame is None:
            if last_ok_frame is None:
                frame = _DEFAULT_BEATMAP_FRAME
            else:
                frame = list(last_ok_frame)
                frame[2] = float("inf")
        else:
            last_ok_frame = frame

        px, py, time_left, is_slider, is_spinner = frame

        chunk.append(np.array([
            px - 0.5,
            py - 0.5,
            time_left < preempt,
            is_slider,
            is_spinner
        ]))

        if len(chunk) == BATCH_LENGTH:
            data.append(chunk)
            _memo[beatmap].append(chunk)
            chunk = []

    if len(chunk) > 0:
        data.append(chunk)
        _memo[beatmap].append(chunk)

    return data

def input_data(dataset, verbose=False) -> pd.DataFrame:
    """Given a osu! ruleset dataset for replays and maps, generate a
    new DataFrame with beatmap object information across time."""

    data = []
    global _memo

    if isinstance(dataset, osu_beatmap.Beatmap):
        beatmaps_list = [(dataset,)]
        dataset = pd.DataFrame(beatmaps_list, columns=['beatmap'])

    beatmaps = dataset['beatmap']

    for index, beatmap in tqdm.tqdm(beatmaps.items(), disable=not verbose):
        chunks = get_beatmap_time_data(beatmap)
        if chunks is not None:
            data.extend(chunks)

    data = pt_pad_sequences(data, maxlen=BATCH_LENGTH, value=0)

    index = pd.MultiIndex.from_product([
        range(len(data)), range(BATCH_LENGTH)
        ], names=['chunk', 'frame'])

    data = np.reshape(data, (-1, len(INPUT_FEATURES)))
    return pd.DataFrame(data, index=index, columns=INPUT_FEATURES, dtype=np.float32)

# extract the data for a single replay
def replay_to_output_data(beatmap: osu_beatmap.Beatmap, replay: osu_replay.Replay):
    target_data = []

    chunk = []
    for time in range(beatmap.start_offset(), beatmap.length(), FRAME_RATE):
        x, y = _replay_frame(beatmap, replay, time)

        chunk.append(np.array([x - 0.5, y - 0.5]))

        if len(chunk) == BATCH_LENGTH:
            target_data.append(chunk)
            chunk = []

    if len(chunk) > 0:
        target_data.append(chunk)

    return target_data

def target_data(dataset: pd.DataFrame, verbose=False):
    """Given a osu! ruleset dataset for replays and maps, generate a
    new DataFrame with replay cursor position across time."""

    target_data = []

    for index, row in tqdm.tqdm(dataset.iterrows(), disable=not verbose):
        replay = row['replay']
        beatmap = row['beatmap']

        if len(beatmap.hit_objects) == 0:
            continue

        chunk = []

        for time in range(beatmap.start_offset(), beatmap.length(), FRAME_RATE):
            x, y = _replay_frame(beatmap, replay, time)

            chunk.append(np.array([x - 0.5, y - 0.5]))

            if len(chunk) == BATCH_LENGTH:
                target_data.append(chunk)
                chunk = []

        if len(chunk) > 0:
            target_data.append(chunk)

    data = pt_pad_sequences(target_data, maxlen=BATCH_LENGTH, value=0)
    index = pd.MultiIndex.from_product([range(len(data)), range(BATCH_LENGTH)], names=['chunk', 'frame'])
    return pd.DataFrame(np.reshape(data, (-1, len(OUTPUT_FEATURES))), index=index, columns=OUTPUT_FEATURES, dtype=np.float32)


def _list_all_replays(osu_path):
    # Returns the full list of *.osr replays available for a given
    # osu! installation
    pattern = os.path.join(osu_path, "Replays", "*.osr")
    return glob(pattern)


# Beatmap caching. This reduces beatmap search time a LOT.
#
# Maybe in the future I'll look into using osu! database file for that,
# but this will do just fine for now.
try:
    with open('../.data/beatmap_cache.dat', 'rb') as f:
        _beatmap_cache = pickle.load(f)
except:
    _beatmap_cache = {}


def _get_replay_beatmap_file(osu_path, replay_file):
    global _beatmap_cache

    m = re.search(r"[^\\/]+ \- (.+ \- .+) \[(.+)\] \(.+\)", replay_file)
    if m is None:
        return None
    beatmap, diff = m[1], m[2]

    beatmap_file_pattern = "*" + glob_escape(beatmap) + "*" + glob_escape("[" + diff + "]") + ".osu"
    if beatmap_file_pattern in _beatmap_cache:
        return _beatmap_cache[beatmap_file_pattern]

    pattern = os.path.join(osu_path, "Songs", "**", beatmap_file_pattern)
    file_matches = glob(pattern)

    if len(file_matches) > 0:
        _beatmap_cache[beatmap_file_pattern] = file_matches[0]
        return file_matches[0]
    else:
        _beatmap_cache[beatmap_file_pattern] = None
        return None


def _beatmap_frame(beatmap, time):
    visible_objects = beatmap.visible_objects(time, count=1)

    if len(visible_objects) > 0:
        obj = visible_objects[0]
        beat_duration = beatmap.beat_duration(obj.time)
        px, py = obj.target_position(time, beat_duration, beatmap['SliderMultiplier'])
        time_left = obj.time - time
        is_slider = int(isinstance(obj, hitobjects.Slider))
        is_spinner = int(isinstance(obj, hitobjects.Spinner))
    else:
        return None

    px = max(0, min(px / osu_core.SCREEN_WIDTH, 1))
    py = max(0, min(py / osu_core.SCREEN_HEIGHT, 1))

    return px, py, time_left, is_slider, is_spinner


def _replay_frame(beatmap, replay, time):
    x, y, _ = replay.frame(time)
    x = max(0, min(x / osu_core.SCREEN_WIDTH, 1))
    y = max(0, min(y / osu_core.SCREEN_HEIGHT, 1))
    return x, y