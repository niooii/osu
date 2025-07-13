import math
import os
import pickle
import random
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
BATCH_LENGTH = 2048
FRAME_RATE = 24

# Feature index
INPUT_FEATURES = ['x', 'y', 'visible', 'is_slider', 'is_spinner']
OUTPUT_FEATURES = ['x', 'y']

# Default beatmap frame information
_DEFAULT_BEATMAP_FRAME = (
    osu_core.SCREEN_WIDTH / 2, osu_core.SCREEN_HEIGHT / 2,  # x, y
    float("inf"), False, False  # time_left, is_slider, is_spinner
)


def pt_pad_sequences(data, maxlen, value=_DEFAULT_BEATMAP_FRAME):
    if not data:
        return np.array([])

    # Convert to list of tensors
    tensor_list = [torch.tensor(seq, dtype=torch.float) for seq in data]
    # print(f'tensor list len: {len(tensor_list)}')
    # print(f'tensor list (first element) shape: {tensor_list[0].shape}')

    # Pad sequences
    padded = pad_sequence(tensor_list, batch_first=True, padding_value=value)

    # Trim or pad to exact maxlen
    if padded.size(1) > maxlen:
        padded = padded[:, :maxlen]
    elif padded.size(1) < maxlen:
        pad = maxlen - padded.size(1)
        # print(f'erm padding by {pad}?')
        padding_tuple = (0, 0, 0, pad)
        # print(f'padding tuple: {padding_tuple}')
        padded = torch.nn.functional.pad(padded, padding_tuple, value=value)

    # print(f'padded tensor list len: {len(padded)}')
    # print(f'padded tensor list (first element) shape: {padded[0].shape}')

    return padded.numpy()


def all_files(osu_path, limit=0, verbose=False):
    """Return a pandas DataFrame mapping replay files to beatmap files"""

    replays = _list_all_replays(osu_path)
    if limit > 0:
        replays = replays[:limit]

    beatmaps = []
    for i in tqdm.tqdm(range(len(replays) - 1, -1, -1), disable=not verbose):

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

    beatmap_cache = dict()

    for index, row in tqdm.tqdm(files.iterrows(), desc='Loading objects from .data/', total=len(files)):
        try:
            beatmap_path: str = row['beatmap']
            replay: osu_replay.Replay = osu_replay.load(row['replay'])
            cache_key = (beatmap_path, replay.mods)
            if replay.has_mods(osu_replay.Mod.DT, osu_replay.Mod.HT):
                if verbose:
                    print('DT, HT are not supported yet.')
                continue

            # TODO temp exclusion until i figure out why the training data is
            # so horrible and the model is learning nothing if i include hr replays
            # if replay.has_mods(osu_replay.Mod.HR, osu_replay.Mod.EZ):
            #     continue

            replays.append(replay)
            if cache_key not in beatmap_cache:
                beatmap_cache[cache_key] = osu_beatmap.load(row['beatmap'])
                # erm
                if replay.has_mods(osu_replay.Mod.DT):
                    beatmap_cache[cache_key].apply_mods(['dt'])
                if replay.has_mods(osu_replay.Mod.HT):
                    beatmap_cache[cache_key].apply_mods(['ht'])
                if replay.has_mods(osu_replay.Mod.EZ):
                    beatmap_cache[cache_key].apply_mods(['ez'])
                if replay.has_mods(osu_replay.Mod.HR):
                    # print("applying hr")
                    # print('cache key:')
                    # print(cache_key)
                    beatmap_cache[cache_key].apply_mods(['hr'])

            beatmaps.append(beatmap_cache[cache_key])

        except Exception as e:
            if verbose:
                print()
                print("\tFailed:", e)
            continue

    # print("enumerating beatmap mods")
    # for beatmap in beatmaps:
    #     print(beatmap.get_mods())

    return pd.DataFrame(list(zip(replays, beatmaps)), columns=['replay', 'beatmap'])


_memo = {}


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

    for index, row in tqdm.tqdm(dataset.iterrows(), desc='Turning replays into time series data', total=len(dataset)):
        replay = row['replay']
        beatmap: osu_beatmap.Beatmap = row['beatmap']

        if len(beatmap.effective_hit_objects) == 0:
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
    return pd.DataFrame(np.reshape(data, (-1, len(OUTPUT_FEATURES))), index=index, columns=OUTPUT_FEATURES,
                        dtype=np.float32)


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


def get_beatmap_time_data(beatmap: osu_beatmap.Beatmap) -> []:
    if beatmap in _memo:
        return _memo[beatmap]

    if len(beatmap.effective_hit_objects) == 0:
        return None

    _memo[beatmap] = []
    data = []
    chunk = []
    preempt, _ = beatmap.approach_rate()
    last_ok_frame = None

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

    for index, beatmap in tqdm.tqdm(beatmaps.items(), desc='Turning beatmaps into time series data',
                                    total=len(beatmaps)):
        chunks = get_beatmap_time_data(beatmap)
        if chunks is not None:
            data.extend(chunks)

    data = pt_pad_sequences(data, maxlen=BATCH_LENGTH, value=0)

    index = pd.MultiIndex.from_product([
        range(len(data)), range(BATCH_LENGTH)
    ], names=['chunk', 'frame'])

    # print(data.shape)
    data = np.reshape(data, (-1, len(INPUT_FEATURES)))
    return pd.DataFrame(data, index=index, columns=INPUT_FEATURES, dtype=np.float32)


import json
import tqdm


def replay_mapping_from_cache(limit: int = None, shuffle: bool = False) -> pd.DataFrame:
    replay_cache_path = '.data/replays'

    files = os.listdir(replay_cache_path)
    metafiles = [f for f in files if f.endswith('.meta')]

    if shuffle:
        random.shuffle(files)

    rows = []

    i = 0
    for path in tqdm.tqdm(metafiles, desc='Loading replay metadata from .data/replays', total=limit or len(metafiles)):
        if limit is not None and i == limit:
            break

        metafile_path = os.path.join(replay_cache_path, path)

        replay_path = metafile_path.replace('.meta', '.osr')

        with open(metafile_path, 'r') as metafile:
            meta_json = json.loads(metafile.read())

        map_path = meta_json['map']
        if not os.path.exists(map_path) or not os.path.exists(replay_path):
            continue

        rows.append((replay_path, map_path))
        i += 1

    df = pd.DataFrame(rows, columns=['replay', 'beatmap'])

    return load(df)
