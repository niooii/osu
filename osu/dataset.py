import os
import pickle
import random
import re

import numpy as np
import pandas as pd

import osu.rulesets.beatmap as osu_beatmap
import osu.rulesets.core as osu_core
import osu.rulesets.hitobjects as hitobjects
import osu.rulesets.replay as osu_replay
from osu.rulesets.keys import Keys

# Constants
BATCH_LENGTH = 2048
FRAME_RATE = 24

# Feature index
INPUT_FEATURES = [
    'x', 'y',
    'time_until_click', # in seconds
    'is_slider', # bool
    'is_spinner', # bool
    'cs', # normalized from 0-1 (cs value / 10)
    'slider_speed', # (slider pixel len / duration)
    'slider_len' # osu pixel len / 600
]

MAX_TIME_UNTIL_CLICK = 2.0
DEFAULT_CS = 0.4
DEFAULT_SLIDER_LEN = 0
DEFAULT_SLIDER_SPEED = 0

class PrecomputedBeatmapData:
    """Pre-compute all expensive beatmap calculations once for massive speedup"""
    
    def __init__(self, beatmap: osu_beatmap.Beatmap):
        self.beatmap = beatmap
        
        # Pre-compute static properties (calculated once)
        self.preempt, _ = beatmap.approach_rate()
        self.circle_size = beatmap.circle_size() / 10.0
        self.slider_multiplier = beatmap.slider_multiplier()
        
        # Pre-compute all hit object data
        self.hit_objects_data = self._precompute_hit_objects()
        
        # Pre-compute timing data for all unique object times
        self.timing_cache = self._precompute_timing_data()
        
        # Build optimized lookup structures for fast object finding
        self._build_time_indexed_lookups()
        
        # Build time-indexed frame data (the key optimization)
        self.frame_cache = self._precompute_all_frames()
    
    def _precompute_hit_objects(self):
        """Extract and pre-process all hit object data"""
        objects_data = []
        
        for obj in self.beatmap.effective_hit_objects:
            obj_data = {
                'time': obj.time,
                'x': obj.x,
                'y': obj.y,
                'type': type(obj).__name__,
                'is_slider': isinstance(obj, hitobjects.Slider),
                'is_spinner': isinstance(obj, hitobjects.Spinner),
                'obj_ref': obj,  # Cache object reference to avoid linear searches
            }
            
            # Pre-compute slider-specific data
            if obj_data['is_slider']:
                obj_data['pixel_length'] = obj.pixel_length
                obj_data['slider_obj'] = obj
                
                # Check if slider is too short - decide once per object
                beat_duration = self.beatmap.beat_duration(obj.time)
                slider_duration = obj.duration(beat_duration, self.slider_multiplier)
                
                # if its below 0.05 its probably some aspire map
                # with like infinite bpm so treat it like a regular note
                if slider_duration < 0.05:
                    obj_data['is_slider'] = False
                else:
                    raw_speed = (obj_data['pixel_length'] / slider_duration)
                    slider_speed = min(raw_speed, 10.0)
                    if slider_speed == 10.0:
                        # NO human is following this. just treat it like a regular note lol
                        # well actually nah not yet
                        print("slider speed got clamped to 10.. odd...")
                        print(f'{self.beatmap.title()} [{self.beatmap.version()}]')
                        print('raw speed value: ' + str(raw_speed))
                        print('slider dur: ' + str(slider_duration))
                        print('slider len: ' + str(obj_data['pixel_length']))
                    
                    obj_data['precomputed_slider_speed'] = slider_speed
                    raw_len = obj_data['pixel_length'] / 600.0
                    obj_data['precomputed_slider_len'] = raw_len
                    # Cache the duration to avoid 200M+ recalculations
                    obj_data['precomputed_duration'] = slider_duration

            objects_data.append(obj_data)
        
        return objects_data
    
    def _precompute_timing_data(self):
        """Pre-compute beat duration for all unique object times"""
        timing_cache = {}
        unique_times = set(obj['time'] for obj in self.hit_objects_data)
        
        for obj_time in unique_times:
            timing_cache[obj_time] = self.beatmap.beat_duration(obj_time)
        
        return timing_cache
    
    def _build_time_indexed_lookups(self):
        import bisect
        
        # Separate sliders and regular objects, sort by time
        self.slider_intervals = []  # (start_time, end_time, obj_data)
        self.sorted_objects = []    # (time, obj_data) sorted by time
        
        for obj_data in self.hit_objects_data:
            if obj_data['is_slider'] and obj_data.get('precomputed_duration'):
                # Build slider intervals for active slider lookup
                start_time = obj_data['time']
                end_time = start_time + obj_data['precomputed_duration']
                self.slider_intervals.append((start_time, end_time, obj_data))
            
            # Add all objects to sorted list for upcoming object lookup
            self.sorted_objects.append((obj_data['time'], obj_data))
        
        # Sort for binary search
        self.slider_intervals.sort(key=lambda x: x[0])  # Sort by start time
        self.sorted_objects.sort(key=lambda x: x[0])    # Sort by time
    
    def _precompute_all_frames(self):
        """Pre-compute frame data for all time points"""
        frame_cache = {}
        
        start_time = self.beatmap.start_offset()
        end_time = self.beatmap.length()
        
        # Pre-compute visible objects for each time point
        for time in range(start_time, end_time, FRAME_RATE):
            # Find visible objects at this time (done once, not per frame)
            visible_objects = self._find_visible_objects_at_time(time)
            
            if visible_objects:
                # Use first visible object (same logic as original)
                obj_data = visible_objects[0]
                obj = obj_data.get('slider_obj') or obj_data.get('obj_ref')
                
                # Get cached timing data
                beat_duration = self.timing_cache[obj_data['time']]
                
                # Calculate position (if slider otherwise we haev it cached kinda)
                px, py = obj.target_position(time, beat_duration, self.slider_multiplier)
                time_left = obj_data['time'] - time

                if obj_data['is_slider']:
                    slider_speed = obj_data.get('precomputed_slider_speed', DEFAULT_SLIDER_SPEED)
                    slider_len = obj_data.get('precomputed_slider_len', DEFAULT_SLIDER_LEN)
                else:
                    slider_speed = DEFAULT_SLIDER_SPEED
                    slider_len = DEFAULT_SLIDER_LEN
                
                # Store pre-computed frame data as tuple for memory efficiency
                frame_cache[time] = (
                    px, py, time_left,
                    int(obj_data['is_slider']), int(obj_data['is_spinner']),
                    self.circle_size, slider_speed, slider_len
                )
            else:
                frame_cache[time] = None
        
        return frame_cache
    
    def _find_visible_objects_at_time(self, time):
        """Find active objects at given time - prioritizes currently playing sliders"""
        import bisect
        
        # check for active sliders using binary search
        active_sliders = []
        
        # Binary search for sliders that could be active at this time
        left_idx = bisect.bisect_right([interval[0] for interval in self.slider_intervals], time)
        
        # Check sliders that start before or at current time
        for i in range(left_idx):
            start_time, end_time, obj_data = self.slider_intervals[i]
            if start_time <= time <= end_time:
                active_sliders.append(obj_data)
        
        if active_sliders:
            # Sort by start time and return the first active slider (preserves original logic)
            active_sliders.sort(key=lambda x: x['time'])
            return active_sliders[:1]
        
        # if no active sliders, find upcoming visible objects using binary search
        visible = []
        time_end = time + self.preempt
        
        # binary search for objects in time range [time, time + preempt]
        left_idx = bisect.bisect_left([obj[0] for obj in self.sorted_objects], time)
        right_idx = bisect.bisect_right([obj[0] for obj in self.sorted_objects], time_end)
        
        # collect objects in time range
        for i in range(left_idx, right_idx):
            obj_time, obj_data = self.sorted_objects[i]
            visible.append(obj_data)
        
        visible.sort(key=lambda x: x['time'])
        return visible[:1]  # Return first object (same as count=1)
    
    
    def get_frame_data(self, time):
        """Get pre-computed frame data - O(1) lookup instead of O(n) calculation"""
        return self.frame_cache.get(time)

# k1 and k2 are the key presses (z, x). no mouse buttons because
# who really uses mouse buttons tbh, dataset will convert
# m1 and m2 into k1 and k2 respectively.
OUTPUT_FEATURES = ['x', 'y', 'k1', 'k2']

# Default beatmap frame information
_DEFAULT_BEATMAP_FRAME = (
    osu_core.SCREEN_WIDTH / 2, osu_core.SCREEN_HEIGHT / 2,  # x, y
    MAX_TIME_UNTIL_CLICK, False, False,  # time_until_click (seconds), is_slider, is_spinner
    DEFAULT_CS, DEFAULT_SLIDER_SPEED,
    DEFAULT_SLIDER_LEN
)


def pt_pad_sequences(data, maxlen, value=_DEFAULT_BEATMAP_FRAME):
    if not data:
        return np.array([])
    
    # Convert each sequence to numpy array (equivalent to torch.tensor step)
    arrays = []
    for seq in data:
        if seq:  # non-empty sequence
            arrays.append(np.stack(seq).astype(np.float32))
        else:  # empty sequence
            # Determine feature dimension from other sequences
            feature_dim = None
            for other_seq in data:
                if other_seq:
                    feature_dim = len(other_seq[0])
                    break
            if feature_dim is None:
                # All sequences empty - use value to determine feature dim
                feature_dim = len(value) if hasattr(value, '__len__') else 8
            arrays.append(np.empty((0, feature_dim), dtype=np.float32))
    
    if not arrays:
        return np.array([])
    
    # Get dimensions
    max_seq_len = max(arr.shape[0] for arr in arrays)
    num_features = arrays[0].shape[1]
    batch_size = len(arrays)
    
    # Step 1: Equivalent to pad_sequence - pad all sequences to max_seq_len
    padded = np.full((batch_size, max_seq_len, num_features), value, dtype=np.float32)
    for i, arr in enumerate(arrays):
        if arr.shape[0] > 0:
            padded[i, :arr.shape[0]] = arr
    
    # Step 2: Trim or pad to exact maxlen (equivalent to torch.nn.functional.pad)
    if max_seq_len > maxlen:
        # Trim to maxlen
        result = padded[:, :maxlen]
    elif max_seq_len < maxlen:
        # Pad to maxlen 
        extra_frames = maxlen - max_seq_len
        extra_padding = np.full((batch_size, extra_frames, num_features), value, dtype=np.float32)
        result = np.concatenate([padded, extra_padding], axis=1)
    else:
        result = padded
    
    return result


# def all_files(osu_path, limit=0, verbose=False):
#     """Return a pandas DataFrame mapping replay files to beatmap files"""
#
#     replays = _list_all_replays(osu_path)
#     if limit > 0:
#         replays = replays[:limit]
#
#     beatmaps = []
#     for i in tqdm.tqdm(range(len(replays) - 1, -1, -1), disable=not verbose):
#
#         beatmap = _get_replay_beatmap_file(osu_path, replays[i])
#
#         if beatmap is None:
#             replays.pop(i)
#         else:
#             beatmaps.insert(0, beatmap)
#
#     global _beatmap_cache
#     with open('../.data/beatmap_cache.dat', 'wb') as f:
#         pickle.dump(_beatmap_cache, f)
#
#     if verbose:
#         print()
#         print()
#
#     files = list(zip(replays, beatmaps))
#     return pd.DataFrame(files, columns=['replay', 'beatmap'])


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

            replays.append(replay)
            if cache_key not in beatmap_cache:
                beatmap_cache[cache_key] = osu_beatmap.load(row['beatmap'])
                beatmap_cache[cache_key].apply_mods(replay.mods)

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


# Removed _memo cache - memoization breaks time-series alignment with target_data


# extract the data for a single replay
def replay_to_output_data(beatmap: osu_beatmap.Beatmap, replay: osu_replay.Replay):
    target_data = []

    chunk = []
    for time in range(beatmap.start_offset(), beatmap.length(), FRAME_RATE):
        x, y, k1, k2 = _replay_frame(beatmap, replay, time)

        chunk.append(np.array([x - 0.5, y - 0.5, k1, k2]))

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
            x, y, k1, k2 = _replay_frame(beatmap, replay, time)

            chunk.append(np.array([x - 0.5, y - 0.5, k1, k2]))

            if len(chunk) == BATCH_LENGTH:
                target_data.append(chunk)
                chunk = []

        if len(chunk) > 0:
            target_data.append(chunk)

    data = pt_pad_sequences(target_data, maxlen=BATCH_LENGTH, value=0)
    index = pd.MultiIndex.from_product([range(len(data)), range(BATCH_LENGTH)], names=['chunk', 'frame'])
    return pd.DataFrame(np.reshape(data, (-1, len(OUTPUT_FEATURES))), index=index, columns=OUTPUT_FEATURES,
                        dtype=np.float32)


# def _list_all_replays(osu_path):
#     # Returns the full list of *.osr replays available for a given
#     # osu! installation
#     pattern = os.path.join(osu_path, "Replays", "*.osr")
#     return glob(pattern)


# Beatmap caching. This reduces beatmap search time a LOT.
#
# Maybe in the future I'll look into using osu! database file for that,
# but this will do just fine for now.
try:
    with open('../.data/beatmap_cache.dat', 'rb') as f:
        _beatmap_cache = pickle.load(f)
except:
    _beatmap_cache = {}


# def _get_replay_beatmap_file(osu_path, replay_file):
#     global _beatmap_cache
#
#     m = re.search(r"[^\\/]+ \- (.+ \- .+) \[(.+)\] \(.+\)", replay_file)
#     if m is None:
#         return None
#     beatmap, diff = m[1], m[2]
#
#     beatmap_file_pattern = "*" + glob_escape(beatmap) + "*" + glob_escape("[" + diff + "]") + ".osu"
#     if beatmap_file_pattern in _beatmap_cache:
#         return _beatmap_cache[beatmap_file_pattern]
#
#     pattern = os.path.join(osu_path, "Songs", "**", beatmap_file_pattern)
#     file_matches = glob(pattern)
#
#     if len(file_matches) > 0:
#         _beatmap_cache[beatmap_file_pattern] = file_matches[0]
#         return file_matches[0]
#     else:
#         _beatmap_cache[beatmap_file_pattern] = None
#         return None


# Pre-cache enum values to avoid repeated enum access
_K1_M1_MASK = Keys.K1 | Keys.M1
_K2_M2_MASK = Keys.K2 | Keys.M2

def _replay_frame(beatmap, replay, time):
    x, y, keys = replay.frame(time)

    # Optimize enum operations - use pre-cached masks
    k1 = bool(keys & _K1_M1_MASK)
    k2 = bool(keys & _K2_M2_MASK)

    # Normalize coordinates
    x = max(0, min(x / osu_core.SCREEN_WIDTH, 1))
    y = max(0, min(y / osu_core.SCREEN_HEIGHT, 1))
    return x, y, k1, k2


def get_beatmap_time_data(beatmap: osu_beatmap.Beatmap) -> []:
    if len(beatmap.effective_hit_objects) == 0:
        return None

    # Create precomputed beatmap data
    precomputed = PrecomputedBeatmapData(beatmap)
    
    data = []
    chunk = []
    last_ok_frame = None

    for time in range(beatmap.start_offset(), beatmap.length(), FRAME_RATE):
        # O(1) lookup
        frame_data = precomputed.get_frame_data(time)

        if frame_data is None:
            if last_ok_frame is None:
                frame = _DEFAULT_BEATMAP_FRAME
            else:
                frame = list(last_ok_frame)
                frame[2] = MAX_TIME_UNTIL_CLICK
        else:
            # Unpack precomputed tuple data for faster access
            px_raw, py_raw, time_left, is_slider, is_spinner, circle_size, slider_speed, slider_len = frame_data
            
            # Convert to normalized coordinates
            px = max(0, min(px_raw / osu_core.SCREEN_WIDTH, 1))
            py = max(0, min(py_raw / osu_core.SCREEN_HEIGHT, 1))
            time_until_click = max(0, min(time_left / 1000, MAX_TIME_UNTIL_CLICK))
            
            frame = (
                px, py,
                time_until_click,
                is_slider,
                is_spinner,
                circle_size,
                slider_speed,
                slider_len
            )
            last_ok_frame = frame

        px, py, time_until_click, is_slider, is_spinner, circle_size, slider_speed, slider_len = frame

        chunk.append(np.array([
            px - 0.5,
            py - 0.5,
            time_until_click,
            is_slider,
            is_spinner,
            circle_size,
            slider_speed,
            slider_len
        ]))

        if len(chunk) == BATCH_LENGTH:
            data.append(chunk)
            chunk = []

    if len(chunk) > 0:
        data.append(chunk)

    return data


def input_data(dataset, verbose=False) -> pd.DataFrame:
    """Given a osu! ruleset dataset for replays and maps, generate a
    new DataFrame with beatmap object information across time."""

    data = []

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
