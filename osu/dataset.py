import math
import os
import pickle
import random
import typing
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import osu.rulesets.beatmap as osu_beatmap
import osu.rulesets.core as osu_core
import osu.rulesets.hitobjects
import osu.rulesets.hitobjects as ho
import osu.rulesets.replay as osu_replay
from osu.rulesets.keys import Keys

BATCH_LENGTH = 2048
# sample rate in ms
SAMPLE_RATE = 24

INPUT_FEATURES = [
    'x', 'y',
    'time_until_click', # in seconds
    'is_slider', # bool
    'is_spinner', # bool
    'is_note', # bool
    'cs', # normalized from 0-1 (cs value / 10)
    'slider_speed', # (slider pixel len / duration)
    'slider_len' # osu pixel len / 600
]

# k1 and k2 are the key presses (z, x). no mouse buttons because
# who really uses mouse buttons tbh, dataset will convert
# m1 and m2 into k1 and k2 respectively.
OUTPUT_FEATURES = ['x', 'y', 'k1', 'k2']

MAX_TIME_UNTIL_CLICK = 2.0
DEFAULT_CS = 0.4
DEFAULT_SLIDER_LEN = 0
DEFAULT_SLIDER_SPEED = 0

# Default frame (pre padding)
DEFAULT_FRAME = (
    osu_core.SCREEN_WIDTH / 2, osu_core.SCREEN_HEIGHT / 2,  # x, y (raw coordinates)
    MAX_TIME_UNTIL_CLICK, False, False, False,  # time_until_click (seconds), is_slider, is_spinner, is_note
    DEFAULT_CS, DEFAULT_SLIDER_SPEED,
    DEFAULT_SLIDER_LEN
)

# Default frame for padding
DEFAULT_NORM_FRAME = (
    0.0, 0.0,  # x, y (normalized center: (256/512 - 0.5) = 0.0, (192/384 - 0.5) = 0.0)
    MAX_TIME_UNTIL_CLICK, False, False, False,  # time_until_click, is_slider, is_spinner, is_note
    DEFAULT_CS, DEFAULT_SLIDER_SPEED,
    DEFAULT_SLIDER_LEN
)

class MapDataCache:
    @dataclass
    class ObjData:
        time: int
        x: int
        y: int
        type: str
        is_slider: bool
        is_spinner: bool
        is_note: bool
        obj_ref: ho.HitObject
        # stores any precomputed data here
        data: dict[str, typing.Any]

    def __init__(self, beatmap: osu_beatmap.Beatmap):
        self.beatmap = beatmap
        
        self.preempt, _ = beatmap.approach_rate()
        self.circle_size = beatmap.circle_size() / 10.0
        self.slider_multiplier = beatmap.slider_multiplier()
        
        # Pre-compute all hit object data
        self.hit_objects_data = self._precompute_hit_objects()
        
        # Pre-compute timing data for all unique object times
        self.timing_cache = self._precompute_timing_data()
        
        self._build_time_lookups()
        
        self.curve_position_cache = {}
        
        self.frame_cache = self._compute_frames()

    import math

    def _precompute_hit_objects(self):
        """Extract and pre-process all hit object data"""
        objects_data = []
        
        for obj in self.beatmap.effective_hit_objects:
            obj_data = self.ObjData(
                time=obj.time,
                x=obj.x,
                y=obj.y,
                type=type(obj).__name__,
                is_slider=isinstance(obj, ho.Slider),
                is_spinner=isinstance(obj, ho.Spinner),
                is_note=not isinstance(obj, ho.Slider) and not isinstance(obj, ho.Spinner),
                obj_ref=obj,
                data={}
            )
            
            # Pre-compute slider-specific data
            if obj_data.is_slider:
                obj_data.data['pixel_length'] = obj.pixel_length
                
                # Check if slider is too short - decide once per object
                beat_duration = self.beatmap.beat_duration(obj.time)
                slider_duration = obj.duration(beat_duration, self.slider_multiplier)
                
                # if its below 0.005 its probably some aspire map
                # with like infinite bpm so treat it like a regular note
                if slider_duration < 0.005:
                    obj_data.is_slider = False
                else:
                    raw_speed = (obj_data.data['pixel_length'] / slider_duration)
                    slider_speed = min(raw_speed if not math.isnan(raw_speed) else 10.0, 10.0)
                    if slider_speed == 10.0:
                        # NO human is following this. just treat it like a regular note lol
                        # well actually nah not yet
                        print("slider speed got clamped to 10.. odd...")
                        print(f'{self.beatmap.title()} [{self.beatmap.version()}]')
                        print('raw speed value: ' + str(raw_speed))
                        print('slider dur: ' + str(slider_duration))
                        print('slider len: ' + str(obj_data.data['pixel_length']))
                    
                    obj_data.data['slider_speed'] = slider_speed
                    obj_data.data['slider_len'] = obj_data.data['pixel_length'] / 600.0
                    obj_data.data['duration'] = slider_duration

            objects_data.append(obj_data)
        
        return objects_data
    
    def _precompute_timing_data(self):
        """Pre-compute beat duration for all unique object times"""
        timing_cache = {}
        unique_times = set(obj.time for obj in self.hit_objects_data)
        
        for obj_time in unique_times:
            timing_cache[obj_time] = self.beatmap.beat_duration(obj_time)
        
        return timing_cache
    
    def _build_time_lookups(self):
        self.slider_intervals = []  # (start_time, end_time, obj_data)
        self.spinner_intervals = [] # (start_time, end_time, obj_data)  
        self.objects = []           # (time, obj_data) sorted by time
        
        for obj_data in self.hit_objects_data:
            if obj_data.is_slider and obj_data.data.get('duration'):
                start_time = obj_data.time
                end_time = start_time + obj_data.data['duration']
                self.slider_intervals.append((start_time, end_time, obj_data))
            elif obj_data.is_spinner:
                start_time = obj_data.time
                spinner_obj = obj_data.obj_ref
                end_time = spinner_obj.end_time
                self.spinner_intervals.append((start_time, end_time, obj_data))
            
            self.objects.append((obj_data.time, obj_data))
        
        self.slider_intervals.sort(key=lambda x: x[0])  # Sort by start time
        self.spinner_intervals.sort(key=lambda x: x[0])
        self.objects.sort(key=lambda x: x[0])           # Sort by time
        
        self.slider_start_times = [interval[0] for interval in self.slider_intervals]
        self.spinner_start_times = [interval[0] for interval in self.spinner_intervals]
        self.object_times = [obj[0] for obj in self.objects]
    
    def _compute_frames(self):
        frame_cache = {}
        
        start_time = self.beatmap.start_offset()
        end_time = self.beatmap.length()
        
        # Pre-compute visible objects for each time point
        for time in range(start_time, end_time, SAMPLE_RATE):
            visible_objects = self._find_visible_objects_at_time(time)
            
            if visible_objects:
                obj_data = visible_objects[0]
                obj = obj_data.obj_ref
                
                # Get cached timing data
                beat_duration = self.timing_cache[obj_data.time]
                
                px, py = self._target_position(obj, time, beat_duration, self.slider_multiplier)

                if obj_data.is_spinner:
                    # for spinners the time_left is different
                    if time > obj_data.time:
                        # the spinner is ACTIVE (should be spun), increase time_left by 0.1x
                        # of the time passed since started (spinners can last quite long)
                        time_left = 0.1 * (time - obj_data.time)
                    else:
                        # the spinner is not yet active proceed as normal
                        time_left = obj_data.time - time
                else:
                    time_left = obj_data.time - time

                if obj_data.is_slider:
                    slider_speed = obj_data.data.get('slider_speed', DEFAULT_SLIDER_SPEED)
                    slider_len = obj_data.data.get('slider_len', DEFAULT_SLIDER_LEN)
                else:
                    slider_speed = DEFAULT_SLIDER_SPEED
                    slider_len = DEFAULT_SLIDER_LEN
                
                frame_cache[time] = (
                    px, py, time_left,
                    int(obj_data.is_slider), int(obj_data.is_spinner), int(obj_data.is_note),
                    self.circle_size, slider_speed, slider_len
                )
            else:
                frame_cache[time] = None
        
        return frame_cache
    
    def _find_visible_objects_at_time(self, time):
        import bisect

        # bsearch for sliders (prioritize these)
        active_sliders = []
        
        left_idx = bisect.bisect_right(self.slider_start_times, time)
        
        for i in range(left_idx):
            start_time, end_time, obj_data = self.slider_intervals[i]
            if start_time <= time <= end_time:
                active_sliders.append(obj_data)
        
        if active_sliders:
            active_sliders.sort(key=lambda x: x.time)
            return active_sliders[:1]
        
        # bsearch for spinners
        active_spinners = []
        
        left_idx = bisect.bisect_right(self.spinner_start_times, time)
        
        for i in range(left_idx):
            start_time, end_time, obj_data = self.spinner_intervals[i]
            if start_time <= time <= end_time:
                active_spinners.append(obj_data)
        
        if active_spinners:
            active_spinners.sort(key=lambda x: x.time)
            return active_spinners[:1]

        # if no sliders or spinners get a regular note
        time_end = time + self.preempt
        
        left_idx = bisect.bisect_left(self.object_times, time)
        right_idx = bisect.bisect_right(self.object_times, time_end)
        
        if left_idx < right_idx:
            visible = [self.objects[i][1] for i in range(left_idx, right_idx)]
            visible.sort(key=lambda x: x.time)
            return visible[:1]
        else:
            return []
    
    
    def _target_position(self, obj, time, beat_duration, slider_multiplier):
        cache_key = (obj.time, obj.x, obj.y, time, beat_duration, slider_multiplier)
        
        if cache_key in self.curve_position_cache:
            return self.curve_position_cache[cache_key]

        # slider curve calculaiton ius SO SLOW
        px, py = obj.target_position(time, beat_duration, slider_multiplier)
        
        self.curve_position_cache[cache_key] = (px, py)
        return px, py
    
    def get_frame_data(self, time):
        return self.frame_cache.get(time)


def pad_map_frames(data, max_len):
    if not data:
        return np.array([])
    
    batch_size = len(data)
    result = np.full((batch_size, max_len, 9), DEFAULT_NORM_FRAME, dtype=np.float32)
    
    for i, seq in enumerate(data):
        if seq:
            seq_array = np.stack(seq).astype(np.float32)
            seq_len = min(len(seq), max_len)
            result[i, :seq_len] = seq_array[:seq_len]
    
    return result


def pad_play_frames(data, maxlen):
    """Pad replay sequences (x, y, k1, k2) with zeros"""
    if not data:
        return np.array([])
    
    batch_size = len(data)
    result = np.zeros((batch_size, maxlen, 4), dtype=np.float32)
    
    for i, seq in enumerate(data):
        if seq:
            seq_array = np.stack(seq).astype(np.float32)
            seq_len = min(len(seq), maxlen)
            result[i, :seq_len] = seq_array[:seq_len]
    
    return result


def load(files, verbose=0) -> pd.DataFrame:
    """Map the replay and beatmap files into osu! ruleset objects"""

    replays = []
    beatmaps = []

    beatmap_cache = dict()

    for index, row in tqdm.tqdm(files.iterrows(), desc='Loading objects from .data/', total=len(files)):
        try:
            beatmap_path: str = row['beatmap']
            replay: osu_replay.Replay = osu_replay.load(row['replay'])
            # only replays from 2022 and forward. hardcoded for now TODO!
            # if replay.timestamp < 637766784000000000:
            #     continue

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


# extract the data for a single replay
def replay_to_output_data(beatmap: osu_beatmap.Beatmap, replay: osu_replay.Replay):
    target_data = []

    chunk = []
    for time in range(beatmap.start_offset(), beatmap.length(), SAMPLE_RATE):
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

        for time in range(beatmap.start_offset(), beatmap.length(), SAMPLE_RATE):
            x, y, k1, k2 = _replay_frame(replay, time)

            chunk.append(np.array([x - 0.5, y - 0.5, k1, k2]))

            if len(chunk) == BATCH_LENGTH:
                target_data.append(chunk)
                chunk = []

        if len(chunk) > 0:
            target_data.append(chunk)

    data = pad_play_frames(target_data, maxlen=BATCH_LENGTH)
    index = pd.MultiIndex.from_product([range(len(data)), range(BATCH_LENGTH)], names=['chunk', 'frame'])
    return pd.DataFrame(np.reshape(data, (-1, len(OUTPUT_FEATURES))), index=index, columns=OUTPUT_FEATURES,
                        dtype=np.float32)


def _replay_frame(replay, time):
    x, y, keys = replay.frame(time)

    k1 = bool(keys & (Keys.K1 | Keys.M1))
    k2 = bool(keys & (Keys.K2 | Keys.M2))

    x = max(0, min(x / osu_core.SCREEN_WIDTH, 1))
    y = max(0, min(y / osu_core.SCREEN_HEIGHT, 1))
    return x, y, k1, k2


def get_beatmap_time_data(beatmap: osu_beatmap.Beatmap) -> []:
    if len(beatmap.effective_hit_objects) == 0:
        return None

    precomputed = MapDataCache(beatmap)
    
    data = []
    chunk = []

    for time in range(beatmap.start_offset(), beatmap.length(), SAMPLE_RATE):
        frame_data = precomputed.get_frame_data(time)

        if frame_data is None:
            # Use default frame
            px_raw, py_raw, time_until_click, is_slider, is_spinner, is_note, circle_size, slider_speed, slider_len = DEFAULT_FRAME
        else:
            px_raw, py_raw, time_left, is_slider, is_spinner, is_note, circle_size, slider_speed, slider_len = frame_data
            time_until_click = max(0, min(time_left / 1000, MAX_TIME_UNTIL_CLICK))
        
        px = max(0, min(px_raw / osu_core.SCREEN_WIDTH, 1))
        py = max(0, min(py_raw / osu_core.SCREEN_HEIGHT, 1))

        chunk.append(np.array([
            px - 0.5,
            py - 0.5,
            time_until_click,
            is_slider,
            is_spinner,
            is_note,
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

    prog = tqdm.tqdm(beatmaps.items(), desc='Turning beatmaps into time series data',
                                    total=len(beatmaps))
    for index, beatmap in prog:
        prog.set_description(f'Turning {beatmap.title()} into time series data')
        chunks = get_beatmap_time_data(beatmap)
        if chunks is not None:
            data.extend(chunks)

    data = pad_map_frames(data, max_len=BATCH_LENGTH)

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

def user_replay_mapping_from_cache(user_id: int, limit: int = None, shuffle: bool = False) -> pd.DataFrame:
    replay_cache_path = f'.data/replays-{user_id}'

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
