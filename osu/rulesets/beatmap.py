import math
from datetime import time

from . import core, hitobjects, timing_points
from ._util.bsearch import bsearch
from .mods import Mods

import re

_SECTION_TYPES = {
    'General': 'a',
    'Editor': 'a',
    'Metadata': 'a',
    'Difficulty': 'a',
    'Events': 'b',
    'TimingPoints': 'b',
    'Colours': 'a',
    'HitObjects': 'b'
}


class _BeatmapFile:
    def __init__(self, file):
        self.file = file
        self.format_version = self.file.readline()

    def read_all_sections(self):
        sections = {}
        section = self._read_section_header()
        while section != None:
            func = "_read_type_%s_section" % _SECTION_TYPES[section]
            section_name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', section).lower()

            sections[section_name] = getattr(self, func)()
            section = self._read_section_header()

        return sections

    def _read_section_header(self):
        header = self.file.readline()
        while header != '' and re.match(r"[^\n\r\s]", header) == None:
            header = self.file.readline()

        m = re.match(r"^\s*\[(\S+)\]\s*$", header)

        if m is None:
            return None

        return m[1]

    def _parse_value(self, v):
        if v.isdigit():
            return int(v)
        elif v.replace('.', '', 1).isdigit():
            return float(v)
        else:
            return v

    # Seção do tipo Chave: Valor
    def _read_type_a_section(self):
        d = dict()

        line = self.file.readline()
        while line != '' and re.match(r"[^\n\r\s]", line) != None:
            m = re.match(r"^\s*(\S+)\s*:\s*(.*)\s*\r?\n$", line)
            if m is None:
                raise RuntimeError("Invalid file")
            else:
                d[m[1]] = self._parse_value(m[2])

            line = self.file.readline()

        return d

    # Seção do tipo a,b,c,...,d
    def _read_type_b_section(self):
        l = list()

        line = self.file.readline()
        while line != '' and re.match(r"[^\n\r\s]", line) != None:
            l.append(list(map(self._parse_value, line.rstrip("\r\n").split(','))))
            line = self.file.readline()

        return l


class Beatmap:
    def __init__(self, file):
        file = _BeatmapFile(file)

        self.format_version = file.format_version
        self.sections = file.read_all_sections()
        self.mods = 0
        self._hr_hit_objects_cache = None  # Cache for HR-transformed hit objects

        if 'timing_points' in self.sections:
            self.timing_points = list(map(timing_points.create, self.sections['timing_points']))
            del self.sections['timing_points']

        if 'hit_objects' in self.sections:
            self.hit_objects = list(map(hitobjects.create, self.sections['hit_objects']))
            del self.sections['hit_objects']

    def combo_color(self, new_combo, combo_skip):
        return (255, 0, 0)

    def __getattr__(self, key):
        if key in self.sections:
            return self.sections[key]
        else:
            return []

    def __getitem__(self, key):
        for section in self.sections.values():
            if key in section:
                return section[key]
        return None

    def apply_mods(self, mods: int):
        # Reset all mods first
        # self.dt = False
        # self.hr = False
        # self.ez = False
        # self.ht = False
        # self._hr_hit_objects_cache = None

        # get the mods that only affect the gameplay
        mods = mods & Mods.STD_GAMEPLAY_AFFECTING
        
        # Check for mutually exclusive mods
        if mods & Mods.DOUBLE_TIME and mods & Mods.HALF_TIME:
            raise ValueError("Tried to apply DT and HT.")

        if mods & Mods.HARD_ROCK and mods & Mods.EASY:
            raise ValueError("Tried to apply HR and EZ. pick a strugle buddy")

        if mods & Mods.DOUBLE_TIME:
            # do something
            pass
        if mods & Mods.HARD_ROCK:
            # need to create hr objects bc the game area is flipped on X axis
            self._create_hr_hit_objects_cache()
        if mods & Mods.EASY:
            # do something
            pass
        if mods & Mods.HALF_TIME:
            # do something
            pass

        self.mods = mods

    def get_mods(self):
        return self.mods

    def base_approach_rate(self):
        """Returns the unmodified approach rate value from the beatmap"""
        return self["ApproachRate"]
    
    def base_approach_rate_timing(self):
        """Returns base preempt and fade_in times without mod effects"""
        ar = self.base_approach_rate()
        
        if ar <= 5:
            preempt = 1200 + 600 * (5 - ar) / 5
            fade_in = 800 + 400 * (5 - ar) / 5
        else:
            preempt = 1200 - 750 * (ar - 5) / 5
            fade_in = 800 - 500 * (ar - 5) / 5
            
        return preempt, fade_in
    
    def approach_rate_value(self):
        """Returns the approach rate value after mods are applied"""
        ar = self.base_approach_rate()
        if self.mods & Mods.EASY:
            ar = ar / 2
        elif self.mods & Mods.HARD_ROCK:
            ar = min(ar * 1.4, 10)
        return ar
    
    def approach_rate(self):
        """Returns preempt and fade_in times considering mods"""
        ar = self.approach_rate_value()
        
        if ar <= 5:
            preempt = 1200 + 600 * (5 - ar) / 5
            fade_in = 800 + 400 * (5 - ar) / 5
        else:
            preempt = 1200 - 750 * (ar - 5) / 5
            fade_in = 800 - 500 * (ar - 5) / 5
            
        # TODO: DT/HT timing effects need proper implementation
        # DT should make preempt/fade_in 33% shorter, HT 33% longer
            
        return preempt, fade_in

    def base_circle_size(self):
        """Returns the unmodified circle size value from the beatmap"""
        return self['CircleSize']
    
    def circle_size(self):
        """Returns the circle size value after mods are applied"""
        cs = self.base_circle_size()
        if self.mods & Mods.EASY:
            cs = cs / 2
        elif self.mods & Mods.HARD_ROCK:
            cs = min(cs * 1.3, 10)
        return cs
    
    def circle_radius(self):
        """Returns the circle radius in osu!pixels after mods are applied"""
        return 27.2 - 2.24 * self.circle_size()

    def title(self):
        return self['Title']

    def title_unicode(self):
        return self['TitleUnicode']

    def artist(self):
        return self['Artist']

    def artist_unicode(self):
        return self['ArtistUnicode']

    def creator(self):
        return self['Creator']

    def version(self):
        return self['Version']

    def source(self):
        return self['Source']

    def tags(self):
        return self['Tags']

    def beatmap_id(self):
        return self['BeatmapID']

    def beatmap_set_id(self):
        return self['BeatmapSetID']

    def base_hp_drain_rate(self):
        """Returns the unmodified HP drain rate value from the beatmap"""
        return self['HPDrainRate']
    
    def hp_drain_rate(self):
        """Returns the HP drain rate value after mods are applied"""
        hp = self.base_hp_drain_rate()
        if self.mods & Mods.EASY:
            hp = hp / 2
        elif self.mods & Mods.HARD_ROCK:
            hp = min(hp * 1.4, 10)
        return hp

    def base_overall_difficulty(self):
        """Returns the unmodified overall difficulty value from the beatmap"""
        return self['OverallDifficulty']
    
    def overall_difficulty(self):
        """Returns the overall difficulty value after mods are applied"""
        od = self.base_overall_difficulty()
        if self.mods & Mods.EASY:
            od = od / 2
        elif self.mods & Mods.HARD_ROCK:
            od = min(od * 1.4, 10)
        return od
    
    def hit_windows(self):
        """Returns hit windows (300, 100, 50) in milliseconds after mods are applied"""
        od = self.overall_difficulty()
        
        window_300 = 80 - 6 * od
        window_100 = 140 - 8 * od
        window_50 = 200 - 10 * od
        
        # TODO: DT/HT timing effects need proper implementation
        # DT should make windows 33% shorter, HT 33% longer
            
        return window_300, window_100, window_50
    
    def transform_position(self, x, y):
        """Transform position coordinates based on active mods"""
        if self.mods & Mods.HARD_ROCK:
            # HR flips the beatmap vertically (around X-axis)
            y = core.SCREEN_HEIGHT - y
        return x, y
    
    def spinner_rotation_rate(self):
        """Returns required spins per second for spinners after mods are applied"""
        od = self.overall_difficulty()
        
        if od < 5:
            rate = 5 - 2 * (5 - od) / 5
        elif od == 5:
            rate = 5
        else:
            rate = 5 + 2.5 * (od - 5) / 5
            
        # TODO: DT/HT timing effects need proper implementation
        # DT should multiply rate by 1.5, HT by 0.75
            
        return rate
    
    def _create_hr_hit_objects_cache(self):
        """Create cached HR-transformed hit objects"""
        import copy
        
        self._hr_hit_objects_cache = []
        for obj in self.hit_objects:
            hr_obj = copy.deepcopy(obj)
            
            # Transform main position
            hr_obj.y = core.SCREEN_HEIGHT - hr_obj.y
            
            # Transform slider curve points if it's a slider
            if hasattr(hr_obj, 'curve_points') and hr_obj.curve_points:
                hr_obj.curve_points = [(x, core.SCREEN_HEIGHT - y) for x, y in hr_obj.curve_points]
                # Clear any cached curve calculations
                if hasattr(hr_obj, '_cached_curve_points'):
                    hr_obj._cached_curve_points = None
                if hasattr(hr_obj, '_cached_base_curve'):
                    hr_obj._cached_base_curve = None
            
            self._hr_hit_objects_cache.append(hr_obj)
    
    @property
    def effective_hit_objects(self):
        """Return HR-transformed hit objects if HR is active, otherwise normal hit objects"""
        if self.mods & Mods.HARD_ROCK and self._hr_hit_objects_cache is not None:
            return self._hr_hit_objects_cache
        return self.hit_objects

    def ar_raw(self):
        return self['ApproachRate']

    def slider_multiplier(self):
        return self['SliderMultiplier']

    def slider_tick_rate(self):
        return self['SliderTickRate']

    def base_start_offset(self):
        """Start offset using base approach rate for timing synchronization"""
        base_preempt, _ = self.base_approach_rate_timing()
        return int(self.effective_hit_objects[0].time - base_preempt)
    
    def start_offset(self):
        preempt, _ = self.approach_rate()
        return int(self.effective_hit_objects[0].time - preempt)

    def length(self):
        if len(self.effective_hit_objects) == 0:
            return 0
        last_obj = self.effective_hit_objects[-1]
        beat_duration = self.beat_duration(last_obj.time)
        return int(last_obj.time + last_obj.duration(beat_duration, self['SliderMultiplier']))

    def _timing(self, time):
        bpm = None
        i = bsearch(self.timing_point, time, lambda tp: tp.time)
        timing_point = self.timing_points[i - 1]

        for tp in self.timing_points[i:]:
            if tp.offset > time:
                break
            if tp.bpm > 0:
                bpm = tp.bpm
            timing_point = tp

        while i >= 0 and bpm is None:
            i -= 1
            if self.timing_points[i].bpm > 0:
                bpm = self.timing_points[i].bpm

        return bpm or 120, timing_point

    def beat_duration(self, time):
        bpm, timing_point = self._timing(time)
        beat_duration = timing_point.bpm
        if beat_duration < 0:
            return bpm * -beat_duration / 100
        return beat_duration

    def visible_objects(self, time, count=None):
        objects = []
        preempt, _ = self.approach_rate()

        i = bsearch(self.effective_hit_objects, time, lambda obj: obj.time + preempt - obj.duration(self.beat_duration(obj.time),
                                                                                          self['SliderMultiplier']))
        i -= 5
        if i < 0:
            i = 0

        n = 0

        for obj in self.effective_hit_objects[i:]:
            obj_duration = obj.duration(self.beat_duration(obj.time), self['SliderMultiplier'])

            if time > obj.time + obj_duration:
                continue
            elif time < obj.time - preempt:
                break
            elif time < obj.time + obj_duration:
                objects.append(obj)

            n += 1
            if not count is None and n >= count:
                return objects

        return objects

    def background_name(self):
        """Returns the background image filename from the Events section, or None if not found"""
        if 'events' not in self.sections:
            return None
        
        for event in self.sections['events']:
            # format: [0, 0, "filename", 0, 0]
            if len(event) >= 3 and event[0] == 0 and event[1] == 0:
                filename = event[2]
                # Remove quotes if present
                if isinstance(filename, str) and filename.startswith('"') and filename.endswith('"'):
                    filename = filename[1:-1]
                return filename
        
        return None

def load(filename):
    with open(filename, 'r', encoding='utf8') as file:
        beatmap = Beatmap(file)
        return beatmap
