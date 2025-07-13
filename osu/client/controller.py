from typing import List

import win32gui, win32process
from pymem import pattern, Pymem
import os

from osu.client.mem import OSUMemorySignatures, MemorySignature, DataType


class GameNamespace:
    def __init__(self, controller):
        self._controller = controller
    
    def status(self) -> int:
        return self._controller.read_value("OsuStatus")
    
    def mode(self) -> int:
        return self._controller.read_value("GameMode")
    
    def retrys(self) -> int:
        return self._controller.read_value("Retrys")
    
    def play_time(self) -> int:
        return self._controller.read_value("PlayTime")
    
    def mods(self) -> int:
        return self._controller.read_value("Mods")
    
    def is_replay(self) -> bool:
        return self._controller.read_value("IsReplay")


class BeatmapNamespace:
    def __init__(self, controller):
        self._controller = controller
    
    def id(self) -> int:
        return self._controller.read_value("MapId")
    
    def set_id(self) -> int:
        return self._controller.read_value("MapSetId")
    
    def string(self) -> str:
        return self._controller.read_value("MapString")
    
    def folder_name(self) -> str:
        return self._controller.read_value("MapFolderName")
    
    def osu_file_name(self) -> str:
        return self._controller.read_value("MapOsuFileName")
    
    def md5(self) -> str:
        return self._controller.read_value("MapMd5")
    
    def ar(self) -> float:
        return self._controller.read_value("MapAr")
    
    def cs(self) -> float:
        return self._controller.read_value("MapCs")
    
    def hp(self) -> float:
        return self._controller.read_value("MapHp")
    
    def od(self) -> float:
        return self._controller.read_value("MapOd")


class PlayNamespace:
    def __init__(self, controller):
        self._controller = controller
    
    def score(self) -> int:
        return self._controller.read_value("Score")
    
    def score_v2(self) -> int:
        return self._controller.read_value("ScoreV2")
    
    def accuracy(self) -> float:
        return self._controller.read_value("Accuracy")
    
    def combo(self) -> int:
        return self._controller.read_value("Combo")
    
    def combo_max(self) -> int:
        return self._controller.read_value("ComboMax")
    
    def hit_300(self) -> int:
        return self._controller.read_value("Hit300")
    
    def hit_100(self) -> int:
        return self._controller.read_value("Hit100")
    
    def hit_50(self) -> int:
        return self._controller.read_value("Hit50")
    
    def hit_geki(self) -> int:
        return self._controller.read_value("HitGeki")
    
    def hit_katsu(self) -> int:
        return self._controller.read_value("HitKatsu")
    
    def hit_miss(self) -> int:
        return self._controller.read_value("HitMiss")
    
    def player_hp(self) -> float:
        return self._controller.read_value("PlayerHp")
    
    def player_hp_smoothed(self) -> float:
        return self._controller.read_value("PlayerHpSmoothed")
    
    def playing_mods(self) -> int:
        return self._controller.read_value("PlayingMods")
    
    def player_name(self) -> str:
        return self._controller.read_value("PlayerName")
    
    def playing_game_mode(self) -> int:
        return self._controller.read_value("PlayingGameMode")


class SkinNamespace:
    def __init__(self, controller):
        self._controller = controller
    
    def current_folder(self) -> str:
        return self._controller.read_value("CurrentSkinFolder")


class TourneyNamespace:
    def __init__(self, controller):
        self._controller = controller
    
    def ipc_state(self) -> int:
        return self._controller.read_value("TourneyIpcState")
    
    def left_stars(self) -> int:
        return self._controller.read_value("TourneyLeftStars")
    
    def right_stars(self) -> int:
        return self._controller.read_value("TourneyRightStars")
    
    def bo(self) -> int:
        return self._controller.read_value("TourneyBO")
    
    def stars_visible(self) -> bool:
        return self._controller.read_value("TourneyStarsVisible")
    
    def score_visible(self) -> bool:
        return self._controller.read_value("TourneyScoreVisible")


APP_DATA = os.getenv("APPDATA")
OSU_PATH = f'{APP_DATA}/../Local/osu!'
print(f'osu! path: {OSU_PATH}')


def _pattern_converter(patt: str) -> bytes:
    # Split the pattern into individual hex pairs (or wildcards)
    hex_parts = patt.split(' ')
    result = b''

    for part in hex_parts:
        if part == '??':
            result += b'.'
        else:
            try:
                byte_value = int(part, 16)
                result += bytes([byte_value])
            except ValueError:
                raise ValueError(f"Invalid hex value in pattern: {part}")

    return result


class OSUController:
    def __init__(self, osu_window_title: str = "osu!"):
        # search for osu pid
        # hwnd = win32gui.FindWindow(None, osu_window_title)
        #
        # if hwnd is None or hwnd == 0:
        #     raise Exception(f"Cannot find osu window. Given name: {osu_window_title}")
        #
        # threadid, pid = win32process.GetWindowThreadProcessId(hwnd)

        pm = Pymem("osu!.exe")
        self._pm = pm
        self._signatures = OSUMemorySignatures.get_all_signatures()
        self.base_addresses = dict()
        self._find_base_patterns()
        
        # Initialize namespaces
        self.game = GameNamespace(self)
        self.beatmap = BeatmapNamespace(self)
        self.play = PlayNamespace(self)
        self.skin = SkinNamespace(self)
        self.tourney = TourneyNamespace(self)

        # print(f'Found osu with pid {pid}')
        # self.pid = pid

    def _find_base_patterns(self):
        """Find all base pattern addresses and cache them"""
        for name, sig in self._signatures.items():
            if sig.pattern:  # This is a base signature with its own pattern
                pattern_bytes = _pattern_converter(sig.pattern)
                address = pattern.pattern_scan_all(self._pm.process_handle, pattern_bytes)
                self.base_addresses[name] = address

    def _get_sig_addr(self, signature_name: str) -> int:
        """
        Translation of ResolveChainOfPointers() from Sig.cs
        https://github.com/Piotrekol/ProcessMemoryDataFinder/blob/master/ProcessMemoryDataFinder/API/Sig.cs
        """
        sig = self._signatures[signature_name]
        
        # line 88-89: if (_resolvedAddress != IntPtr.Zero) return _resolvedAddress;
        # TODO caching

        addr = 0
        
        if sig.parent_name is not None:
            addr = self._get_sig_addr(sig.parent_name)
            addr += sig.base_offset
        
        if sig.pattern is not None:
            if sig.name in self.base_addresses:
                addr = self.base_addresses[sig.name]
            else:
                return 0
        
        if addr != 0:
            addr = self._resolve_chain_of_pointers(addr, sig.pointer_offsets)
        
        return addr
    
    def _resolve_chain_of_pointers(self, base_address: int, pointer_offsets: list) -> int:
        """
        Translation of ResolveChainOfPointers() from Sig.cs lines 122-136
        """

        if not pointer_offsets or len(pointer_offsets) == 0:
            return base_address
        
        pointer = base_address
        pointer = self._pm.read_int(pointer)
        
        for i in range(len(pointer_offsets) - 1):
            offset = pointer_offsets[i]
            pointer = self._pm.read_int(pointer + offset)
        
        pointer = pointer + pointer_offsets[len(pointer_offsets) - 1]
        
        return pointer

    def read_value(self, sig_name: str):
        """Generic method to read any value using signature name"""
        sig = OSUMemorySignatures.get_signature(sig_name)
        if not sig:
            raise ValueError(f"Unknown signature: {sig_name}")

        addr = self._get_sig_addr(sig_name)

        # addr = self._pm.read_int(addr)

        if sig.data_type is not None:
            dtype = sig.data_type
            if dtype == DataType.INT:
                return self._pm.read_int(addr)
            elif dtype == DataType.DOUBLE:
                return self._pm.read_double(addr)
            elif dtype == DataType.USHORT:
                return self._pm.read_ushort(addr)
            elif dtype == DataType.SHORT:
                return self._pm.read_short(addr)
            elif dtype == DataType.FLOAT:
                return self._pm.read_float(addr)
            elif dtype == DataType.STRING:
                # this is a pointer to a string
                str_addr = self._pm.read_int(addr)
                # .NET string has the length at offset +4, and actual UTF-16 contents at +8
                length = self._pm.read_int(str_addr + 4)
                string_bytes = self._pm.read_bytes(str_addr + 8, length * 2)
                return string_bytes.decode('utf-16le')
            # elif read == 'int_list':
            #     return self.read_int_list(address)
            else:
                raise ValueError(f'Invalid read type: {dtype}')


