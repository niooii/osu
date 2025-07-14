from dataclasses import dataclass
from typing import List, Optional, Union
from enum import Enum, auto


class DataType(Enum):
    INT = "int"
    FLOAT = "float"
    DOUBLE = "double"
    USHORT = "ushort"
    SHORT = "short"
    BYTE = "byte"
    BOOLEAN = "boolean"
    STRING = "string"
    INT_LIST = "int_list"


# addresses taken from
# https://github.com/sakkyoi/reinforcement-osu/blob/master/util/OSUInjector.py
# and
# https://github.com/Piotrekol/ProcessMemoryDataFinder/blob/master/OsuMemoryDataProvider/OsuMemoryReader.cs


@dataclass
class MemorySignature:
    """Represents a memory signature with pattern and offset information"""
    name: str
    pattern: Optional[str] = None  # Hex pattern (eg. "F8 01 74 04 83 65")
    mask: Optional[str] = None  # Mask for pattern matching (eg. "xxx??xx????xx")
    parent_name: Optional[str] = None  # Name of parent signature
    base_offset: int = 0  # Base offset from signature location
    pointer_offsets: List[int] = None  # Chain of pointer offsets
    data_type: DataType = DataType.INT

    def __post_init__(self):
        if self.pointer_offsets is None:
            self.pointer_offsets = []


class OSUMemorySignatures:
    # Base signatures - these are the fundamental patterns we search for
    BASE_SIGNATURES = {
        "OsuBase": MemorySignature(
            name="OsuBase",
            pattern="F8 01 74 04 83 65",
            base_offset=0,
            pointer_offsets=[]
        ),

        "CurrentRuleset": MemorySignature(
            name="CurrentRuleset",
            pattern="C7 86 48 01 00 00 01 00 00 00 A1",
            base_offset=0xB,
            pointer_offsets=[0x4]
        ),

        "Mods": MemorySignature(
            name="Mods",
            pattern="C8 FF 00 00 00 00 00 81 0D 00 00 00 00 00 08 00 00",
            mask="xx?????xx????xxxx",
            base_offset=9,
            pointer_offsets=[0x0],
            data_type=DataType.INT
        ),

        "IsReplay": MemorySignature(
            name="IsReplay",
            pattern="8B FA B8 01 00 00 00",
            mask="xxxxxxx",
            base_offset=0x2A,
            pointer_offsets=[0x0],
            data_type=DataType.BOOLEAN
        ),

        "TourneyBase": MemorySignature(
            name="TourneyBase",
            pattern="7D 15 A1 00 00 00 08 85 C0",
            mask="xxx????xx",
            base_offset=-0xB,
            pointer_offsets=[]
        ),

        "CurrentSkinData": MemorySignature(
            name="CurrentSkinData",
            pattern="75 21 8B 1D",
            base_offset=4,
            pointer_offsets=[0x0, 0x0]
        )
    }

    # Game status and basic info
    GAME_STATUS = {
        "OsuStatus": MemorySignature(
            name="OsuStatus",
            parent_name="OsuBase",
            base_offset=-60,
            pointer_offsets=[0x0],
            data_type=DataType.INT
        ),

        "GameMode": MemorySignature(
            name="GameMode",
            parent_name="OsuBase",
            base_offset=-51,
            pointer_offsets=[0x0],
            data_type=DataType.INT
        ),

        "Retrys": MemorySignature(
            name="Retrys",
            parent_name="OsuBase",
            base_offset=-51,
            pointer_offsets=[0x8],
            data_type=DataType.INT
        ),

        "PlayTime": MemorySignature(
            name="PlayTime",
            parent_name="OsuBase",
            base_offset=100,
            pointer_offsets=[-16],
            data_type=DataType.INT
        )
    }

    # Current beatmap information
    BEATMAP_DATA = {
        "CurrentBeatmapData": MemorySignature(
            name="CurrentBeatmapData",
            parent_name="OsuBase",
            base_offset=-12,
            pointer_offsets=[0x0]
        ),

        "MapId": MemorySignature(
            name="MapId",
            parent_name="CurrentBeatmapData",
            pointer_offsets=[0xC8],
            data_type=DataType.INT
        ),

        "MapSetId": MemorySignature(
            name="MapSetId",
            parent_name="CurrentBeatmapData",
            pointer_offsets=[0xCC],
            data_type=DataType.INT
        ),

        "MapString": MemorySignature(
            name="MapString",
            parent_name="CurrentBeatmapData",
            pointer_offsets=[0x80],
            data_type=DataType.STRING
        ),

        "MapFolderName": MemorySignature(
            name="MapFolderName",
            parent_name="CurrentBeatmapData",
            pointer_offsets=[0x78],
            data_type=DataType.STRING
        ),

        "MapOsuFileName": MemorySignature(
            name="MapOsuFileName",
            parent_name="CurrentBeatmapData",
            pointer_offsets=[0x90],
            data_type=DataType.STRING
        ),

        "MapMd5": MemorySignature(
            name="MapMd5",
            parent_name="CurrentBeatmapData",
            pointer_offsets=[0x6C],
            data_type=DataType.STRING
        ),

        "MapAr": MemorySignature(
            name="MapAr",
            parent_name="CurrentBeatmapData",
            pointer_offsets=[44],
            data_type=DataType.FLOAT
        ),

        "MapCs": MemorySignature(
            name="MapCs",
            parent_name="CurrentBeatmapData",
            pointer_offsets=[48],
            data_type=DataType.FLOAT
        ),

        "MapHp": MemorySignature(
            name="MapHp",
            parent_name="CurrentBeatmapData",
            pointer_offsets=[52],
            data_type=DataType.FLOAT
        ),

        "MapOd": MemorySignature(
            name="MapOd",
            parent_name="CurrentBeatmapData",
            pointer_offsets=[56],
            data_type=DataType.FLOAT
        )
    }

    # Live gameplay data
    PLAY_DATA = {
        "PlayContainer": MemorySignature(
            name="PlayContainer",
            parent_name="CurrentRuleset",
            base_offset=0,
            pointer_offsets=[0x68]
        ),

        "Score": MemorySignature(
            name="Score",
            parent_name="CurrentRuleset",
            pointer_offsets=[0x100],
            data_type=DataType.INT
        ),

        "ScoreV2": MemorySignature(
            name="ScoreV2",
            parent_name="CurrentRuleset",
            pointer_offsets=[0x100],
            data_type=DataType.INT
        ),

        "Accuracy": MemorySignature(
            name="Accuracy",
            parent_name="PlayContainer",
            pointer_offsets=[72, 20],
            data_type=DataType.DOUBLE
        ),

        "Combo": MemorySignature(
            name="Combo",
            parent_name="PlayContainer",
            pointer_offsets=[56, 148],
            data_type=DataType.USHORT
        ),

        "ComboMax": MemorySignature(
            name="ComboMax",
            parent_name="PlayContainer",
            pointer_offsets=[56, 104],
            data_type=DataType.USHORT
        ),

        "Hit300": MemorySignature(
            name="Hit300",
            parent_name="PlayContainer",
            pointer_offsets=[56, 138],
            data_type=DataType.USHORT
        ),

        "Hit100": MemorySignature(
            name="Hit100",
            parent_name="PlayContainer",
            pointer_offsets=[56, 136],
            data_type=DataType.USHORT
        ),

        "Hit50": MemorySignature(
            name="Hit50",
            parent_name="PlayContainer",
            pointer_offsets=[56, 140],
            data_type=DataType.USHORT
        ),

        "HitGeki": MemorySignature(
            name="HitGeki",
            parent_name="PlayContainer",
            pointer_offsets=[56, 142],
            data_type=DataType.USHORT
        ),

        "HitKatsu": MemorySignature(
            name="HitKatsu",
            parent_name="PlayContainer",
            pointer_offsets=[56, 144],
            data_type=DataType.USHORT
        ),

        "HitMiss": MemorySignature(
            name="HitMiss",
            parent_name="PlayContainer",
            pointer_offsets=[56, 146],
            data_type=DataType.USHORT
        ),

        "PlayerHp": MemorySignature(
            name="PlayerHp",
            parent_name="PlayContainer",
            pointer_offsets=[64, 28],
            data_type=DataType.DOUBLE
        ),

        "PlayerHpSmoothed": MemorySignature(
            name="PlayerHpSmoothed",
            parent_name="PlayContainer",
            pointer_offsets=[64, 20],
            data_type=DataType.DOUBLE
        ),

        "PlayingMods": MemorySignature(
            name="PlayingMods",
            parent_name="PlayContainer",
            pointer_offsets=[56, 28],
            data_type=DataType.INT
        ),

        "PlayerName": MemorySignature(
            name="PlayerName",
            parent_name="PlayContainer",
            pointer_offsets=[56, 40],
            data_type=DataType.STRING
        ),

        "HitErrors": MemorySignature(
            name="HitErrors",
            parent_name="PlayContainer",
            pointer_offsets=[56, 56],
            data_type=DataType.INT_LIST
        ),

        "PlayingGameMode": MemorySignature(
            name="PlayingGameMode",
            parent_name="PlayContainer",
            pointer_offsets=[56, 104],
            data_type=DataType.INT
        )
    }

    # Skin information
    SKIN_DATA = {
        "CurrentSkinFolder": MemorySignature(
            name="CurrentSkinFolder",
            parent_name="CurrentSkinData",
            base_offset=68,
            data_type=DataType.STRING
        )
    }

    # Tournament mode data
    TOURNEY_DATA = {
        "TourneyIpcState": MemorySignature(
            name="TourneyIpcState",
            parent_name="TourneyBase",
            pointer_offsets=[4, 0x54],
            data_type=DataType.INT
        ),

        "TourneyLeftStars": MemorySignature(
            name="TourneyLeftStars",
            parent_name="TourneyBase",
            pointer_offsets=[4, 0x1C, 0x2C],
            data_type=DataType.INT
        ),

        "TourneyRightStars": MemorySignature(
            name="TourneyRightStars",
            parent_name="TourneyBase",
            pointer_offsets=[4, 0x20, 0x2C],
            data_type=DataType.INT
        ),

        "TourneyBO": MemorySignature(
            name="TourneyBO",
            parent_name="TourneyBase",
            pointer_offsets=[4, 0x20, 0x30],
            data_type=DataType.INT
        ),

        "TourneyStarsVisible": MemorySignature(
            name="TourneyStarsVisible",
            parent_name="TourneyBase",
            pointer_offsets=[4, 0x20, 0x38],
            data_type=DataType.BOOLEAN
        ),

        "TourneyScoreVisible": MemorySignature(
            name="TourneyScoreVisible",
            parent_name="TourneyBase",
            pointer_offsets=[4, 0x20, 0x39],
            data_type=DataType.BOOLEAN
        )
    }

    @classmethod
    def get_all_signatures(cls) -> dict:
        """Returns a dictionary containing all signatures from all categories"""
        all_sigs = {}
        all_sigs.update(cls.BASE_SIGNATURES)
        all_sigs.update(cls.GAME_STATUS)
        all_sigs.update(cls.BEATMAP_DATA)
        all_sigs.update(cls.PLAY_DATA)
        all_sigs.update(cls.SKIN_DATA)
        all_sigs.update(cls.TOURNEY_DATA)
        return all_sigs

    @classmethod
    def get_signature(cls, name: str) -> Optional[MemorySignature]:
        """Get a specific signature by name"""
        all_sigs = cls.get_all_signatures()
        return all_sigs.get(name)

    @classmethod
    def get_signatures_by_category(cls, category: str) -> dict:
        """Get all signatures from a specific category"""
        category_map = {
            "base": cls.BASE_SIGNATURES,
            "status": cls.GAME_STATUS,
            "beatmap": cls.BEATMAP_DATA,
            "play": cls.PLAY_DATA,
            "skin": cls.SKIN_DATA,
            "tourney": cls.TOURNEY_DATA
        }
        return category_map.get(category.lower(), {})

