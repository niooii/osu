from enum import IntFlag, auto

class Mods(IntFlag):
    NONE = 0
    NO_FAIL = 1                    # Bit 0
    EASY = 2                       # Bit 1
    TOUCH_DEVICE = 4               # Bit 2 - Replaces unused NoVideo mod
    HIDDEN = 8                     # Bit 3
    HARD_ROCK = 16                 # Bit 4
    SUDDEN_DEATH = 32              # Bit 5
    DOUBLE_TIME = 64               # Bit 6
    RELAX = 128                    # Bit 7
    HALF_TIME = 256                # Bit 8
    NIGHTCORE = 512                # Bit 9 - Always used with DT: 512 + 64 = 576, Replaces unused Taiko mod
    FLASHLIGHT = 1024              # Bit 10
    AUTOPLAY = 2048                # Bit 11
    SPUN_OUT = 4096                # Bit 12
    RELAX2 = 8192                  # Bit 13 - Autopilot
    PERFECT = 16384                # Bit 14
    KEY4 = 32768                   # Bit 15
    KEY5 = 65536                   # Bit 16
    KEY6 = 131072                  # Bit 17
    KEY7 = 262144                  # Bit 18
    KEY8 = 524288                  # Bit 19
    FADE_IN = 1048576              # Bit 20
    RANDOM = 2097152               # Bit 21
    LAST_MOD = 4194304             # Bit 22
    CINEMA = 8388608               # Bit 23
    TARGET_PRACTICE = 8388608      # Bit 23 - osu!cuttingedge only
    KEY9 = 16777216                # Bit 24
    COOP = 33554432                # Bit 25
    KEY1 = 67108864                # Bit 26
    KEY3 = 134217728               # Bit 27
    KEY2 = 268435456               # Bit 28
    SCORE_V2 = 536870912           # Bit 29
    MIRROR = 1073741824            # Bit 30

    STD_GAMEPLAY_AFFECTING = EASY | HARD_ROCK | DOUBLE_TIME | HALF_TIME