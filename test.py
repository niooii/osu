#!/usr/bin/env python3
"""Quick test script for osu-nn models.

Usage:
    python test.py assets/rally_map.osu
    python test.py assets/rally_map.osu --audio assets/rally_song.mp3
    python test.py assets/rally_map.osu --model .trained/replayvae_most_recent.pt
"""

import argparse
import os

import numpy as np
import torch

import osu.dataset as dataset
import osu.rulesets.beatmap as bm
import osu.rulesets.replay as osr
from models.base import OsuModel
from osu.rulesets.mods import Mods
from osu.preview.preview import preview_replay_raw, export_replay_video

MOD_ABBREVIATIONS = {
    "nf": Mods.NO_FAIL,
    "ez": Mods.EASY,
    "hd": Mods.HIDDEN,
    "hr": Mods.HARD_ROCK,
    "sd": Mods.SUDDEN_DEATH,
    "dt": Mods.DOUBLE_TIME,
    "rx": Mods.RELAX,
    "ht": Mods.HALF_TIME,
    "nc": Mods.NIGHTCORE | Mods.DOUBLE_TIME,
    "fl": Mods.FLASHLIGHT,
    "so": Mods.SPUN_OUT,
    "pf": Mods.PERFECT,
}


def parse_mods(mod_string):
    mod_string = mod_string.lower()
    if len(mod_string) % 2 != 0:
        raise argparse.ArgumentTypeError(f"Mod string must be pairs of 2 characters, got '{mod_string}'")
    result = Mods.NONE
    for i in range(0, len(mod_string), 2):
        abbr = mod_string[i:i+2]
        if abbr not in MOD_ABBREVIATIONS:
            raise argparse.ArgumentTypeError(f"Unknown mod '{abbr}'")
        result |= MOD_ABBREVIATIONS[abbr]
    return result


def main():
    parser = argparse.ArgumentParser(description="Test an osu-nn model on a beatmap")
    parser.add_argument("beatmap", help="Path to .osu beatmap file")
    parser.add_argument("--model", "-m", default=".trained/recent.pt",
                        help="Model checkpoint (default: .trained/recent.pt)")
    parser.add_argument("--audio", "-a",
                        help="Audio file (auto-detected from beatmap name if not given)")
    parser.add_argument("--save", "-s",
                        help="Save generated replay to .npy instead of previewing")
    parser.add_argument("--stats-window", "-w", type=int, default=100,
                        help="Smoothing window in ms for cursor stats (default: 100)")
    parser.add_argument("--trail", "-t", type=int, default=400,
                        help="Trail length in ms (default: 400)")
    parser.add_argument("--replay", "-r",
                        help="Reference .osr replay to overlay (purple cursor)")
    parser.add_argument("--mods", type=parse_mods, default=Mods.NONE,
                        help="Mods as 2-char pairs (e.g. hdhrdt, hdhr, dtfl)")
    parser.add_argument("--export", "-e",
                        help="Export as video to this path instead of previewing (requires ffmpeg)")
    args = parser.parse_args()

    # Auto-detect audio from naming convention ({name}_map.osu -> {name}_song.mp3)
    audio = args.audio
    if audio is None:
        candidate = args.beatmap.replace("_map.osu", "_song.mp3")
        if candidate != args.beatmap and os.path.exists(candidate):
            audio = candidate

    model = OsuModel.auto_load(args.model)

    beatmap = bm.load(args.beatmap)
    if args.mods:
        beatmap.apply_mods(args.mods)
    data = dataset.input_data(beatmap)
    data = np.reshape(data.values, (-1, dataset.SEQ_LEN, len(dataset.INPUT_FEATURES)))
    data = torch.FloatTensor(data)

    output = model.generate(data)
    output = np.concatenate(output)

    # Pad to [x, y, k1, k2] if model only outputs position
    if output.shape[-1] == 2:
        output = np.pad(output, ((0, 0), (0, 2)), mode="constant")

    # Load reference replay if provided
    ref_data = None
    if args.replay:
        replay = osr.load(args.replay)
        ref_chunks = dataset.target_data_single(beatmap, replay)
        ref_data = np.concatenate([np.array(chunk) for chunk in ref_chunks])

    if args.save:
        np.save(args.save, output)
        print(f"Saved to {args.save}")
    elif args.export:
        export_replay_video(output, args.beatmap, args.export, mods=args.mods,
                            stats_window_ms=args.stats_window, trail_ms=args.trail,
                            reference_replay=ref_data)
    else:
        preview_replay_raw(output, beatmap_path=args.beatmap, mods=args.mods,
                           audio_file=audio, stats_window_ms=args.stats_window,
                           trail_ms=args.trail, reference_replay=ref_data)


if __name__ == "__main__":
    main()
