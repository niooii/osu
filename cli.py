#!/usr/bin/env python3

import datetime
import os
from pathlib import Path

import arguably


@arguably.command
def mysql_to_csv(
    *,
    dump_dir: str = "dump"
):
    """Turn osu's monthly MySQL dumps into a csv file.
    
    Args:
        csv_file: Path to CSV file containing score data
        min_accuracy: Minimum accuracy threshold (0.0-1.0)
        min_pp: Minimum PP threshold
        min_beatmapset_id: Minimum beatmapset ID threshold
        max_downloads: Maximum number of replays to download
        score_type: Type of scores to download
        only: Filter for specific score types
        verbose: Enable verbose output
    """
    pass

@arguably.command
def download(
    *,
    user: int = 7562902,
    csv: str = "all_scores.csv",
    min_acc: float = 0.9,
    min_pp: float = 400.0,
    min_mapset: int = 100000,
    max_dl: int = 1000,
    type: str = "ALL",
    only: str = "ALL",
    verbose: bool = False,
):
    """Download osu! replay data for a specific user.
    
    Args:
        user: osu! user ID to download replays for
        csv: path to CSV file containing score data
        min_acc: minimum accuracy threshold (0.0-1.0)
        min_pp: minimum PP threshold
        min_mapset: minimum beatmapset ID threshold
        max_dl: maximum number of replays to download
        type: public scores to download ("none", "best", "firsts", "recent", "ALL")
        only: filter for specific score types ("FC" | "S" | "ALL")
        verbose: enable verbose output
    """
    import polars as pl
    import osu.downloader as downloader
    
    print(f"Downloading replays for user {user}")
    
    downloaded = 0

    if os.path.exists(csv):
        print(f"Loading scores from {csv}")
        df = pl.read_csv(csv)
        
        df_filtered = df.filter(
            (pl.col("accuracy") > min_acc)
            & (pl.col("user_id") == user)
            & (pl.col("beatmapset_id") > min_mapset)
            # TODO don't filter all pp for is not null, because we want loved maps
            # preferably as well?
            & (pl.col("pp").cast(pl.Float64, strict=False).is_not_null())
            & (pl.col("pp").cast(pl.Float64, strict=False) > min_pp)
        )
        
        print(f"found {len(df_filtered)} matching scores from {csv}")
        replays = downloader.download_from_df(df_filtered, user_id=user)
        if replays is not None:
            downloaded += len(replays)
    else:
        print(f"file {csv} not found")
    
    if type != "none":
        print("Downloading scores from osu website..")
        replays = downloader.download_user_scores(
            user_id=user, 
            score_type=type, 
            only=only, 
            max=max_dl
        )

        downloaded += len(replays)
    
    print(f"Downloaded {downloaded} replays")


@arguably.command
def create(
    *,
    user: int = 7562902,
    rp_path: str = ".data/replays-{user}",
    out: str = ".datasets",
    limit: int = 2000,
    verbose: bool = False,
    seq_len: int = None,
    rate: int = None,
):
    """Create training datasets from downloaded replays for a user.
    
    Args:
        user: osu! user ID to create dataset for
        rp_path: path to replay files (use {user} placeholder)
        out: directory to save dataset files
        limit: max number of replays to process
        verbose: enable verbose output
        seq_len: sequence length for chunking data (default: dataset.SEQ_LEN)
        rate: sampling rate in ms (default: dataset.SAMPLE_RATE)
    """
    import numpy as np
    import osu.dataset as dataset
    
    # Use default values if not provided
    if seq_len is None:
        seq_len = dataset.SEQ_LEN
    if rate is None:
        rate = dataset.SAMPLE_RATE
    
    actual_replay_path = rp_path.format(user=user)
    
    print(f"Creating dataset for user {user}")
    print(f"Replay path: {actual_replay_path}")
    print(f"Replay limit: {limit}")
    print(f"Sequence length: {seq_len}")
    print(f"Sample rate: {rate}ms")
    
    obj_dataset = dataset.user_replay_mapping_from_cache(
        user_id=user, 
        replay_path=actual_replay_path, 
        limit=limit
    )
    
    print(f"Got {len(obj_dataset)} good replays")
    
    input_data = dataset.input_data(obj_dataset, verbose=verbose, seq_len=seq_len, sample_rate=rate)
    output_data = dataset.target_data(obj_dataset, verbose=verbose, seq_len=seq_len, sample_rate=rate)
    
    # resize into sequence len
    xs = np.reshape(input_data.values, (-1, seq_len, len(dataset.INPUT_FEATURES)))
    ys = np.reshape(output_data.values, (-1, seq_len, len(dataset.OUTPUT_FEATURES)))
    
    Path(out).mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    
    input_file = f'{out}/xs_{len(obj_dataset)}_{timestamp}.npy'
    output_file = f'{out}/ys_{len(obj_dataset)}_{timestamp}.npy'
    
    np.save(input_file, xs)
    np.save(output_file, ys)
    
    print(f"Saved input data: {input_file} (shape: {xs.shape})")
    print(f"Saved output data: {output_file} (shape: {ys.shape})")


@arguably.command
def preview(
    *,
    rp: str,
    bm: str,
    audio: str = None,
):
    """Preview a replay with its beatmap from .osr and .osu files.
    
    Args:
        rp: Path to .osr replay file
        bm: Path to .osu beatmap file  
        audio: Path to audio file (optional)
    """
    import osu.rulesets.replay as replay_module
    from osu.preview.preview import preview_replay
    
    print(f"Loading replay: {rp}")
    print(f"Loading beatmap: {bm}")
    if audio:
        print(f"Loading audio: {audio}")
    
    replay = replay_module.load(rp)
    
    preview_replay(replay, bm, audio)


@arguably.command
def preview_data(
    *,
    xs: str,
    ys: str,
    rate: int = None,
):
    """Preview training data from xs and ys files.
    
    Args:
        xs: Path to input data .npy file (map frames)
        ys: Path to output data .npy file (replay frames)
        rate: sampling rate in ms (default: dataset.SAMPLE_RATE)
    """
    import numpy as np
    import osu.dataset as dataset
    from osu.preview.preview import preview_training_data
    
    if rate is None:
        rate = dataset.SAMPLE_RATE
    
    print(f"Loading data from {xs} and {ys}")
    print(f"Sample rate: {rate}ms")
    
    xs = np.load(xs)
    ys = np.load(ys)
    
    print(f"xs shape: {xs.shape}")
    print(f"ys shape: {ys.shape}")
    
    preview_training_data(xs, ys, sample_rate=rate)


if __name__ == "__main__":
    arguably.run()
