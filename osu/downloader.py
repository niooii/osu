import hashlib
import io
import json
import os
import re
import shutil
import time
import zipfile
from typing import Callable, Literal

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

import osu.rulesets.beatmap as beatmap
import osu.rulesets.replay as replay
from osu.rulesets.core import OSU_PATH

load_dotenv()

osu_session = os.getenv("OSU_SESSION")

os.makedirs("../.data/replays", exist_ok=True)


# TODO we can make this async and then batch the donwloads, but nah im too lazy
# TODO add cache to map beatmaps to the replays
def get_replay(
    score_id: int, map_md5: str, verbose: bool = False
) -> (replay.Replay, int):
    save_path = f".data/replays/{score_id}.osr"
    if os.path.exists(save_path):
        return (replay.load(save_path), score_id)

    url = f"https://osu.ppy.sh/scores/{score_id}/download"
    retry_times = 0
    headers = {
        "Cookie": f"osu_session={osu_session}",
        "Referer": f"https://osu.ppy.sh/scores/{score_id}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:141.0) Gecko/20100101 Firefox/141.0",
    }
    response = requests.get(url, headers=headers)
    if response.status_code // 100 != 2:
        if response.status_code == 404:
            return None
        verbose and print(
            f"osu responded with status code {response.status_code} for replay {score_id}, waiting a few secs"
        )
        while retry_times < 5 and response.status_code // 100 != 2:
            time.sleep(4 * (retry_times + 1))
            response = requests.get(url, headers=headers)
            if response.status_code == 404:
                return None
            if response.status_code // 100 == 2:
                break
            verbose and print(
                f"osu responded with status code {response.status_code} for replay {score_id}, waiting a few secs"
            )
            retry_times += 1
        if retry_times == 5:
            print(
                f"Error: Retried 5 or so times without a 200 status code for replay {score_id}, stopping. check osu session."
            )
            return None

    rp = replay.Replay(io.BytesIO(response.content))
    if rp.map_md5 != map_md5:
        verbose and print(f"replay {score_id}: mismatching hash.")
        return None

    # save to disk cache
    with open(save_path, "wb") as file:
        file.write(response.content)

    return (rp, score_id)


def download_t50_replays(
    beatmap: beatmap.Beatmap,
    cache_map_path: str,
    map_md5: str,
    max: int = 50,
    only: Literal["FC", "S", "ALL"] = "S",
    verbose: bool = False,
) -> list[replay.Replay]:
    retry_times = 0
    beatmap_id = beatmap.beatmap_id()
    t50_url = f"https://osu.ppy.sh/beatmaps/{beatmap_id}/scores?mode=osu&type=global"

    if not os.path.exists(".data/replays/"):
        os.makedirs(".data/replays/")

    response = requests.get(t50_url)
    if response.status_code // 100 != 2:
        if response.status_code == 404:
            return None
        verbose and print(
            f"osu responded with status code {response.status_code}, waiting a few secs"
        )

        while retry_times < 5 and response.status_code // 100 != 2:
            time.sleep(4 * (retry_times + 1))
            response = requests.get(t50_url)
            if response.status_code == 404:
                return None
            if response.status_code // 100 == 2:
                break
            verbose and print(
                f"osu responded with status code {response.status_code}, waiting a few secs"
            )
            retry_times += 1

        if retry_times == 5:
            print(
                "Error: Retried 5 or so times without a 200 status code, stopping. check osu session."
            )
            return None

    response = json.loads(response.text)
    scores = response["scores"]

    # we also filter out the ones that had no replay obviously
    replay_ids = []
    for score in scores:
        rank = score["rank"]
        if only == "S" and not rank.startswith("S"):
            continue
        elif only == "FC" and not score["is_perfect_combo"]:
            continue

        if score["has_replay"]:
            replay_ids.append(score["id"])

    replay_ids = replay_ids[0:max]

    # download replays
    replays = [get_replay(id, map_md5, verbose) for id in replay_ids]
    replays = [t for t in replays if t is not None]

    if replays is None:
        verbose and print(f"Failed to download replays for map id {beatmap_id}")
        return None

    for replay, id in replays:
        meta_path = f".data/replays/{id}.meta"
        if not os.path.exists(meta_path):
            with open(meta_path, "w") as f:
                f.write(json.dumps({"map": cache_map_path}))

    replays = [replay for (replay, id) in replays]

    return replays


def download_local_mapset_t50_replays(
    mapset_folder: str,
    verbose: bool = False,
    filter: None | Callable[[beatmap.Beatmap], bool] = None,
    **kwargs,
) -> pd.DataFrame:
    mapset_folder_name = mapset_folder
    mapset_folder = f"{OSU_PATH}/Songs/{mapset_folder}"
    files = os.listdir(mapset_folder)
    maps = []
    replays = []

    for path in files:
        if not path.endswith(".osu"):
            continue

        verbose and print(f"loading {path}")
        map_path = f"{mapset_folder}/{path}"
        mp = beatmap.load(map_path)

        if mp.beatmap_id() is None:
            continue

        if filter is not None:
            if not filter(mp):
                continue

        cache_map_path = f".data/songs/{mapset_folder_name}/{mp.beatmap_id()}_{path}"

        with open(map_path, "rb") as f:
            hash = hashlib.md5(f.read()).hexdigest()

        downloaded_replays = download_t50_replays(
            mp, cache_map_path=cache_map_path, map_md5=hash, verbose=verbose, **kwargs
        )
        if downloaded_replays is None:
            continue

        for rp in downloaded_replays:
            maps.append(mp)
            replays.append(rp)

        if len(downloaded_replays) > 0:
            if not os.path.exists(cache_map_path):
                os.makedirs(os.path.dirname(cache_map_path), exist_ok=True)
                shutil.copyfile(map_path, cache_map_path)

    return pd.DataFrame(list(zip(replays, maps)), columns=["replay", "beatmap"])


import tqdm


# call download_local_mapset_t50_replays for all in the list
def download_mapsets(
    mapset_folders: list[str],
    return_dataframe: bool = False,
    verbose: bool = False,
    **kwargs,
):
    dataframe = pd.DataFrame(columns=["replay", "beatmap"])

    pbar = tqdm.tqdm(
        mapset_folders, total=len(mapset_folders), desc="Downloading replays"
    )
    for mapset_folder in pbar:
        # if the mapset folder doesn't start with a number, chances are it won't have useful metadata in the maps anyways
        pattern = r"^(\d+) "
        match = re.match(pattern, mapset_folder)
        if not match or int(match.group(1)) < 100000:
            verbose and print(f"Skipping folder {mapset_folder}")
            continue
        pbar.set_description(f"Downloading replays from maps in {mapset_folder}")
        df = download_local_mapset_t50_replays(mapset_folder, verbose=verbose, **kwargs)
        if df is not None:
            verbose and print(f"Got {len(df)} replays that meet the criteria.")
            if return_dataframe:
                dataframe = pd.concat([dataframe, df])

    if not return_dataframe:
        return None
    else:
        return dataframe


def get_all_mapset_folders_on_disk() -> list[str]:
    return os.listdir(f"{OSU_PATH}/Songs/")


def get_user_replay(
    score_id: int, map_md5: str, user_id: int, verbose: bool = False
) -> (replay.Replay, int):
    save_path = f".data/replays-{user_id}/{score_id}.osr"
    if os.path.exists(save_path):
        return (replay.load(save_path), score_id)

    url = f"https://osu.ppy.sh/scores/{score_id}/download"
    retry_times = 0
    headers = {
        "Cookie": f"osu_session={osu_session}",
        "Referer": f"https://osu.ppy.sh/scores/{score_id}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:141.0) Gecko/20100101 Firefox/141.0",
    }
    response = requests.get(url, headers=headers)
    if response.status_code // 100 != 2:
        if response.status_code == 404:
            return None
        verbose and print(
            f"osu responded with status code {response.status_code} for replay {score_id}, waiting a few secs"
        )
        while retry_times < 5 and response.status_code // 100 != 2:
            time.sleep(4 * (retry_times + 1))
            response = requests.get(url, headers=headers)
            if response.status_code == 404:
                return None
            if response.status_code // 100 == 2:
                break
            verbose and print(
                f"osu responded with status code {response.status_code} for replay {score_id}, waiting a few secs"
            )
            retry_times += 1
        if retry_times == 5:
            print(
                f"Error: Retried 5 or so times without a 200 status code for replay {score_id}, stopping. check osu session."
            )
            return None

    rp = replay.Replay(io.BytesIO(response.content))
    if rp.map_md5 != map_md5:
        verbose and print(f"replay {score_id}: mismatching hash.")
        return None

    # save to disk cache
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as file:
        file.write(response.content)

    return (rp, score_id)


def download_user_best_scores(
    user_id: int,
    max: int = 10000,
    only: Literal["FC", "S", "ALL"] = "S",
    verbose: bool = False,
) -> list[replay.Replay]:
    retry_times = 0
    scores_url = (
        f"https://osu.ppy.sh/api/v2/users/{user_id}/scores/best?mode=osu&limit={max}"
    )

    if not os.path.exists(f".data/replays-{user_id}/"):
        os.makedirs(f".data/replays-{user_id}/")

    response = requests.get(scores_url)
    if response.status_code // 100 != 2:
        if response.status_code == 404:
            return None
        verbose and print(
            f"osu responded with status code {response.status_code}, waiting a few secs"
        )

        while retry_times < 5 and response.status_code // 100 != 2:
            time.sleep(4 * (retry_times + 1))
            response = requests.get(scores_url)
            if response.status_code == 404:
                return None
            if response.status_code // 100 == 2:
                break
            verbose and print(
                f"osu responded with status code {response.status_code}, waiting a few secs"
            )
            retry_times += 1

        if retry_times == 5:
            print("Error: Retried 5 or so times without a 200 status code, stopping.")
            return None

    scores = json.loads(response.text)

    # filter scores based on criteria
    replay_ids = []
    for score in scores:
        rank = score["rank"]
        if only == "S" and not rank.startswith("S"):
            continue
        elif only == "FC" and not score["is_perfect_combo"]:
            continue

        if score["has_replay"]:
            replay_ids.append(
                (
                    score["id"],
                    score["beatmap"]["checksum"],
                    score["beatmap"]["beatmapset_id"],
                    score["beatmap"]["id"],
                )
            )

    # download beatmapsets and replays
    replays = []
    score_metadata = {}  # Store beatmapset_id for each score_id

    for score_id, map_md5, beatmapset_id, beatmap_id in replay_ids:
        score_metadata[score_id] = {
            "beatmapset_id": beatmapset_id,
            "beatmap_id": beatmap_id,
        }

        # Download beatmapset (function handles caching automatically)
        beatmapset_path = download_beatmapset(beatmapset_id, verbose=verbose)
        if not beatmapset_path and verbose:
            print(f"Failed to download beatmapset {beatmapset_id} for score {score_id}")

        # Download replay
        replay_result = get_user_replay(score_id, map_md5, user_id, verbose)
        if replay_result is not None:
            replays.append(replay_result)

    replays = [t for t in replays if t is not None]

    if replays is None:
        verbose and print(f"Failed to download replays for user {user_id}")
        return None

    for replay, id in replays:
        meta_path = f".data/replays-{user_id}/{id}.meta"
        if not os.path.exists(meta_path):
            with open(meta_path, "w") as f:
                metadata = score_metadata.get(id, {})
                f.write(
                    json.dumps(
                        {
                            "user_id": user_id,
                            "score_type": "best",
                            "beatmapset_id": metadata.get("beatmapset_id"),
                            "beatmap_id": metadata.get("beatmap_id"),
                        }
                    )
                )

    replays = [replay for (replay, id) in replays]

    return replays


def download_user_recent_scores(
    user_id: int,
    max: int = 10000,
    only: Literal["FC", "S", "ALL"] = "S",
    verbose: bool = False,
) -> list[replay.Replay]:
    retry_times = 0
    scores_url = (
        f"https://osu.ppy.sh/api/v2/users/{user_id}/scores/recent?mode=osu&limit={max}"
    )

    if not os.path.exists(f".data/replays-{user_id}/"):
        os.makedirs(f".data/replays-{user_id}/")

    response = requests.get(scores_url)
    if response.status_code // 100 != 2:
        if response.status_code == 404:
            return None
        verbose and print(
            f"osu responded with status code {response.status_code}, waiting a few secs"
        )

        while retry_times < 5 and response.status_code // 100 != 2:
            time.sleep(4 * (retry_times + 1))
            response = requests.get(scores_url)
            if response.status_code == 404:
                return None
            if response.status_code // 100 == 2:
                break
            verbose and print(
                f"osu responded with status code {response.status_code}, waiting a few secs"
            )
            retry_times += 1

        if retry_times == 5:
            print("Error: Retried 5 or so times without a 200 status code, stopping.")
            return None

    scores = json.loads(response.text)

    # filter scores based on criteria
    replay_ids = []
    for score in scores:
        rank = score["rank"]
        if only == "S" and not rank.startswith("S"):
            continue
        elif only == "FC" and not score["is_perfect_combo"]:
            continue

        if score["has_replay"]:
            replay_ids.append(
                (
                    score["id"],
                    score["beatmap"]["checksum"],
                    score["beatmap"]["beatmapset_id"],
                    score["beatmap"]["id"],
                )
            )

    # download beatmapsets and replays
    replays = []
    score_metadata = {}  # Store beatmapset_id for each score_id

    for score_id, map_md5, beatmapset_id, beatmap_id in replay_ids:
        score_metadata[score_id] = {
            "beatmapset_id": beatmapset_id,
            "beatmap_id": beatmap_id,
        }

        # Download beatmapset (function handles caching automatically)
        beatmapset_path = download_beatmapset(beatmapset_id, verbose=verbose)
        if not beatmapset_path and verbose:
            print(f"Failed to download beatmapset {beatmapset_id} for score {score_id}")

        # Download replay
        replay_result = get_user_replay(score_id, map_md5, user_id, verbose)
        if replay_result is not None:
            replays.append(replay_result)

    replays = [t for t in replays if t is not None]

    if replays is None:
        verbose and print(f"Failed to download replays for user {user_id}")
        return None

    for replay, id in replays:
        meta_path = f".data/replays-{user_id}/{id}.meta"
        if not os.path.exists(meta_path):
            with open(meta_path, "w") as f:
                metadata = score_metadata.get(id, {})
                f.write(
                    json.dumps(
                        {
                            "user_id": user_id,
                            "score_type": "recent",
                            "beatmapset_id": metadata.get("beatmapset_id"),
                            "beatmap_id": metadata.get("beatmap_id"),
                        }
                    )
                )

    replays = [replay for (replay, id) in replays]

    return replays


def download_user_first_scores(
    user_id: int,
    max: int = 10000,
    only: Literal["FC", "S", "ALL"] = "S",
    verbose: bool = False,
) -> list[replay.Replay]:
    retry_times = 0
    scores_url = (
        f"https://osu.ppy.sh/api/v2/users/{user_id}/scores/firsts?mode=osu&limit={max}"
    )

    if not os.path.exists(f".data/replays-{user_id}/"):
        os.makedirs(f".data/replays-{user_id}/")

    response = requests.get(scores_url)
    if response.status_code // 100 != 2:
        if response.status_code == 404:
            return None
        verbose and print(
            f"osu responded with status code {response.status_code}, waiting a few secs"
        )

        while retry_times < 5 and response.status_code // 100 != 2:
            time.sleep(4 * (retry_times + 1))
            response = requests.get(scores_url)
            if response.status_code == 404:
                return None
            if response.status_code // 100 == 2:
                break
            verbose and print(
                f"osu responded with status code {response.status_code}, waiting a few secs"
            )
            retry_times += 1

        if retry_times == 5:
            print("Error: Retried 5 or so times without a 200 status code, stopping.")
            return None

    scores = json.loads(response.text)

    # filter scores based on criteria
    replay_ids = []
    for score in scores:
        rank = score["rank"]
        if only == "S" and not rank.startswith("S"):
            continue
        elif only == "FC" and not score["is_perfect_combo"]:
            continue

        if score["has_replay"]:
            replay_ids.append(
                (
                    score["id"],
                    score["beatmap"]["checksum"],
                    score["beatmap"]["beatmapset_id"],
                    score["beatmap"]["id"],
                )
            )

    # download beatmapsets and replays
    replays = []
    score_metadata = {}  # Store beatmapset_id for each score_id

    for score_id, map_md5, beatmapset_id, beatmap_id in replay_ids:
        score_metadata[score_id] = {
            "beatmapset_id": beatmapset_id,
            "beatmap_id": beatmap_id,
        }

        # Download beatmapset (function handles caching automatically)
        beatmapset_path = download_beatmapset(beatmapset_id, verbose=verbose)
        if not beatmapset_path and verbose:
            print(f"Failed to download beatmapset {beatmapset_id} for score {score_id}")

        # Download replay
        replay_result = get_user_replay(score_id, map_md5, user_id, verbose)
        if replay_result is not None:
            replays.append(replay_result)

    replays = [t for t in replays if t is not None]

    if replays is None:
        verbose and print(f"Failed to download replays for user {user_id}")
        return None

    for replay, id in replays:
        meta_path = f".data/replays-{user_id}/{id}.meta"
        if not os.path.exists(meta_path):
            with open(meta_path, "w") as f:
                metadata = score_metadata.get(id, {})
                f.write(
                    json.dumps(
                        {
                            "user_id": user_id,
                            "score_type": "firsts",
                            "beatmapset_id": metadata.get("beatmapset_id"),
                            "beatmap_id": metadata.get("beatmap_id"),
                        }
                    )
                )

    replays = [replay for (replay, id) in replays]

    return replays


def download_all_user_scores(
    user_id: int,
    max: int = 10000,
    only: Literal["FC", "S", "ALL"] = "S",
    verbose: bool = False,
) -> dict:
    """Download all user scores (best, recent, firsts) with the same parameters"""
    results = {}

    if verbose:
        print(f"Downloading all scores for user {user_id}...")

    if verbose:
        print("  Downloading best scores...")
    results["best"] = download_user_best_scores(
        user_id, max=max, only=only, verbose=verbose
    )

    if verbose:
        print("  Downloading recent scores...")
    results["recent"] = download_user_recent_scores(
        user_id, max=max, only=only, verbose=verbose
    )

    if verbose:
        print("  Downloading first scores...")
    results["firsts"] = download_user_first_scores(
        user_id, max=max, only=only, verbose=verbose
    )

    if verbose:
        total_replays = sum(
            len(replays) if replays else 0 for replays in results.values()
        )
        print(f"Total downloaded: {total_replays} replays")

    return results


def download_beatmapset(beatmapset_id: int, verbose: bool = False) -> str:
    """Download and extract beatmapset from osu! API

    Args:
        beatmapset_id: The beatmapset ID to download
        verbose: Whether to print progress messages

    Returns:
        Path to extracted beatmapset directory, or None if failed
    """
    songs_cache_path = ".data/songs"
    if not os.path.exists(songs_cache_path):
        os.makedirs(songs_cache_path, exist_ok=True)

    # Check if beatmapset already exists in cache
    existing_dirs = [
        d for d in os.listdir(songs_cache_path) if d.startswith(str(beatmapset_id))
    ]
    if existing_dirs:
        beatmapset_path = os.path.join(songs_cache_path, existing_dirs[0])
        if verbose:
            print(
                f"Beatmapset {beatmapset_id} already exists in cache: {existing_dirs[0]}"
            )
        return beatmapset_path

    url = f"https://osu.ppy.sh/beatmapsets/{beatmapset_id}/download"
    retry_times = 0
    headers = {
        "Host": "osu.ppy.sh",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:141.0) Gecko/20100101 Firefox/141.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Referer": f"https://osu.ppy.sh/beatmapsets/{beatmapset_id}",
        "Connection": "keep-alive",
        "Cookie": f"osu_session={osu_session}",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    if verbose:
        print(f"Downloading beatmapset {beatmapset_id}...")
        print(f"Request URL: {url}")
        print(
            f"Session cookie: {osu_session[:50]}..."
            if osu_session
            else "No session cookie found!"
        )

    response = requests.get(url, headers=headers, allow_redirects=True)

    if verbose:
        print(f"Response status: {response.status_code}")
        if response.history:
            print(f"Redirected from: {[r.url for r in response.history]}")
        print(f"Final URL: {response.url}")

    if response.status_code // 100 != 2:
        if response.status_code == 404:
            if verbose:
                print(f"Beatmapset {beatmapset_id} not found (404)")
            return None
        elif response.status_code == 401:
            print(
                f"Authentication failed (401) for beatmapset {beatmapset_id}. Check your osu_session cookie."
            )
            if verbose:
                print(
                    "Current session cookie:",
                    osu_session[:50] + "..." if osu_session else "None",
                )
                print(
                    "Expected format from .har file:",
                    "eyJpdiI6IlpKZFVuZ244RXVPRzBTV3p6Q000ckE9PSIsInZhbHVlIjoi...",
                )
            return None

        verbose and print(
            f"osu responded with status code {response.status_code} for beatmapset {beatmapset_id}, waiting a few secs"
        )

        while retry_times < 5 and response.status_code // 100 != 2:
            time.sleep(4 * (retry_times + 1))
            response = requests.get(url, headers=headers, allow_redirects=True)
            if response.status_code == 404:
                if verbose:
                    print(f"Beatmapset {beatmapset_id} not found (404)")
                return None
            elif response.status_code == 401:
                print(
                    f"Authentication failed (401) for beatmapset {beatmapset_id}. Check your osu_session cookie."
                )
                return None
            if response.status_code // 100 == 2:
                break
            verbose and print(
                f"osu responded with status code {response.status_code} for beatmapset {beatmapset_id}, waiting a few secs"
            )
            retry_times += 1

        if retry_times == 5:
            print(
                f"Error: Retried 5 times without success for beatmapset {beatmapset_id}, stopping. check osu session."
            )
            return None

    # Save zip file temporarily
    temp_zip_path = os.path.join(songs_cache_path, f"{beatmapset_id}_temp.zip")
    try:
        with open(temp_zip_path, "wb") as f:
            f.write(response.content)

        # Extract zip file
        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            # Get the first directory name from the zip to determine the beatmapset folder name
            zip_contents = zip_ref.namelist()
            if not zip_contents:
                if verbose:
                    print(f"Empty zip file for beatmapset {beatmapset_id}")
                return None

            # Extract all contents
            zip_ref.extractall(songs_cache_path)

            # Find the extracted directory (it should start with the beatmapset_id)
            extracted_dirs = [
                d
                for d in os.listdir(songs_cache_path)
                if os.path.isdir(os.path.join(songs_cache_path, d))
                and d.startswith(str(beatmapset_id))
            ]

            if not extracted_dirs:
                # If no directory starts with beatmapset_id, look for the main directory in zip
                main_dir = (
                    zip_contents[0].split("/")[0]
                    if "/" in zip_contents[0]
                    else zip_contents[0]
                )
                if os.path.isdir(os.path.join(songs_cache_path, main_dir)):
                    # Rename to include beatmapset_id
                    old_path = os.path.join(songs_cache_path, main_dir)
                    new_path = os.path.join(
                        songs_cache_path, f"{beatmapset_id} {main_dir}"
                    )
                    os.rename(old_path, new_path)
                    beatmapset_path = new_path
                else:
                    if verbose:
                        print(
                            f"Could not find extracted directory for beatmapset {beatmapset_id}"
                        )
                    return None
            else:
                beatmapset_path = os.path.join(songs_cache_path, extracted_dirs[0])

        if verbose:
            print(f"Successfully downloaded and extracted beatmapset {beatmapset_id}")

        return beatmapset_path

    except Exception as e:
        if verbose:
            print(f"Error extracting beatmapset {beatmapset_id}: {e}")
        return None

    finally:
        # Clean up temporary zip file
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
