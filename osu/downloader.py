import hashlib
import io
import json
import math
import os
import re
import shutil
import time
import polars as pl
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

# a lot of room for code reusability here, a bit too lazy though

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
            elif response.status_code == 429:
                print("Rate limit reached... trying again in 5 minutes")
                time.sleep(300)
                print("Trying again now... :/")
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
        print(f"replay {score_id}: mismatching hash.")
        return None

    # save to disk cache
    with open(save_path, "wb") as file:
        file.write(response.content)

    return (rp, score_id)


def download_t100_replays(
        beatmap: beatmap.Beatmap,
        cache_map_path: str,
        map_md5: str,
        max: int = 50,
        only: Literal["FC", "S", "ALL"] = "S",
        by_user_id: int | None = None,
        verbose: bool = False,
) -> list[replay.Replay]:
    retry_times = 0
    beatmap_id = beatmap.beatmap_id()
    t100_url = f"https://osu.ppy.sh/beatmaps/{beatmap_id}/scores?mode=osu&type=global&limit=100"

    if not os.path.exists(".data/replays/"):
        os.makedirs(".data/replays/")

    response = requests.get(t100_url)
    if response.status_code // 100 != 2:
        if response.status_code == 404:
            return None
        verbose and print(
            f"osu responded with status code {response.status_code}, waiting a few secs"
        )

        while retry_times < 5 and response.status_code // 100 != 2:
            time.sleep(4 * (retry_times + 1))
            response = requests.get(t100_url)
            if response.status_code == 404:
                return None
            elif response.status_code == 429:
                print("Rate limit reached... trying again in 5 minutes")
                time.sleep(300)
                print("Trying again now... :/")
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

    if by_user_id is not None:
        scores = [score for score in scores if int(score["user_id"]) == int(by_user_id)]

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
    if by_user_id is not None:
        replays = [get_user_replay(id, map_md5, user_id=by_user_id, verbose=verbose) for id in replay_ids]
    else:
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


def download_local_mapset_t100_replays(
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

        downloaded_replays = download_t100_replays(
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


# call download_local_mapset_t100_replays for all in the list
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
    gotten = 0
    for mapset_folder in pbar:
        # if the mapset folder doesn't start with a number, chances are it won't have useful metadata in the maps anyways
        pattern = r"^(\d+) "
        match = re.match(pattern, mapset_folder)
        # skip way too old maps to have metadata
        if not match or int(match.group(1)) < 100000:
            verbose and print(f"Skipping folder {mapset_folder}")
            continue
        pbar.set_description(f"Downloading replays from maps in {mapset_folder} (TOTAL: {gotten})")
        df = download_local_mapset_t100_replays(mapset_folder, verbose=verbose, **kwargs)
        if df is not None:
            verbose and print(f"Got {len(df)} replays that meet the criteria.")
            gotten += len(df)
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

    # dodge rate limit
    time.sleep(1.5)
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


def _download_user_scores_by_type(
        user_id: int,
        score_type: Literal["best", "recent", "firsts"],
        max: int = 10000,
        only: Literal["FC", "S", "ALL"] = "S",
        verbose: bool = False,
) -> list[replay.Replay]:
    retry_times = 0
    replays = []

    for offset in range(int(math.ceil(max / 100.0))):
        scores_url = (
            f"https://osu.ppy.sh/users/{user_id}/scores/{score_type}?mode=osu&limit={max}&offset={offset * 100}"
        )

        replays_path = f".data/replays-{user_id}/"
        if not os.path.exists(replays_path):
            os.makedirs(replays_path)

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
                elif response.status_code == 429:
                    print("Rate limit reached... trying again in 5 minutes")
                    time.sleep(300)
                    print("Trying again now... :/")
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
        if len(replay_ids) == 0:
            return replays

        for score_id, map_md5, beatmapset_id, beatmap_id in tqdm.tqdm(replay_ids,
                                                                      desc=f"Downloading replays in category {score_type}, offset {offset * 100}"):
            # Download beatmapset (function handles caching automatically)
            beatmapset_path = download_beatmapset(
                beatmapset_id,
                needs=beatmap_id,
                use_mirror=False,
                verbose=verbose
            )
            if not beatmapset_path:
                verbose and print(f"Failed to download beatmapset {beatmapset_id} for score {score_id}")
                continue

            # Download replay
            replay_result = get_user_replay(score_id, map_md5, user_id, verbose)
            if replay_result is None:
                continue

            replays.append(replay_result)

            (replay, id) = replay_result

            # Find the specific map it was for (linear search oh well)
            mapfile_path = find_diff(target_id=beatmap_id, mapset_path=beatmapset_path)

            if mapfile_path is None:
                verbose and print("Failed to get the specific map for replay..")
                continue

            meta_path = f"{replays_path}/{id}.meta"
            if not os.path.exists(meta_path):
                with open(meta_path, "w") as f:
                    f.write(json.dumps({"map": mapfile_path}))

            replays.append(replay)

    replays = [t for t in replays if t is not None]

    if len(replays) == 0:
        verbose and print(f"Failed to download replays for user {user_id}")
        return None

    return replays

# requires a polar dataframe with at least two columns:
# id: int, beatmapset_id: int
def download_from_df(df: pl.DataFrame, user_id: int, verbose: bool = False):
    replays = []

    replays_path = f".data/replays-{user_id}/"
    if not os.path.exists(replays_path):
        os.makedirs(replays_path)

    prog = tqdm.tqdm(df.select(['id', 'beatmapset_id', 'beatmap_id']).iter_rows(),
                                                                 desc=f"Downloading replays in dataframe..", total=len(df))
    for score_id, beatmapset_id, beatmap_id in prog:
        beatmapset_path = download_beatmapset(beatmapset_id, needs=beatmap_id, verbose=verbose)
        if not beatmapset_path:
            verbose and print(f"Failed to download beatmapset {beatmapset_id} for score {score_id}")
            continue

        # Find the specific map it was for (linear search oh well)
        mapfile_path = find_diff(target_id=beatmap_id, mapset_path=beatmapset_path)

        if mapfile_path is None:
            verbose and print("Failed to get the specific map for replay..")
            continue

        map_md5 = hashlib.md5(open(mapfile_path, "rb").read()).hexdigest()

        # Download replay
        replay_result = get_user_replay(score_id, map_md5, user_id, verbose)
        if replay_result is None:
            verbose and print('Failed to get user replay, skipping')
            continue

        (replay, id) = replay_result

        try:
            meta_path = f"{replays_path}/{id}.meta"
            if not os.path.exists(meta_path):
                with open(meta_path, "w") as f:
                    f.write(json.dumps({"map": mapfile_path}))
        except:
            print("Something went wrong writing meta file, skipping")
            continue

        replays.append(replay)
        prog.set_description(f"Downloading replays in dataframe ({len(replays)} downloaded)..")

    replays = [t for t in replays if t is not None]

    if len(replays) == 0:
        verbose and print(f"Failed to download replays for user {user_id}")
        return None

    return replays

def download_user_best_scores(
        user_id: int,
        max: int = 10000,
        only: Literal["FC", "S", "ALL"] = "S",
        verbose: bool = False,
) -> list[replay.Replay]:
    return _download_user_scores_by_type(user_id, "best", max, only, verbose)


def download_user_recent_scores(
        user_id: int,
        max: int = 10000,
        only: Literal["FC", "S", "ALL"] = "S",
        verbose: bool = False,
) -> list[replay.Replay]:
    return _download_user_scores_by_type(user_id, "recent", max, only, verbose)


def download_user_first_scores(
        user_id: int,
        max: int = 10000,
        only: Literal["FC", "S", "ALL"] = "S",
        verbose: bool = False,
) -> list[replay.Replay]:
    return _download_user_scores_by_type(user_id, "firsts", max, only, verbose)


def download_user_scores(
        user_id: int,
        score_type: Literal["best", "firsts", "recent", "ALL"],
        max: int = 10000,
        only: Literal["FC", "S", "ALL"] = "S",
        verbose: bool = False,
):
    """Download user scores by type or all types"""
    if score_type == "ALL":
        results = {}

        if verbose:
            print(f"Downloading all scores for user {user_id}...")

        if verbose:
            print("  Downloading best scores...")
        results["best"] = _download_user_scores_by_type(
            user_id, "best", max=max, only=only, verbose=verbose
        )

        if verbose:
            print("  Downloading recent scores...")
        results["recent"] = _download_user_scores_by_type(
            user_id, "recent", max=max, only=only, verbose=verbose
        )

        if verbose:
            print("  Downloading first scores...")
        results["firsts"] = _download_user_scores_by_type(
            user_id, "firsts", max=max, only=only, verbose=verbose
        )

        if verbose:
            total_replays = sum(
                len(replays) if replays else 0 for replays in results.values()
            )
            print(f"Total downloaded: {total_replays} replays")

        return results
    else:
        return _download_user_scores_by_type(user_id, score_type, max, only, verbose)


def find_diff(target_id: str, mapset_path: str) -> str | None:
    target_id = str(target_id)
    mapfiles = [path for path in os.listdir(mapset_path) if path.endswith('.osu')]

    mapfile_path = None
    for mapfile in mapfiles:
        mappath = f'{mapset_path}/{mapfile}'
        try:
            bm = beatmap.load(mappath)
        except RuntimeError as e:
            print(f'Error loading beatmap: {e}. Skipping')
            return None
        id = str(bm.beatmap_id()).strip()
        if target_id.strip() == id:
            mapfile_path = mappath
            break

    return mapfile_path


def download_beatmapset(
        beatmapset_id: int,
        needs: int,
        # the mirror (beatconnect) rate limits less afaik, but WAY slower
        use_mirror: bool = True,
        verbose: bool = False) -> str:
    songs_cache_path = ".data/songs"
    if not os.path.exists(songs_cache_path):
        os.makedirs(songs_cache_path, exist_ok=True)

    # Check if beatmapset already exists in cache
    existing_dirs = [
        d for d in os.listdir(songs_cache_path) if d.startswith(str(beatmapset_id))
    ]
    if existing_dirs:
        beatmapset_path = os.path.join(songs_cache_path, existing_dirs[0])
        mapfile = find_diff(target_id=str(needs), mapset_path=beatmapset_path)
        if mapfile:
            return beatmapset_path
        else:
            verbose and print("found map folder but no corresponding diff: redownloading..")

    retry_times = 0

    if use_mirror:
        url = f"https://beatconnect.io/b/{beatmapset_id}/"
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:141.0) Gecko/20100101 Firefox/141.0",
        }
    else:
        url = f"https://osu.ppy.sh/beatmapsets/{beatmapset_id}/download"
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

    response = requests.get(url, headers=headers, allow_redirects=True)

    if verbose:
        print(f"Downloading from: {response.url}")

    if response.status_code // 100 != 2:
        if response.status_code == 404:
            if verbose:
                print(f"Beatmapset {beatmapset_id} not found (404)")
            return None
        elif response.status_code == 401:
            print(
                f"Authentication failed (401) for beatmapset {beatmapset_id}. Check your osu_session cookie."
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
            elif response.status_code == 429:
                print("Rate limit reached... trying again in 5 minutes")
                time.sleep(300)
                print("Trying again now... :/")
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
            print(response.content[:2000])
            return None

    # Save zip file temporarily
    temp_zip_path = os.path.join(songs_cache_path, f"TEMP_{beatmapset_id}.zip")

    # get map name
    mapname = str(beatmapset_id)
    content_disposition = response.headers.get('Content-Disposition')
    if content_disposition:
        filename_match = re.findall(r'filename="?([^"]+)"?', content_disposition)
        if filename_match:
            mapname = filename_match[0]
        else:
            # Handle cases where filename might not be quoted or present
            mapname = None

    mapname = mapname.strip(".osz")

    try:
        with open(temp_zip_path, "wb") as f:
            f.write(response.content)

        # Extract the map
        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            mapset_path = f"{songs_cache_path}/{mapname}"
            zip_ref.extractall(mapset_path)

        if verbose:
            print(f"Successfully downloaded and extracted beatmapset {beatmapset_id}")

        return mapset_path

    except Exception as e:
        verbose and print(f"Error extracting beatmapset {beatmapset_id}: {e}")
        return None

    finally:
        # Clean up temporary zip file
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
