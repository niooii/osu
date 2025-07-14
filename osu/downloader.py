import re
import io
import requests
from bs4 import BeautifulSoup
import json
import osu.rulesets.beatmap as beatmap
import osu.rulesets.replay as replay
import os
from typing import Literal, Callable
from dotenv import load_dotenv
import pandas as pd
import time
import shutil
import hashlib
from osu.rulesets.core import OSU_PATH

load_dotenv()

osu_session = os.getenv("OSU_SESSION")

os.makedirs('../.data/replays', exist_ok=True)


# TODO we can make this async and then batch the donwloads, but nah im too lazy
# TODO add cache to map beatmaps to the replays
def get_replay(score_id: int, map_md5: str, verbose: bool = False) -> (replay.Replay, int):
    save_path = f".data/replays/{score_id}.osr"
    if os.path.exists(save_path):
        return (replay.load(save_path), score_id)

    url = f'https://osu.ppy.sh/scores/{score_id}/download'
    retry_times = 0
    headers = {
        'Cookie': f'osu_session={osu_session}',
        'Referer': f'https://osu.ppy.sh/scores/{score_id}',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:141.0) Gecko/20100101 Firefox/141.0'
    }
    response = requests.get(url, headers=headers)
    if response.status_code // 100 != 2:
        if response.status_code == 404:
            return None
        verbose and print(
            f'osu responded with status code {response.status_code} for replay {score_id}, waiting a few secs')
        while retry_times < 5 and response.status_code // 100 != 2:
            time.sleep(4 * (retry_times + 1))
            response = requests.get(url, headers=headers)
            if response.status_code == 404:
                return None
            if response.status_code // 100 == 2:
                break
            verbose and print(
                f'osu responded with status code {response.status_code} for replay {score_id}, waiting a few secs')
            retry_times += 1
        if retry_times == 5:
            print(
                f'Error: Retried 5 or so times without a 200 status code for replay {score_id}, stopping. check osu session.')
            return None

    rp = replay.Replay(io.BytesIO(response.content))
    if rp.map_md5 != map_md5:
        verbose and print(f'replay {score_id}: mismatching hash.')
        return None

    # save to disk cache
    with open(save_path, 'wb') as file:
        file.write(response.content)

    return (rp, score_id)


def download_t50_replays(beatmap: beatmap.Beatmap, cache_map_path: str, map_md5: str, max: int = 50,
                         only: Literal['FC', 'S', 'ALL'] = 'S', verbose: bool = False) -> \
        list[replay.Replay]:
    retry_times = 0
    beatmap_id = beatmap.beatmap_id()
    t50_url = f'https://osu.ppy.sh/beatmaps/{beatmap_id}/scores?mode=osu&type=global'

    if not os.path.exists('.data/replays/'):
        os.makedirs('.data/replays/')

    response = requests.get(t50_url)
    if response.status_code // 100 != 2:
        if response.status_code == 404:
            return None
        verbose and print(f'osu responded with status code {response.status_code}, waiting a few secs')

        while retry_times < 5 and response.status_code // 100 != 2:
            time.sleep(4 * (retry_times + 1))
            response = requests.get(t50_url)
            if response.status_code == 404:
                return None
            if response.status_code // 100 == 2:
                break
            verbose and print(f'osu responded with status code {response.status_code}, waiting a few secs')
            retry_times += 1

        if retry_times == 5:
            print('Error: Retried 5 or so times without a 200 status code, stopping. check osu session.')
            return None

    response = json.loads(response.text)
    scores = response['scores']

    # we also filter out the ones that had no replay obviously
    replay_ids = []
    for score in scores:
        rank = score['rank']
        if only == 'S' and not rank.startswith('S'):
            continue
        elif only == 'FC' and not score['is_perfect_combo']:
            continue

        if score['has_replay']:
            replay_ids.append(score['id'])

    replay_ids = replay_ids[0:max]

    # download replays
    replays = [get_replay(id, map_md5, verbose) for id in replay_ids]
    replays = [t for t in replays if t is not None]

    if replays is None:
        verbose and print(f'Failed to download replays for map id {beatmap_id}')
        return None

    for replay, id in replays:
        meta_path = f".data/replays/{id}.meta"
        if not os.path.exists(meta_path):
            with open(meta_path, 'w') as f:
                f.write(json.dumps({'map': cache_map_path}))

    replays = [replay for (replay, id) in replays]

    return replays


def download_local_mapset_t50_replays(mapset_folder: str, verbose: bool = False,
                                      filter: None | Callable[[beatmap.Beatmap], bool] = None,
                                      **kwargs) -> pd.DataFrame:
    mapset_folder_name = mapset_folder
    mapset_folder = f'{OSU_PATH}/Songs/{mapset_folder}'
    files = os.listdir(mapset_folder)
    maps = []
    replays = []

    for path in files:
        if not path.endswith('.osu'):
            continue

        verbose and print(f'loading {path}')
        map_path = f'{mapset_folder}/{path}'
        mp = beatmap.load(map_path)

        if mp.beatmap_id() is None:
            continue

        if filter is not None:
            if not filter(mp):
                continue

        cache_map_path = f'.data/songs/{mapset_folder_name}/{mp.beatmap_id()}_{path}'

        with open(map_path, 'rb') as f:
            hash = hashlib.md5(f.read()).hexdigest()

        downloaded_replays = download_t50_replays(mp, cache_map_path=cache_map_path, map_md5=hash, verbose=verbose,
                                                  **kwargs)
        if downloaded_replays is None:
            continue

        for rp in downloaded_replays:
            maps.append(mp)
            replays.append(rp)

        if len(downloaded_replays) > 0:
            if not os.path.exists(cache_map_path):
                os.makedirs(os.path.dirname(cache_map_path), exist_ok=True)
                shutil.copyfile(map_path, cache_map_path)

    return pd.DataFrame(list(zip(replays, maps)), columns=['replay', 'beatmap'])


import tqdm


# call download_local_mapset_t50_replays for all in the list
def download_mapsets(mapset_folders: list[str], return_dataframe: bool = False, verbose: bool = False, **kwargs):
    dataframe = pd.DataFrame(columns=['replay', 'beatmap'])

    pbar = tqdm.tqdm(mapset_folders, total=len(mapset_folders), desc='Downloading replays')
    for mapset_folder in pbar:
        # if the mapset folder doesn't start with a number, chances are it won't have useful metadata in the maps anyways
        pattern = r'^(\d+) '
        match = re.match(pattern, mapset_folder)
        if not match or int(match.group(1)) < 100000:
            verbose and print(f'Skipping folder {mapset_folder}')
            continue
        pbar.set_description(f'Downloading replays from maps in {mapset_folder}')
        df = download_local_mapset_t50_replays(mapset_folder, verbose=verbose, **kwargs)
        if df is not None:
            verbose and print(f'Got {len(df)} replays that meet the criteria.')
            if return_dataframe:
                dataframe = pd.concat([dataframe, df])

    if not return_dataframe:
        return None
    else:
        return dataframe


def get_all_mapset_folders_on_disk() -> list[str]:
    return os.listdir(f'{OSU_PATH}/Songs/')
