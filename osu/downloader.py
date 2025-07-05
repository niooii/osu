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

load_dotenv()

osu_session = os.getenv("OSU_SESSION")

APP_DATA = os.getenv("APPDATA")
OSU_PATH = f'{APP_DATA}/../Local/osu!'

print(OSU_PATH)

os.makedirs('../.data/replays', exist_ok=True)


# TODO we can make this async and then batch the donwloads, but nah im too lazy
def get_replay(score_id: int) -> replay.Replay:
    save_path = f"../.data/replays/{score_id}.osr"
    if os.path.exists(save_path):
        return replay.load(save_path)

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
        print(f'osu responded with status code {response.status_code} for replay {score_id}, waiting a few secs')
        while retry_times < 5 and response.status_code // 100 != 2:
            time.sleep(4)
            response = requests.get(url, headers=headers)
            if response.status_code == 404:
                return None
            if response.status_code // 100 == 2:
                break
            print(f'osu responded with status code {response.status_code} for replay {score_id}, waiting a few secs')
            retry_times += 1
        if retry_times == 5:
            print(f'retried 5 or so times for replay {score_id}, stopping. check osu session.')
            return None
    # save to disk cache
    with open(save_path, 'wb') as file:
        file.write(response.content)
    return replay.load(save_path)


def download_t50_replays(beatmap_id: int, max: int = 50, only: Literal['FC', 'S', 'ALL'] = 'S') -> list[replay.Replay]:
    retry_times = 0
    t50_url = f'https://osu.ppy.sh/beatmaps/{beatmap_id}/scores?mode=osu&type=global'

    response = requests.get(t50_url)
    if response.status_code // 100 != 2:
        if response.status_code == 404:
            return None
        print(f'osu responded with status code {response.status_code}, waiting a few secs')

        while retry_times < 5 and response.status_code // 100 != 2:
            time.sleep(4)
            response = requests.get(t50_url)
            if response.status_code == 404:
                return None
            if response.status_code // 100 == 2:
                break
            print(f'osu responded with status code {response.status_code}, waiting a few secs')
            retry_times += 1

        if retry_times == 5:
            print('retried 5 or so times, stopping. check osu session.')
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
    replays = [get_replay(id) for id in replay_ids]

    return replays


def download_local_mapset_t50_replays(mapset_folder: str, filter: None | Callable[[beatmap.Beatmap], bool] = None,
                                      **kwargs) -> pd.DataFrame:
    mapset_folder = f'{OSU_PATH}/Songs/{mapset_folder}'
    files = os.listdir(mapset_folder)
    maps = []
    replays = []

    for path in files:
        if not path.endswith('.osu'):
            continue

        print(f'loading {path}')
        mp = beatmap.load(f'{mapset_folder}/{path}')

        if mp.beatmap_id() is None:
            continue

        if filter is not None:
            if not filter(mp):
                continue

        downloaded_replays = download_t50_replays(mp.beatmap_id(), **kwargs)
        if downloaded_replays is None:
            continue

        for rp in downloaded_replays:
            maps.append(mp)
            replays.append(rp)

    return pd.DataFrame(list(zip(replays, maps)), columns=['replay', 'beatmap'])


# call download_local_mapset_t50_replays for all in the list
def download_mapsets(mapset_folders: list[str], **kwargs):
    dataframe = pd.DataFrame(columns=['replay', 'beatmap'])
    for mapset_folder in mapset_folders:
        df = download_local_mapset_t50_replays(mapset_folder, **kwargs)
        if df is not None:
            dataframe = pd.concat([dataframe, df])

    return dataframe


def download_mapset_t50_replays(mapset_id: int, above_stars: float, **kwargs):
    pass