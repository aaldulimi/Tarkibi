import itertools
from datetime import timedelta
import os
import shutil

BASE_DIR = '.tarkibi'
AUDIO_RAW = 'audio_raw'
AUDIO_SPECS = 'audio_specs'
AUDIO_NN = 'audio_nn'
DOWNLOADS = 'downloads'
AUDIO_CLIPS = 'audio_clips'
AUDIO_FINAL =  'audio_final'

def make_directories() -> None:
    base_dir = '.tarkibi'
    subdirectories = [AUDIO_FINAL, AUDIO_CLIPS, DOWNLOADS, AUDIO_RAW, AUDIO_SPECS, AUDIO_NN]

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    for directory in subdirectories:
        subdir_path = os.path.join(base_dir, directory)
        if not os.path.exists(subdir_path):
            os.mkdir(subdir_path)

def convert_time_to_minutes(time_str: str) -> float:
    parts = time_str.split(':')

    if len(parts) == 2:
        minutes, seconds = map(int, parts)
        total_minutes =  minutes + seconds / 60.0
    elif len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        total_minutes = hours * 60 + minutes + seconds / 60.0
    else:
        raise ValueError(f'Invalid time string: {time_str}')

    return total_minutes   
    

def find_closest_combination(clips: dict, target_duration: timedelta) -> tuple | list:
    target_minutes = target_duration.total_seconds() / 60

    clips.sort(key=lambda x: convert_time_to_minutes(x['length']))

    closest_combination = []
    closest_duration = 0

    for i in range(1, len(clips) + 1):
        for combination in itertools.combinations(clips, i):
            total_duration = sum(convert_time_to_minutes(clip['length']) for clip in combination)

            if total_duration <= target_minutes and total_duration > closest_duration:
                closest_combination = combination
                closest_duration = total_duration

    return closest_combination

def format_time(seconds) -> str:
    return str(timedelta(seconds=seconds))

def light_clean(audio_id: str) -> None:
    shutil.rmtree(f'{BASE_DIR}/{AUDIO_NN}/{audio_id}')
    os.remove(f'{BASE_DIR}/{AUDIO_RAW}/{audio_id}.wav')
    os.remove(f'{BASE_DIR}/{DOWNLOADS}/{audio_id}.mp4')

def deep_clean() -> None:
    shutil.rmtree(BASE_DIR)