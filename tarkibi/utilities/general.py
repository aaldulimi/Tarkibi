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
    subdirectories = [AUDIO_FINAL, AUDIO_CLIPS, DOWNLOADS, AUDIO_RAW, AUDIO_SPECS, AUDIO_NN]

    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)

    for directory in subdirectories:
        subdir_path = os.path.join(BASE_DIR, directory)
        if not os.path.exists(subdir_path):
            os.mkdir(subdir_path)

def light_clean(audio_id: str) -> None:
    shutil.rmtree(f'{BASE_DIR}/{AUDIO_NN}/{audio_id}')
    os.remove(f'{BASE_DIR}/{AUDIO_RAW}/{audio_id}.wav')
    os.remove(f'{BASE_DIR}/{DOWNLOADS}/{audio_id}.mp4')

def deep_clean() -> None:
    shutil.rmtree(BASE_DIR, ignore_errors=True)
