from simple_diarizer.diarizer import Diarizer
import subprocess
from datetime import timedelta
import os
import tarkibi.utilities.general, tarkibi.utilities.youtube, tarkibi.utilities.audio, tarkibi.utilities.agent

AUDIO_NN = tarkibi.utilities.general.AUDIO_NN
AUDIO_RAW = tarkibi.utilities.general.AUDIO_RAW
AUDIO_CLIPS = tarkibi.utilities.general.AUDIO_CLIPS
AUDIO_SPECS = tarkibi.utilities.general.AUDIO_SPECS
AUDIO_FINAL = tarkibi.utilities.general.AUDIO_FINAL
BASE_DIR = tarkibi.utilities.general.BASE_DIR

def noise_reduction(file_path: str) -> None:
    output_path = f'{BASE_DIR}/{AUDIO_NN}'
    subprocess.run(f'spleeter separate -o {output_path} {file_path}', shell=True)
   
def diarize_audio(file_path: str, audio_id: str) -> None:    
    diar = Diarizer(
      embed_model='xvec',  
      cluster_method='sc' 
    )

    segments = diar.diarize(file_path, 
        num_speakers=None,
        threshold=1e-1,
    )

    speakers = {}
    for segment in segments:
        speaker_id = segment['label']
        segment_start = segment['start']
        segment_end = segment['end']

        if speaker_id not in speakers:
            speakers[speaker_id] = {
                'duration': segment_end - segment_start, 
                'segments': [{
                    'start': segment_start,
                    'end': segment_end
                }]
            }
        
        else:
            speakers[speaker_id]['duration'] += segment_end - segment_start
            speakers[speaker_id]['segments'].append({
                'start': segment_start,
                'end': segment_end
            })
    

    for speaker, info in speakers.items():
        if not os.path.exists(f'{BASE_DIR}/{AUDIO_CLIPS}/{audio_id}'):
            os.mkdir(f'{BASE_DIR}/{AUDIO_CLIPS}/{audio_id}')
            
        i = 0
        for segment in info['segments']:
            segment_start = tarkibi.utilities.general.format_time(segment['start'])
            segment_end = tarkibi.utilities.general.format_time(segment['end']) 

            subprocess.run(
                f'ffmpeg -i {file_path} -ss {segment_start} -to {segment_end} -c copy {BASE_DIR}/{AUDIO_CLIPS}/{audio_id}/{speaker}_{i}.wav',
                shell=True,
                stdout=subprocess.DEVNULL
            )

            i += 1
    
def process_video(video_id: str, debug_mode: bool = False):
    try:
        tarkibi.utilities.youtube.download_video(video_id, output_path=f'{BASE_DIR}/{AUDIO_RAW}')
    except:
        return
    
    noise_reduction(f'{BASE_DIR}/{AUDIO_RAW}/{video_id}.wav')
    diarize_audio(f'{BASE_DIR}/{AUDIO_NN}/{video_id}/vocals.wav', video_id)
    
    if not debug_mode: tarkibi.utilities.general.light_clean(video_id)
     
     
def build_dataset(author: str, target_length: timedelta, output_path: str = 'dataset', sample_rate: int = 24000):
    tarkibi.utilities.general.make_directories()
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    search_query = tarkibi.utilities.agent.generate_search_query(author)
    videos = tarkibi.utilities.youtube.youtube_search(search_query)

    closest_combination = tarkibi.utilities.general.find_closest_combination(videos, target_length)
    for video in closest_combination:
        video_id = video['id']
        process_video(video_id)

    i = 0
    audio_groups = tarkibi.utilities.audio.find_similar_clips(f'{BASE_DIR}/{AUDIO_CLIPS}')
    for ag, clips in audio_groups.items():
        concat_file = f'{BASE_DIR}/{AUDIO_SPECS}/{ag}_concat.txt'
        with open(concat_file, 'w') as f:
            for clip in clips:
                clip = os.path.join(*clip.split('/')[1:])
                f.write(f'file ../{clip}\n')
            
        subprocess.run(f'ffmpeg -f concat -safe 0 -i {concat_file} -c copy {BASE_DIR}/{AUDIO_FINAL}/{ag}.wav', shell=True)
        subprocess.run(f'ffmpeg -i {BASE_DIR}/{AUDIO_FINAL}/{ag}.wav -ar {str(sample_rate)} {output_path}/{i}.wav', shell=True)
        i += 1

    tarkibi.utilities.general.deep_clean()
    