from simple_diarizer.diarizer import Diarizer
import subprocess
from datetime import timedelta
import os
import tarkibi.utilities.general, tarkibi.utilities.youtube, tarkibi.utilities.audio, tarkibi.utilities.agent
import shutil
import typing
import itertools
import wave

AUDIO_NN = tarkibi.utilities.general.AUDIO_NN
AUDIO_RAW = tarkibi.utilities.general.AUDIO_RAW
AUDIO_CLIPS = tarkibi.utilities.general.AUDIO_CLIPS
AUDIO_SPECS = tarkibi.utilities.general.AUDIO_SPECS
AUDIO_FINAL = tarkibi.utilities.general.AUDIO_FINAL
BASE_DIR = tarkibi.utilities.general.BASE_DIR

class Tarkibi:
    _AUDIO_RAW_PATH = f'{BASE_DIR}/{AUDIO_RAW}'
    _DURATION_MULTIPLIER = 1.2

    def __init__(self):
        self._noise_reduction = _NoiseReduction()
        self._diarization = _Diarization()
        self._agent = tarkibi.utilities.agent._Agent()
        self._youtube = tarkibi.utilities.youtube._Youtube()

    def _convert_time_to_minutes(self, time_str: str) -> float:
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

    def _find_closest_combination(self, clips: dict, target_duration: timedelta) -> tuple | list:
        target_minutes = (target_duration.total_seconds() / 60) * self._DURATION_MULTIPLIER
        clips.sort(key=lambda x: self._convert_time_to_minutes(x['length']))

        closest_combination = []
        closest_duration = 0

        for i in range(1, len(clips) + 1):
            for combination in itertools.combinations(clips, i):
                total_duration = sum(self._convert_time_to_minutes(clip['length']) for clip in combination)

                if total_duration <= target_minutes and total_duration > closest_duration:
                    closest_combination = combination
                    closest_duration = total_duration

        return closest_combination

    def _process_video(self, video_id: str, debug_mode: bool = False):
        self._youtube._download_video(video_id, output_path=self._AUDIO_RAW_PATH)
    
        nr_output_path = self._noise_reduction._noise_reduction(f'{self._AUDIO_RAW_PATH}/{video_id}.wav')
        _ = self._diarization._diarize_audio(f'{nr_output_path}/{video_id}/vocals.wav', video_id)

        if not debug_mode:
            os.remove(f'{self._AUDIO_RAW_PATH}/{video_id}.wav')
            shutil.rmtree(f'{nr_output_path}/{video_id}')
    
    def _total_duration(self, audio_directory: str) -> float:
        total_duration = 0
        for filename in os.listdir(audio_directory):
            if filename.endswith(".wav"):
                wav_file_path = os.path.join(audio_directory, filename)
                with wave.open(wav_file_path, 'rb') as wav_file:
                    duration = wav_file.getnframes() / float(wav_file.getframerate())
                    total_duration += duration

        return total_duration

    def build_dataset(self, author: str, reference_audio: str, target_duration: timedelta, output_path: str = 'dataset', sample_rate: int = 24000,
                    _offset: int | None = 0, _clips_used: list | None = None):
        tarkibi.utilities.general.make_directories()
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        search_query = self._agent._generate_search_query(author)
        videos = self._youtube._search(search_query)

        if _clips_used:
            videos = [video for video in videos if video['id'] not in _clips_used]
        else: _clips_used = []

        closest_combination = self._find_closest_combination(videos, target_duration)
        for video in closest_combination:
            video_id = video['id']
            _clips_used.append(video_id)
            self._process_video(video_id)
        
        audio_groups = tarkibi.utilities.audio.find_similar_clips(f'{BASE_DIR}/{AUDIO_CLIPS}', reference_audio)

        for ag, clips in audio_groups.items():
            concat_file = f'{BASE_DIR}/{AUDIO_SPECS}/{ag}_concat.txt'
            with open(concat_file, 'w') as f:
                for clip in clips:
                    clip = os.path.join(*clip.split('/')[1:])
                    f.write(f'file ../{clip}\n')
                
            subprocess.run(f'ffmpeg -f concat -safe 0 -i {concat_file} -c copy {BASE_DIR}/{AUDIO_FINAL}/{ag}.wav', shell=True)
            subprocess.run(f'ffmpeg -i {BASE_DIR}/{AUDIO_FINAL}/{ag}.wav -ar {str(sample_rate)} {output_path}/{_offset}.wav', shell=True)
            shutil.rmtree(f'{BASE_DIR}/{AUDIO_CLIPS}/{ag}')
            _offset += 1
        
        remaining_duration = target_duration.total_seconds() - self._total_duration(output_path)
        if remaining_duration > (0.2 * target_duration.total_seconds()):
            self.build_dataset(author, reference_audio, timedelta(seconds=remaining_duration), output_path, sample_rate, _offset, _clips_used)
        
        tarkibi.utilities.general.deep_clean()

class _NoiseReduction:
    _NR_OUTPUT_PATH = f'{BASE_DIR}/{AUDIO_NN}'

    def __init__(self):
        pass
        
    def _noise_reduction(self, file_path: str = None) -> str:
        subprocess.run(f'spleeter separate -o {self._NR_OUTPUT_PATH} {file_path}', shell=True)

        return self._NR_OUTPUT_PATH

class _Diarization:
    _AUDIO_CLIPS_PATH = f'{BASE_DIR}/{AUDIO_CLIPS}'

    def __init__(self):
        pass
        
    def _format_time(seconds) -> str:
        return str(timedelta(seconds=seconds))
    
    def _diarize_audio_to_segments(self, file_path: str = None) -> list | dict[str, typing.Any]:
        diar = Diarizer(
            embed_model='xvec',  
            cluster_method='sc' 
        )

        segments = diar.diarize(
            file_path, 
            num_speakers=None,
            threshold=1e-1,
        )
        
        return segments

    def _group_segments_by_speaker(self, segments: list | dict[str, typing.Any]) -> dict[str, typing.Any]:
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
        
        return speakers
    

    def _segment_audio_clips(self, speakers: dict[str, typing.Any], file_path: str, audio_id: str) -> None:
        for speaker, info in speakers.items():
            audio_clips_path = f'{self._AUDIO_CLIPS_PATH}/{audio_id}'

            if not os.path.exists(audio_clips_path):
                os.mkdir(audio_clips_path)
                
            i = 0
            for segment in info['segments']:
                segment_start = self._format_time(segment['start'])
                segment_end = self._format_time(segment['end']) 

                subprocess.run(
                    f'ffmpeg -i {file_path} -ss {segment_start} -to {segment_end} -c copy {audio_clips_path}/{speaker}_{i}.wav',
                    shell=True,
                    stdout=subprocess.DEVNULL
                )

                i += 1
    
    def _diarize_audio(self, file_path: str, audio_id: str) -> str:
        segments = self._diarize_audio_to_segments(file_path)
        speakers = self._group_segments_by_speaker(segments)
        self._segment_audio_clips(speakers, file_path, audio_id)

        return self._AUDIO_CLIPS_PATH


     
# def build_dataset(author: str, reference_audio: str, target_duration: timedelta, output_path: str = 'dataset', sample_rate: int = 24000,
#                   _offset: int | None = 0, _clips_used: list | None = None):
#     tarkibi.utilities.general.make_directories()
#     if not os.path.exists(output_path):
#         os.mkdir(output_path)

#     search_query = tarkibi.utilities.agent.generate_search_query(author)
#     videos = tarkibi.utilities.youtube.youtube_search(search_query)

#     if _clips_used:
#         videos = [video for video in videos if video['id'] not in _clips_used]
#     else: _clips_used = []
    
#     closest_combination = tarkibi.utilities.general.find_closest_combination(videos, target_duration)
#     for video in closest_combination:
#         video_id = video['id']
#         _clips_used.append(video_id)
#         process_video(video_id)
    
#     audio_groups = tarkibi.utilities.audio.find_similar_clips(f'{BASE_DIR}/{AUDIO_CLIPS}', reference_audio)
    
#     for ag, clips in audio_groups.items():
#         concat_file = f'{BASE_DIR}/{AUDIO_SPECS}/{ag}_concat.txt'
#         with open(concat_file, 'w') as f:
#             for clip in clips:
#                 clip = os.path.join(*clip.split('/')[1:])
#                 f.write(f'file ../{clip}\n')
            
#         subprocess.run(f'ffmpeg -f concat -safe 0 -i {concat_file} -c copy {BASE_DIR}/{AUDIO_FINAL}/{ag}.wav', shell=True)
#         subprocess.run(f'ffmpeg -i {BASE_DIR}/{AUDIO_FINAL}/{ag}.wav -ar {str(sample_rate)} {output_path}/{_offset}.wav', shell=True)
#         shutil.rmtree(f'{BASE_DIR}/{AUDIO_CLIPS}/{ag}')
#         _offset += 1

#     remaining_duration = target_duration.total_seconds() - tarkibi.utilities.general.total_duration(output_path)
#     if remaining_duration > (0.2 * target_duration.total_seconds()):
#         build_dataset(author, reference_audio, timedelta(seconds=remaining_duration), output_path, sample_rate, _offset, _clips_used)

#     tarkibi.utilities.general.deep_clean()
    