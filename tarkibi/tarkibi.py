import subprocess
from datetime import timedelta
import os
import shutil
import itertools
import wave
import tarkibi.utilities.general, tarkibi.utilities.youtube, tarkibi.utilities.agent
import tarkibi.audio.noise_reduction, tarkibi.audio.diarization, tarkibi.audio.speaker_verification

class Tarkibi:
    _BASE_DIR = tarkibi.utilities.general.BASE_DIR
    _AUDIO_RAW_PATH = f'{_BASE_DIR}/audio_raw'
    _AUDIO_SPECS_PATH = f'{_BASE_DIR}/audio_specs'
    _AUDIO_FINAL_PATH = f'{_BASE_DIR}/audio_final'
    _DURATION_MULTIPLIER = 1.2

    def __init__(self) -> None:
        self._noise_reduction = tarkibi.audio.noise_reduction._NoiseReduction()
        self._diarization = tarkibi.audio.diarization._Diarization()
        self._speaker_verification = tarkibi.audio.speaker_verification._SpeakerVerification()

        self._agent = tarkibi.utilities.agent._Agent()
        self._youtube = tarkibi.utilities.youtube._Youtube()

        self._offset = 0
        self._clips_used = []

        # will get overriden, this is just a default
        self._ac_output_path = f'{self._BASE_DIR}/audio_clips'

        tarkibi.utilities.general.make_directories([self._BASE_DIR, self._AUDIO_RAW_PATH, self._AUDIO_SPECS_PATH, self._AUDIO_FINAL_PATH])

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

    def _process_video(self, video_id: str, debug_mode: bool = False) -> None:
        self._youtube._download_video(video_id, output_path=self._AUDIO_RAW_PATH)
    
        nr_output_path = self._noise_reduction._noise_reduction(f'{self._AUDIO_RAW_PATH}/{video_id}.wav')
        self._ac_output_path = self._diarization._diarize_audio(f'{nr_output_path}/{video_id}/vocals.wav', video_id)

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

    def _join_audio_group_to_output(self, ac_output_path: str, audio_group: str, clips: list[str], output_path: str, 
                                    sample_rate: int) -> None:
        concat_file = f'{self._AUDIO_SPECS_PATH}/{audio_group}_concat.txt'
        with open(concat_file, 'w') as f:
            for clip in clips:
                f.write(f'file ../../{clip}\n')

        subprocess.run(f'ffmpeg -f concat -safe 0 -i {concat_file} -c copy {self._AUDIO_FINAL_PATH}/{audio_group}.wav', shell=True)
        subprocess.run(f'ffmpeg -i {self._AUDIO_FINAL_PATH}/{audio_group}.wav -ar {str(sample_rate)} {output_path}/{self._offset}.wav', shell=True)
        shutil.rmtree(f'{ac_output_path}/{audio_group}')
        self._offset += 1
    
    def _deep_clean(self) -> None:
        shutil.rmtree(self._BASE_DIR, ignore_errors=True)

    def build_dataset(self, author: str, reference_audio: str, target_duration: timedelta, output_path: str = 'dataset', 
                      sample_rate: int = 24000) -> None:
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        search_query = self._agent._generate_search_query(author)
        videos = self._youtube._search(search_query)

        if self._clips_used:
            videos = [video for video in videos if video['id'] not in self._clips_used]

        closest_combination = self._find_closest_combination(videos, target_duration)
        for video in closest_combination:
            self._clips_used.append(video['id'])
            self._process_video(video['id'])

        audio_groups = self._speaker_verification.find_similar_clips(self._ac_output_path, reference_audio)

        for audio_group, clips in audio_groups.items():
            self._join_audio_group_to_output(self._ac_output_path, audio_group, clips, output_path, sample_rate)

        remaining_duration = target_duration.total_seconds() - self._total_duration(output_path)
        if remaining_duration > (0.2 * target_duration.total_seconds()):
            self.build_dataset(author, reference_audio, timedelta(seconds=remaining_duration), output_path, sample_rate)
        
        self._deep_clean()
