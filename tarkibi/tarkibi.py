import subprocess
from datetime import timedelta
import os
import shutil
import itertools
import wave
import tarkibi.utilities.general, tarkibi.utilities.youtube, tarkibi.utilities.agent
import tarkibi.audio.noise_reduction, tarkibi.audio.diarization, tarkibi.audio.speaker_verification, tarkibi.audio.transcription
import random
from tarkibi.utilities._config import logger

logger = logger.getChild(__name__)

class Tarkibi:
    _BASE_DIR = tarkibi.utilities.general.BASE_DIR
    _AUDIO_RAW_PATH = f'{_BASE_DIR}/audio_raw'
    _AUDIO_SPECS_PATH = f'{_BASE_DIR}/audio_specs'
    _AUDIO_FINAL_PATH = f'{_BASE_DIR}/audio_final'
    _DURATION_MULTIPLIER = 2.0

    _DEFAULT_SAMPLE_RATE = 16000
    _MAX_CLIP_DURATION = 15
    _MIN_CLIP_DURATION = 1

    def __init__(self) -> None:
        self._noise_reduction = tarkibi.audio.noise_reduction._NoiseReduction()
        self._diarization = tarkibi.audio.diarization._Diarization()
        self._speaker_verification = tarkibi.audio.speaker_verification._SpeakerVerification()
        self._transcription = tarkibi.audio.transcription._Transcription()

        self._agent = tarkibi.utilities.agent._Agent()
        self._youtube = tarkibi.utilities.youtube._Youtube()

        self._offset = 0
        self._clips_used = []

        tarkibi.utilities.general.make_directories([self._BASE_DIR, self._AUDIO_RAW_PATH, self._AUDIO_SPECS_PATH, self._AUDIO_FINAL_PATH])

    def _convert_time_to_minutes(self, time_str: str) -> float:
        """
        Function to convert a time string to minutes
        paramaters
        ----------
        time_str : str (required)
            The time string to convert

        returns
        -------
        float
            The time in minutes
        """
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

    def _find_closest_combination(self, clips: list[dict[str, str]], target_duration: timedelta) -> tuple | list:
        """
        Function to find the closest combination of audio clips to the target duration
        paramaters
        ----------
        clips : list[dict[str, str]] (required)
            A dictionary of the audio clips
        target_duration : timedelta (required)
            The target duration of the audio clips

        returns
        -------
        tuple | list
            The closest combination of audio clips to the target duration
        """
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

    def _process_video(self, video_id: str, reference_path: str, debug_mode: bool = False) -> dict[str, list] | None:
        """
        Function to process a video
        paramaters
        ----------
        video_id : str (required)
            The id of the video to process
        reference_path : str (required)
            The path to the reference audio file to compare the audio clips to
        debug_mode : bool (optional)
            Whether to keep the raw audio files or not
            Default is False

        returns
        -------
        dict[str, list] | None
            The audio groups
        """
        logger.info(f"Tarkibi _process_video: Processing video {video_id}")
        # fix age restricted video download error
        try:
            self._youtube._download_video(video_id, output_path=self._AUDIO_RAW_PATH)
        except Exception as e:
            logger.error(f"Error downloading video {video_id}: {e}")
            return 
    
        nr_output_path = self._noise_reduction._noise_reduction(f'{self._AUDIO_RAW_PATH}/{video_id}.wav')
        ac_output_path = self._diarization._diarize_audio(f'{nr_output_path}/{video_id}/vocals.wav', video_id)
        audio_groups = self._speaker_verification._speaker_recognition(ac_output_path, reference_path)

        if not debug_mode:
            os.remove(f'{self._AUDIO_RAW_PATH}/{video_id}.wav')
            shutil.rmtree(f'{nr_output_path}/{video_id}')

        return audio_groups
    
    def _single_duration(self, audio_file: str) -> float:
        """
        Function to calculate the duration of a single audio file
        paramaters
        ----------
        audio_file : str (required)
            The path to the audio file

        returns
        -------
        float
            The duration of the audio file
        """
        with wave.open(audio_file, 'rb') as wav_file:
            duration = wav_file.getnframes() / float(wav_file.getframerate())
        
        return duration
    
    def _total_duration(self, audio_directory: str) -> float:
        """
        Function to calculate the total duration of a directory of audio files
        paramaters
        ----------
        audio_directory : str (required)
            The path to the directory containing the audio files
        
        returns
        -------
        float
            The total duration of the audio files
        """
        total_duration = 0
        for filename in os.listdir(audio_directory):
            if filename.endswith(".wav"):
                total_duration += self._single_duration(f'{audio_directory}/{filename}')

        return total_duration

    def _join_audio_group_to_output(self, output_dir: str, audio_group: str, clips: list[str]) -> None:
        """
        Function to join a group of audio clips into one audio file
        paramaters
        ----------
        output_dir : str (required)
            The path to the output directory
        audio_group : str (required)
            The name of the audio group
        clips : list[str] (required)
            A list of the audio clips to join

        returns
        -------
        None
        """
        logger.info(f"Tarkibi _join_audio_group_to_output: Joining audio group {audio_group} to output")
        concat_file = f'{self._AUDIO_SPECS_PATH}/{audio_group}_concat.txt'
        with open(concat_file, 'w') as f:
            sorted_clips = sorted(clips, key=lambda x: int(x.split('/')[-1].split('.')[0]))

            for clip in sorted_clips:
                f.write(f'file ../../{clip}\n')

        subprocess.run(f'ffmpeg -f concat -safe 0 -i {concat_file} -ar {str(self._DEFAULT_SAMPLE_RATE)} {output_dir}/{audio_group}.wav', shell=True)

    def _split_audio_clips_to_dataset(self, output_path: str, audio_file: str) -> list[str]:
        """
        Function to split an audio file into clips and add the clips to the dataset
        paramaters
        ----------
        output_path : str (required)
            The path to the dataset
        audio_file : str (required)
            The path to the audio file to split

        returns
        -------
        list[str]
            A list of the audio clips added to the dataset
        """
        logger.info(f"Tarkibi _split_audio_clips_to_dataset: Splitting audio file {audio_file} to dataset")
        total_duration = self._single_duration(audio_file)

        output_files = []
        while total_duration > self._MIN_CLIP_DURATION:
            clip_duration = random.uniform(self._MIN_CLIP_DURATION, self._MAX_CLIP_DURATION)
            clip_duration = min(clip_duration, total_duration)
            
            start_time = random.uniform(0, total_duration - clip_duration)
            
            output_file = os.path.join(output_path, f"{self._offset:05d}.wav")
            subprocess.run([
                'ffmpeg',
                '-ss', str(start_time),
                '-i', audio_file,
                '-t', str(clip_duration),
                '-c:a', 'copy',
                output_file
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            total_duration -= clip_duration
            self._offset += 1
            output_files.append(output_file)

        return output_files
    
    def _deep_clean(self) -> None:
        items = os.listdir(self._BASE_DIR)
        for item in items:
            item_path = os.path.join(self._BASE_DIR, item)
            
            if os.path.isdir(item_path) and item != 'whisper.cpp':
                shutil.rmtree(item_path, ignore_errors=True)
        
    def _format_transcription_ljspeech(self, transcription_directory: str, dataset_path: str) -> None:
        """
        Function to format the transcription files to the LJSpeech format
        paramaters
        ----------
        transcription_directory : str (required)
            The path to the directory containing the transcription files
        dataset_path : str (required)
            The path to the dataset

        returns
        -------
        None
        """
        transcription_files = []
        for root, _, files in os.walk(transcription_directory):
            for filename in files:
                if filename.endswith('.txt') and filename != 'metadata.txt':
                    transcription_files.append(os.path.join(root, filename))
        
        sorted_transcription_files = sorted(transcription_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))

        with open(f'{dataset_path}/metadata.txt', 'w') as metadata_file:
            for file in sorted_transcription_files:
                file_id = file.split('/')[-1].split('.')[0]
                with open(file, 'r') as g:
                    file_content = g.read()
                    file_content = file_content.replace('\n', ' ')
                
                metadata_file.write(f'{file_id}|{file_content[1:]}\n')
                os.remove(file)
            
    def _transcribe_files(self, audio_files: list[str]) -> None:
        for audio_file in audio_files:
            output_name = audio_file.split('/')[-1].split('.')[0]
            self._transcription.transcribe_file(audio_file, output_name)
    
    def _collect_audio_clips(self, author: str, target_duration: timedelta, reference_audio: str | None = None) -> list[str]:
        """
        Function to collect audio clips for a particular person
        paramaters
        ----------
        author : str (required)
            The name of the person to collect audio clips for
        target_duration : timedelta (required)
            The target duration of the audio clips
        reference_audio : str (optional)
            The path to the reference audio file to compare the audio clips to
            Default is None

        returns
        -------
        list[str]
            A list of the audio clips collected
        """
        logger.info(f"Tarkibi _collect_audio_clips: Collecting audio clips for {author} with target duration {target_duration}")
        search_query = self._agent._generate_search_query(author)
        videos = self._youtube._search(search_query)

        if self._clips_used:
            videos = [video for video in videos if video['id'] not in self._clips_used]

        audio_groups: dict[str, list] = {}
        closest_combination = self._find_closest_combination(videos, target_duration)
        for video in closest_combination:
            self._clips_used.append(video['id'])
            audio_groups.update(self._process_video(video['id'], reference_audio))

        output_files = []
        for audio_group, clips in audio_groups.items():
            self._join_audio_group_to_output(self._AUDIO_FINAL_PATH, audio_group, clips)
            output_files.append(f'{self._AUDIO_FINAL_PATH}/{audio_group}.wav')

        logger.info(f"Tarkibi _collect_audio_clips: Output files: {output_files}")
        return output_files
    
    def _create_dataset_dirs(self, output_path: str, wav_file_output_path: str) -> None:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        if not os.path.exists(wav_file_output_path):
            os.mkdir(wav_file_output_path)
    
    def _update_sample_rate(output_path: str, sample_rate: int, all_output_files: list[str]):
        """
        Function to update the sample rate of the dataset
        paramaters
        ----------
        output_path : str (required)
            The path to the dataset
        sample_rate : int (required)
            The sample rate to update to
        all_output_files : list[str] (required)
            A list of all the audio files in the dataset

        returns
        -------
        None
        """
        final_path = f'{output_path}/wavs'
        temp_path = f'{output_path}/wavs_temp'
        file_id = file.split("/")[-1]

        os.makedirs(temp_path)
        for file in all_output_files:
            subprocess.run(f'ffmpeg -i {file} -ar {str(sample_rate)} {temp_path}/{file_id}', shell=True)

            os.remove(file)
            shutil.move(f'{temp_path}/{file_id}', f'{final_path}/{file_id}')

        shutil.rmtree(temp_path, ignore_errors=True)
            
    def build_dataset(self, author: str, reference_audio: str, target_duration: timedelta, output_path: str = 'dataset', 
                      sample_rate: int = _DEFAULT_SAMPLE_RATE, with_transcription: bool = True) -> None:
        """
        Function to build an LJSpeech-like dataset for a particular person. Uses Youtube as the source for the audio clips.
        paramaters
        ----------
        author : str (required)
            The name of the person to build the dataset for
        reference_audio : str (required)
            The path to the reference audio file to compare the audio clips to
        target_duration : timedelta (required)
            The target duration of the dataset
        output_path : str (optional)
            The path to save the dataset to.
            Default is 'dataset'
        sample_rate : int (optional)
            The sample rate of the dataset
            Default is 16KHz
        with_transcription : bool (optional)
            Whether to transcribe the dataset or not
            Default is True

        returns
        -------
        None
        """
        logger.info(f"Tarkibi _build_dataset: Building dataset for {author} with target duration {target_duration}")
        wav_file_output_path = f'{output_path}/wavs'  
        self._create_dataset_dirs(output_path, wav_file_output_path)
        
        remaining_duration = target_duration.total_seconds() - self._total_duration(wav_file_output_path)
        while remaining_duration > (0.2 * target_duration.total_seconds()):
            joined_final_paths = self._collect_audio_clips(author, target_duration, reference_audio)
            for final_path in joined_final_paths:
                self._split_audio_clips_to_dataset(wav_file_output_path, final_path)
            
            remaining_duration = target_duration.total_seconds() - self._total_duration(wav_file_output_path)
        
        all_output_files = [f'{wav_file_output_path}/{file}' for file in os.listdir(wav_file_output_path) if file.endswith('.wav')]
        if with_transcription:
            self._transcribe_files(all_output_files)
            self._format_transcription_ljspeech(output_path, output_path)

        if sample_rate != self._DEFAULT_SAMPLE_RATE:
            self._update_sample_rate(output_path, sample_rate, all_output_files)

        self._deep_clean()
