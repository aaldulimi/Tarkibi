import os
import typing
import subprocess
from datetime import timedelta
import tarkibi.utilities.general
from simple_diarizer.diarizer import Diarizer
import logging
from pydub import AudioSegment

logging.basicConfig(
    filename='tarkibi.log',  
    level=logging.INFO,  
    format='%(asctime)s [%(levelname)s] %(message)s' 
)

class _Diarization:
    _AUDIO_CLIPS_PATH = f'{tarkibi.utilities.general.BASE_DIR}/audio_clips'

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        tarkibi.utilities.general.make_directories([self._AUDIO_CLIPS_PATH])

        
    def _format_time(self, seconds) -> str:
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
        
        self._logger.info(f'Tarkibi _group_segments_by_speaker: {speakers}')
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
                )

                i += 1
    
    def _segment_audio_clips_v2(self, speakers: dict[str, typing.Any], file_path: str, audio_id: str) -> None:
        for speaker, info in speakers.items():
            audio_clips_path = f'{self._AUDIO_CLIPS_PATH}/{audio_id}'
            if not os.path.exists(audio_clips_path):
                os.mkdir(audio_clips_path)
                
            segments = []

            for _, segment in enumerate(info['segments']):
                segment_start = segment['start'] * 1000  
                segment_end = segment['end'] * 1000      

                audio = AudioSegment.from_wav(file_path)
                segment_audio = audio[segment_start:segment_end]

                segments.append(segment_audio)

            concatenated_audio = sum(segments)
            output_file = f'{audio_clips_path}/{speaker}.wav'
            concatenated_audio.export(output_file, format="wav")
    
    def _diarize_audio(self, file_path: str, audio_id: str) -> str:
        segments = self._diarize_audio_to_segments(file_path)
        speakers = self._group_segments_by_speaker(segments)
        self._segment_audio_clips_v2(speakers, file_path, audio_id)

        return f'{self._AUDIO_CLIPS_PATH}/{audio_id}'
    