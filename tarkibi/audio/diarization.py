import os
import typing
import subprocess
from datetime import timedelta
import tarkibi.utilities.general
from simple_diarizer.diarizer import Diarizer
from pydub import AudioSegment
from tarkibi.utilities._config import logger

logger = logger.getChild(__name__)

class _Diarization:
    _AUDIO_CLIPS_PATH = f'{tarkibi.utilities.general.BASE_DIR}/audio_clips'

    def __init__(self) -> None:
        tarkibi.utilities.general.make_directories([self._AUDIO_CLIPS_PATH])

        
    def _format_time(self, seconds) -> str:
        return str(timedelta(seconds=seconds))
    
    def _diarize_audio_to_segments(self, file_path: str = None) -> list | dict[str, typing.Any]:
        logger.info(f'Tarkibi _diarize_audio_to_segments: Diarizing audio file: {file_path}')
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
        logger.info(f'Tarkibi _group_segments_by_speaker: Grouping segments by speaker: {segments}')
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
    
    def _segment_audio_clips(self, speakers: dict[str, typing.Any], file_path: str, audio_id: str) -> str:
        logger.info(f'Tarkibi _segment_audio_clips: Segmenting audio clips: {speakers}')
        audio_clips_path = f'{self._AUDIO_CLIPS_PATH}/{audio_id}'
        if not os.path.exists(audio_clips_path):
            os.mkdir(audio_clips_path)

        for speaker, info in speakers.items():
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

        return audio_clips_path
    
    def _diarize_audio(self, file_path: str, audio_id: str) -> str:
        logger.info(f'Tarkibi _diarize_audio: Diarizing audio file: {file_path}')
        segments = self._diarize_audio_to_segments(file_path)
        speakers = self._group_segments_by_speaker(segments)
        audio_clips_path = self._segment_audio_clips(speakers, file_path, audio_id)

        return audio_clips_path
    