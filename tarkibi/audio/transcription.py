
import subprocess
import tarkibi.utilities.general
import os
from tarkibi.utilities._config import logger

logger = logger.getChild(__name__)

class _Transcription:
    _WHISPER_DEFAULT_MODEL = 'tiny.en'
    _WHISPER_CPP_REPO = 'https://github.com/ggerganov/whisper.cpp.git'
    _TRANSCRIPTION_DIR = f'{tarkibi.utilities.general.BASE_DIR}/whisper.cpp'
    _WHISPER_ARGS = [
        '--output-txt',
        '--print-progress',
        '--no-timestamps'
    ]

    def __init__(self, model: str = _WHISPER_DEFAULT_MODEL) -> None:
        self.model = model
        self.model_path = f'{self._TRANSCRIPTION_DIR}/models/ggml-{self.model}.bin'
        
    # put this in parent and inherit
    def _get_audio_files(self, audio_directory: str) -> list[str]:
        audio_files = []

        for root, _, files in os.walk(audio_directory):
            for filename in files:
                if filename.endswith('.wav'):
                    audio_files.append(os.path.join(root, filename))

        return audio_files
    
    def _check_whisper_cpp_exists(self) -> bool:
        if os.path.exists(self._TRANSCRIPTION_DIR):
            return True

        return False

    def _check_whisper_cpp_model_exists(self) -> bool:
        if os.path.exists(f'{self.model_path}'):
            return True
        
        return False

    def _clone_whisper_cpp(self) -> None:
        logger.info(f'Tarkibi _clone_whisper_cpp: Cloning whisper.cpp repo to {self._TRANSCRIPTION_DIR}')
        subprocess.run(f'git clone {self._WHISPER_CPP_REPO} {self._TRANSCRIPTION_DIR}/', shell=True)
        
    def _download_and_make_whisper_cpp_model(self) -> None:
        logger.info(f'Tarkibi _download_and_make_whisper_cpp_model: Downloading and making whisper.cpp model: {self.model}')
        download_model_cmd = f'bash .tarkibi/whisper.cpp/models/download-ggml-model.sh {self.model}'
        subprocess.run(download_model_cmd, shell=True)
        
        # make model 
        make_model_cmd = 'cd .tarkibi/whisper.cpp && make clean && WHISPER_NO_METAL=true make'
        subprocess.run(make_model_cmd, shell=True)

    def transcribe_file(self, audio_directory: str, output_name: str) -> None:
        if not self._check_whisper_cpp_exists():
            self._clone_whisper_cpp()
            self._download_and_make_whisper_cpp_model()
        
        elif not self._check_whisper_cpp_model_exists():
            self._download_and_make_whisper_cpp_model()

        args = self._WHISPER_ARGS + [f'-of ../../dataset/{output_name}']
        args_text = ' '.join(args)

        transcription_cmd = f'cd .tarkibi/whisper.cpp/ && ./main -m models/ggml-{self.model}.bin {args_text} ../../{audio_directory}'
        subprocess.run(transcription_cmd, shell=True)
