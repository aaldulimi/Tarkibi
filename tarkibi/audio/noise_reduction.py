import subprocess
import tarkibi.utilities.general
from tarkibi.utilities._config import logger

logger = logger.getChild(__name__)


class _NoiseReduction:
    _NR_OUTPUT_PATH = f"{tarkibi.utilities.general.BASE_DIR}/audio_nn"

    def __init__(self) -> None:
        tarkibi.utilities.general.make_directories([self._NR_OUTPUT_PATH])

    def _noise_reduction(self, audio_file_path: str, output_file_path: str) -> str:
        """
        Reduce the noise of an audio file
        parameters
        ----------
        file_path: str
            The path to the audio file to reduce the noise of
        """
        logger.info(f"Tarkibi _noise_reduction: Noise reduction on file: {audio_file_path}")

        filename_without_extension = audio_file_path.split("/")[-1].split(".")[0] 

        spleeter_cmd = f'spleeter separate -o {output_file_path} {audio_file_path}'
        spleeter_cmd += ' -f {filename}_{instrument}.{codec}'
        subprocess.run(
            spleeter_cmd, shell=True, check=True
        )

        return f'{output_file_path}/{filename_without_extension}_vocals.wav'
