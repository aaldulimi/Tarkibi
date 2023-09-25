import subprocess
import tarkibi.utilities.general
from tarkibi.utilities._config import logger

logger = logger.getChild(__name__)

class _NoiseReduction:
    _NR_OUTPUT_PATH = f'{tarkibi.utilities.general.BASE_DIR}/audio_nn'

    def __init__(self) -> None:
        tarkibi.utilities.general.make_directories([self._NR_OUTPUT_PATH])
        
    def _noise_reduction(self, file_path: str = None) -> str:
        logger.info(f'Tarkibi _noise_reduction: Noise reduction on file: {file_path}')
        subprocess.run(f'spleeter separate -o {self._NR_OUTPUT_PATH} {file_path}', shell=True)

        return self._NR_OUTPUT_PATH
    