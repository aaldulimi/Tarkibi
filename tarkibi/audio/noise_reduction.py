import subprocess
import tarkibi.utilities.general

class _NoiseReduction:
    _NR_OUTPUT_PATH = f'{tarkibi.utilities.general.BASE_DIR}/audio_nn'

    def __init__(self) -> None:
        tarkibi.utilities.general.make_directories([self._NR_OUTPUT_PATH])
        
    def _noise_reduction(self, file_path: str = None) -> str:
        subprocess.run(f'spleeter separate -o {self._NR_OUTPUT_PATH} {file_path}', shell=True)

        return self._NR_OUTPUT_PATH
    