from datetime import timedelta
from tarkibi import build_dataset

if __name__ == '__main__':
    build_dataset('Elon Musk', 
        reference_audio='reference.wav',
        target_duration=timedelta(minutes=10), 
        output_path='dataset', 
        sample_rate=48000,
    )