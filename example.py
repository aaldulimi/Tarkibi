from datetime import timedelta
from tarkibi import build_dataset

target_duration = timedelta(minutes=10)
build_dataset('Elon Musk', target_length=target_duration, output_path='dataset', sample_rate=24000)