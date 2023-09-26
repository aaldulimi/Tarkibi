# Tarkibi
Build LJSpeech-like single speaker audio datasets within minutes. Requires two things, a popular, identifiable person that has a decent amount of clips on youtube of them speaking, and a reference clip of that speaker (>=10s).

## Usage 
#### 1. Clone repo and install dependencies 
```bash
git clone https://github.com/aaldulimi/Tarkibi.git
cd Tarkibi
poetry install (or 'pip install -r requirements.txt')
```

#### 2. Use the library (taken from `example.py`)
```python
from datetime import timedelta
from tarkibi import Tarkibi

tarkibi = Tarkibi()
tarkibi.build_dataset(
    'Elon Musk', 
    reference_audio='reference.wav',
    target_duration=timedelta(minutes=10), 
    output_path='dataset', 
    sample_rate=16000,
    with_transcription=True
)
```

## Dataset format 
The dataset format is the same as the LJSpeech dataset.

Audio clips are in (1s <= duration >= 15s)
```
/wavs
  | - 00001.wav
  | - 00002.wav
  | - 00003.wav
  ...
```
While transcription content is in 
```
metadata.txt

00001|This tool is amazing
00002|We going to Mars
00003|Something else.
... 
```
The full picture:
```
/{output_path}
      | -> metadata.txt
      | -> /wavs
              | -> 00001.wav
              | -> 00002.wav
              | -> 00003.wav
              | ...
```


#### Use responsibly.