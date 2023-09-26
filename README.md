# Tarkibi
Build LJSpeech-like single speaker audio datasets within minutes. Requires two things, a popular, identifiable person that has a decent amount of clips on youtube of them speaking, and a reference clip of that speaker (>=10s, 16KHz).

### Performance 
Speaker diarization, noise reduction, speaker verification and transcription models are used. All are offline models (will possibly add option to use some cloud provider in the future).
Performance is OK on my Macbook Pro (2019) 13 inch base model.
A 10 minute target duration takes about 10 minutes to run.

## Usage 
#### 1. Clone repo and install dependencies 
```bash
git clone https://github.com/aaldulimi/Tarkibi.git
cd Tarkibi
poetry install (or 'pip install -r requirements.txt')
```

#### 2. Create .env
Create a .env environment file with the key `OPENAI_API_KEY` containing your OPENAI key (used when calling GPT4 to generate the youtube search query). 

#### 3. Use the library (taken from `example.py`)
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
While the transcription content is in 
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