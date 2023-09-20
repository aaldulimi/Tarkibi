import librosa
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import nemo.collections.asr as nemo_asr

def get_audio_files(audio_directory: str) -> list:
    audio_files = []

    for root, _, files in os.walk(audio_directory):
        for filename in files:
            if filename.endswith('.wav'):
                audio_files.append(os.path.join(root, filename))

    return audio_files

def extract_mfcc_features(audio_file: str) -> np.ndarray:
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs

def most_common_audio(audio_directory: str) -> dict:
    audio_files = get_audio_files(audio_directory)
    all_features = [extract_mfcc_features(audio_file) for audio_file in audio_files]
    max_frames = max(features.shape[1] for features in all_features)

    padded_features = [np.pad(features, ((0, 0), (0, max_frames - features.shape[1])), mode='constant') for features in all_features]
    average_feature_matrix = np.mean(padded_features, axis=0)

    similarities = []
    for features in padded_features:
        similarity = cosine_similarity(features.T, average_feature_matrix.T)
        similarities.append(similarity[0][0]) 
    
    similar_clips = [audio_files[i] for i in range(len(similarities)) if similarities[i] > 0.98]

    audio_groups = {}
    for result in similar_clips:
        video_id = result.split('/')[-2]
        if video_id not in audio_groups:
            audio_groups[video_id] = []

        audio_groups[video_id].append(result)
        
    return audio_groups

def speaker_recognition(audio_directory: str, reference_audio: str):
    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    audio_files = get_audio_files(audio_directory)

    similar_clips = []
    for audio_file in audio_files:
        is_similar = speaker_model.verify_speakers(reference_audio, audio_file)
        if is_similar:
            similar_clips.append(audio_file)

    audio_groups = {}
    for result in similar_clips:
        video_id = result.split('/')[-2]
        if video_id not in audio_groups:
            audio_groups[video_id] = []

        audio_groups[video_id].append(result)

    return audio_groups

def find_similar_clips(audio_directory: str, reference_audio: str | None = None) -> dict:
    if reference_audio:
        return speaker_recognition(audio_directory, reference_audio)
   
    return most_common_audio(audio_directory)