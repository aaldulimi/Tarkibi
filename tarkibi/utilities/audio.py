import librosa
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import nemo.collections.asr as nemo_asr

class _SpeakerVerification:
    _FAILED_THRESHOLD = 5
    _PASSED_THRESHOLD = 5
    _NVIDIA_NEMO_MODEL_NAME = "nvidia/speakerverification_en_titanet_large"

    def __init__(self):
        pass

    def _get_audio_files(self, audio_directory: str) -> list[str]:
        audio_files = []

        for root, _, files in os.walk(audio_directory):
            for filename in files:
                if filename.endswith('.wav'):
                    audio_files.append(os.path.join(root, filename))

        return audio_files

    def _generate_audio_groups_from_files(self, audio_files: list[str]) -> dict[str, list]:
        audio_groups: dict[str, list] = {}
        for result in audio_files:
            video_id = result.split('/')[-2]
            if video_id not in audio_groups:
                audio_groups[video_id] = []

            audio_groups[video_id].append(result)
            
        return audio_groups
    
    def _extract_mfcc_features(self, audio_file: str) -> np.ndarray:
        y, sr = librosa.load(audio_file, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfccs
    
    def _most_common_audio(self, audio_directory: str) -> dict[str, list]:
        audio_files = self._get_audio_files(audio_directory)
        all_features = [self._extract_mfcc_features(audio_file) for audio_file in audio_files]
        max_frames = max(features.shape[1] for features in all_features)

        padded_features = [np.pad(features, ((0, 0), (0, max_frames - features.shape[1])), mode='constant') for features in all_features]
        average_feature_matrix = np.mean(padded_features, axis=0)

        similarities = []
        for features in padded_features:
            similarity = cosine_similarity(features.T, average_feature_matrix.T)
            similarities.append(similarity[0][0]) 
        
        similar_clips = [audio_files[i] for i in range(len(similarities)) if similarities[i] > 0.98]

        return self._generate_audio_groups_from_files(similar_clips)
    
    def _speaker_recognition(self, audio_directory: str, reference_audio: str) -> dict[str, list]:
        speaker_model: nemo_asr.models.EncDecSpeakerLabelModel = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(self._NVIDIA_NEMO_MODEL_NAME)
        audio_files = self._get_audio_files(audio_directory)

        speaker_performance = {}
        similar_clips = []
        for audio_file in audio_files:
            audio_id = audio_file.split('/')[-2]
            speaker_id = audio_id + '_' + audio_file.split('/')[-1].split('_')[0]

            # if failed > FAILED_THRESHOLD, then skip speaker
            if (speaker_id in speaker_performance) and (speaker_performance[speaker_id]['failed'] >= self._FAILED_THRESHOLD):
                continue
            
            # if passed > PASSED_THRESHOLD, then no need for further verification 
            if (speaker_id in speaker_performance) and (speaker_performance[speaker_id]['passed'] >= self._PASSED_THRESHOLD):
                similar_clips.append(audio_file)
                continue
                
            is_similar = speaker_model.verify_speakers(reference_audio, audio_file)
            if is_similar:
                similar_clips.append(audio_file)

                if speaker_id not in speaker_performance:
                    speaker_performance[speaker_id] = {
                        'passed': 1,
                        'failed': 0
                    }
                else:
                    old_pass = speaker_performance[speaker_id]['passed']
                    speaker_performance[speaker_id]['passed'] = old_pass + 1

            else:
                # not similar, i.e. different speaker
                if speaker_id not in speaker_performance:
                    speaker_performance[speaker_id] = {
                        'passed': 0,
                        'failed': 1
                    }
                else:
                    old_fail = speaker_performance[speaker_id]['failed']
                    speaker_performance[speaker_id]['failed'] = old_fail + 1

        return self._generate_audio_groups_from_files(similar_clips)
    
    def find_similar_clips(self, audio_directory: str, reference_audio: str | None = None) -> dict[str, list]:
        if reference_audio:
            return self._speaker_recognition(audio_directory, reference_audio)
        
        return self._most_common_audio(audio_directory)
    