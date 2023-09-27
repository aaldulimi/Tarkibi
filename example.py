from datetime import timedelta
from tarkibi import Tarkibi, TarkibiV2

if __name__ == "__main__":
    # tarkibi = Tarkibi()
    # tarkibi.build_dataset(
    #     "Elon Musk",
    #     reference_audio="reference.wav",
    #     target_duration=timedelta(minutes=10),
    #     output_path="dataset",
    #     sample_rate=48000,
    #     with_transcription=True,
    # )

    tarkibi_v2 = TarkibiV2()
    tarkibi_v2.build_dataset(
        "Elon Musk",
        reference_audio="reference.wav",
        target_duration=timedelta(minutes=10),
        output_path="dataset",
        sample_rate=48000,
        with_transcription=True,
    )
