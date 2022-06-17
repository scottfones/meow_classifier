"""MATH 637: Meow Classifier."""

import sys
from pathlib import Path

import librosa
from librosa import get_duration, load
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def get_stats(samples: np.ndarray, sample_rate: int) -> float:
    max_energy = np.max(librosa.feature.rms(y=samples))

    total_duration = librosa.get_duration(y=samples, sr=sample_rate)

    # fundamental frequency
    f0, voiced_flag, voiced_probs = librosa.pyin(samples, fmin=400, fmax=8096)

    # mean fundamental frequency
    mu_f0 = np.mean(f0[np.isfinite(f0)])

    return -1

def load_files():
    file_path = Path('./dataset')
    audio_files = librosa.util.find_files(file_path, ext='wav')
    print(f"found {len(audio_files)} files")

    brush_durs = []
    food_durs = []
    iso_durs = []
    for file in audio_files:
        print(f"working on file: {file}")
        samples, sample_rate = load(file)
        duration = get_duration(y=samples, sr=sample_rate)
        match Path(file).parts[-1][0]:
            case 'B':
                brush_durs.append(duration)
            case 'F':
                food_durs.append(duration)
            case 'I':
                iso_durs.append(duration)

    for d_list in [brush_durs, food_durs, iso_durs]:
        print(sum(d_list) / len(d_list))

def main():
    """Display audio file information."""
    if len(sys.argv) != 2:
        print("Usage: gen_stats.py <audio_file>")
        sys.exit(1)

    audio_file = Path(sys.argv[1])
    samples, sample_rate = librosa.load(audio_file)
    duration = librosa.get_duration(y=samples, sr=sample_rate)

    fig, ax = plt.subplots(nrows=3, sharex=True)
    librosa.display.waveshow(samples, sr=sample_rate, ax=ax[0], x_axis='time')
    onsets = librosa.onset.onset_detect(y=samples, sr=sample_rate, units='time', backtrack=True)
    for onset in onsets:
        ax[0].vlines(x=onset, ymin=-0.25, ymax=0.25, color="red")

    mel_spec = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_fft=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sample_rate,
                                   fmax=4096, ax=ax[1])

    img2 = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sample_rate,
                                    fmax=4096, ax=ax[2])
    f0, voiced_flag, voiced_probs = librosa.pyin(samples, fmin=200, fmax=4096)
    times = librosa.times_like(f0)
    cent = librosa.feature.spectral_centroid(y=samples, sr=sample_rate)
    times2 = librosa.times_like(cent)

    ax[2].plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax[2].plot(times2, cent.T, color='green', linewidth=3)


    plt.colorbar(img, format='%+2.0f dB', ax=ax[:])
    plt.xticks(np.arange(0, duration + .1, .1))
    plt.show()

    print(f"duration: {librosa.get_duration(y=samples, sr=sample_rate):.04f}")
    print(f"Sample Rate: {sample_rate} ({type(sample_rate)})")

    # load_files()

if __name__ == "__main__":
    sys.exit(main())
