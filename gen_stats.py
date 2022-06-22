"""MATH 637: Meow Classifier."""

import sys
from pathlib import Path
from unicodedata import name

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from librosa import get_duration, load
from plotly.subplots import make_subplots


def gen_plotly_plots(audio_file: Path) -> None:
    """Generate plotly plots for audio file's waveform and mel-spectrogram."""
    samples, sample_rate = librosa.load(audio_file)

    # Basic stats
    duration = librosa.get_duration(y=samples, sr=sample_rate)
    f_min = librosa.note_to_hz("A1")
    f_max = librosa.note_to_hz("B7")

    # Generate Spectrograms
    mel_spec = librosa.feature.melspectrogram(
        y=samples, sr=sample_rate, n_mels=256, n_fft=2048, fmin=f_min, fmax=f_max
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # type: ignore

    mel_spec_m, mel_spec_n = mel_spec_db.shape

    # Define subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        specs=[[{}], [{"rowspan": 3}], [None], [None]],
        subplot_titles=("Waveform", "Mel-Spectrogram"),
    )

    # Add waveform plot
    fig.add_trace(
        go.Scatter(
            x=np.linspace(0, duration, len(samples)), y=samples, name="Waveform"
        ),
        row=1,
        col=1,
    )

    # Add mel-spectrogram plot
    fig.add_trace(
        go.Heatmap(
            z=mel_spec_db,
            x=np.linspace(0, duration, mel_spec_n),
            y=np.linspace(f_min, f_max, mel_spec_m),
            colorbar={"title": "Relative dB"},
            name="Mel-Spectrogram",
        ),
        row=2,
        col=1,
    )

    # Add axis labels
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Mel-Frequency (Hz)", row=2, col=1)

    fig.update_layout(title_text=audio_file.name)
    fig.show()


def get_stats(samples: np.ndarray, sample_rate: int) -> float:
    """Scratch pad for potential feature extraction."""
    max_energy = np.max(librosa.feature.rms(y=samples))

    total_duration = librosa.get_duration(y=samples, sr=sample_rate)

    # fundamental frequency
    f0, voiced_flag, voiced_probs = librosa.pyin(samples, fmin=400, fmax=8096)

    # fundamental frequency stats
    max_f0 = np.max(f0[np.isfinite(f0)])
    min_f0 = np.min(f0[np.isfinite(f0)])
    mu_f0 = np.mean(f0[np.isfinite(f0)])

    return -1


def load_files():
    """Scratch pad for mass processing."""
    file_path = Path("./dataset")
    audio_files = librosa.util.find_files(file_path, ext="wav")
    print(f"found {len(audio_files)} files")

    brush_durs = []
    food_durs = []
    iso_durs = []
    for file in audio_files:
        print(f"working on file: {file}")
        samples, sample_rate = load(file)
        duration = get_duration(y=samples, sr=sample_rate)
        match Path(file).parts[-1][0]:
            case "B":
                brush_durs.append(duration)
            case "F":
                food_durs.append(duration)
            case "I":
                iso_durs.append(duration)

    for d_list in [brush_durs, food_durs, iso_durs]:
        print(sum(d_list) / len(d_list))


def main():
    """Display audio file information."""
    if len(sys.argv) != 2:
        print("Usage: gen_stats.py <audio_file>")
        sys.exit(1)

    audio_file = Path(sys.argv[1])
    gen_plotly_plots(audio_file)
    samples, sample_rate = librosa.load(audio_file)
    duration = librosa.get_duration(y=samples, sr=sample_rate)
    f_min = librosa.note_to_hz("A1")
    f_max = librosa.note_to_hz("B7")

    fig, ax = plt.subplots(nrows=3, sharex=True)
    librosa.display.waveshow(samples, sr=sample_rate, ax=ax[0], x_axis="time")  # type: ignore
    onsets = librosa.onset.onset_detect(
        y=samples, sr=sample_rate, units="time", backtrack=True
    )
    for onset in onsets:
        ax[0].vlines(x=onset, ymin=-0.25, ymax=0.25, color="red")  # type: ignore

    mel_spec = librosa.feature.melspectrogram(
        y=samples, sr=sample_rate, n_mels=256, n_fft=2048, fmin=f_min, fmax=f_max
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # type: ignore
    img = librosa.display.specshow(
        mel_spec_db, x_axis="time", y_axis="mel", sr=sample_rate, fmax=f_max, ax=ax[1]  # type: ignore
    )

    img2 = librosa.display.specshow(
        mel_spec_db, x_axis="time", y_axis="mel", sr=sample_rate, fmax=f_max, ax=ax[2]  # type: ignore
    )
    f0, voiced_flag, voiced_probs = librosa.pyin(samples, fmin=f_min, fmax=f_max)
    times = librosa.times_like(f0)
    cent = librosa.feature.spectral_centroid(y=samples, sr=sample_rate)
    times2 = librosa.times_like(cent)

    ax[2].plot(times, f0, label="f0", color="cyan", linewidth=2)  # type: ignore
    ax[2].plot(times2, cent.T, color="green", linewidth=2)  # type: ignore

    plt.colorbar(img, format="%+2.0f dB", ax=ax[:])  # type: ignore
    plt.xticks(np.linspace(0, duration, mel_spec_db.shape[1]))
    plt.show()

    print(f"duration: {librosa.get_duration(y=samples, sr=sample_rate):.04f} seconds")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Spec Shape: {mel_spec_db.shape}")
    print(f"f0: {f0}")
    print(f"voiced_flag: {voiced_flag}")
    print(f"voiced_probs: {voiced_probs}")

    # load_files()


if __name__ == "__main__":
    sys.exit(main())
