"""MATH 637: Meow Classifier."""

import sys
from pathlib import Path

import librosa
import librosa.display
import noisereduce as nr
import numpy as np
import plotly.graph_objects as go
from librosa import get_duration, load
from plotly.subplots import make_subplots


def gen_plotly_plots(audio_file: Path) -> None:
    """Generate plotly plots for audio file's waveform and mel-spectrogram."""
    samples, sample_rate = librosa.load(audio_file)
    # samples = nr.reduce_noise(y=samples, sr=sample_rate)

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
        rows=3,
        cols=1,
        specs=[
            [{}],
            [{"rowspan": 2}],
            [None],
        ],
        subplot_titles=("Waveform", "Mel-Spectrogram"),
    )

    # Add waveform plot
    fig.add_trace(
        go.Scatter(
            x=np.linspace(0, duration, len(samples)),
            y=samples,
            name="Waveform",
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

    # Add percentile centroid plot
    mag_spec, _ = librosa.magphase(librosa.stft(y=samples))
    spec_centroid = librosa.feature.spectral_centroid(S=mag_spec)
    spec_centroid_times = librosa.times_like(spec_centroid)
    percentile_thresh = np.percentile(mel_spec_db, 95)
    percentile_times = spec_centroid_times[
        np.max(mel_spec_db, axis=0) >= percentile_thresh
    ]
    percentile_centroid = spec_centroid.T[
        np.max(mel_spec_db, axis=0) >= percentile_thresh
    ]
    fig.add_trace(
        go.Scatter(
            x=percentile_times,
            y=percentile_centroid.T[0],
            line=dict(color='cyan', width=3),
            mode='lines+markers',
            name="Centroid",
        ),
        row=2,
        col=1,
    )

    # Add axis labels
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Mel-Frequency (Hz)", row=2, col=1)

    fig.update_layout(
        title_text=audio_file.name,
        showlegend=False,
    )
    fig.show()


def get_stats(samples: np.ndarray, sample_rate: int) -> float:
    """Potential features for extraction."""
    # Define min and max frequencies
    f_min = librosa.note_to_hz("A1")
    f_max = librosa.note_to_hz("B7")

    # Generate spec
    mel_spec = librosa.feature.melspectrogram(
        y=samples, sr=sample_rate, n_mels=256, n_fft=2048, fmin=f_min, fmax=f_max
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # type: ignore
    mel_spec_m, mel_spec_n = mel_spec_db.shape

    # RMS stats
    rms = librosa.feature.rms(y=samples)
    rms_max = np.max(rms)
    rms_mean = np.mean(rms)
    rms_sum = np.sum(rms)

    # Duration stats
    sample_count = len(samples)
    samples_top_twenty_db = np.sum(mel_spec_db >= -20)
    samples_top_forty_db = np.sum(mel_spec_db >= -40)
    samples_top_sixty_db = np.sum(mel_spec_db >= -60)
    samples_top_eighty_db = np.sum(mel_spec_db >= -80)

    # fundamental frequency
    f0, voiced_flag, voiced_probs = librosa.pyin(samples, fmin=f_min, fmax=f_max)

    # fundamental frequency stats
    f0_max = np.max(f0[np.isfinite(f0)])
    f0_min = np.min(f0[np.isfinite(f0)])
    f0_mean = np.mean(f0[np.isfinite(f0)])

    return -1


def load_files():
    """Scratch pad for mass processing."""
    file_path = Path("./dataset")
    audio_files = librosa.util.find_files(file_path, ext="wav")
    print(f"found {len(audio_files)} files")

    brush_durations = []
    food_durations = []
    iso_durations = []
    for file in audio_files:
        print(f"working on file: {file}")
        samples, sample_rate = load(file)
        duration = get_duration(y=samples, sr=sample_rate)
        match Path(file).parts[-1][0]:
            case "B":
                brush_durations.append(duration)
            case "F":
                food_durations.append(duration)
            case "I":
                iso_durations.append(duration)

    for d_list in [brush_durations, food_durations, iso_durations]:
        print(sum(d_list) / len(d_list))


def main():
    """Display audio file information."""
    if len(sys.argv) != 2:
        print("Usage: gen_stats.py <audio_file>")
        sys.exit(1)

    audio_file = Path(sys.argv[1])
    gen_plotly_plots(audio_file)

    # load_files()


if __name__ == "__main__":
    sys.exit(main())
