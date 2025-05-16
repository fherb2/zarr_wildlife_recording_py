import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.cluster import KMeans

def create_stereo_mix(left_src, right_src, sr):
    min_len = min(len(left_src), len(right_src))
    left_src = left_src[:min_len]
    right_src = right_src[:min_len]

    left = 0.7 * left_src + 0.3 * right_src
    right = 0.7 * right_src + 0.3 * left_src
    stereo = np.stack([left, right])
    return stereo, left, right

def delay_signal(signal, delay_samples):
    if delay_samples == 0:
        return signal
    elif delay_samples > 0:
        # Verzögern: vorne Nullen, hinten abschneiden
        delayed = np.concatenate([np.zeros(delay_samples), signal[:-delay_samples]])
    else:
        # Vorziehen: hinten Nullen, vorne abschneiden
        delayed = np.concatenate([signal[-delay_samples:], np.zeros(-delay_samples)])
    return delayed

def main():
    sr = 22050

    # === 1. Lade Mono-Dateien ===
    bird1, _ = librosa.load("bird1.wav", sr=sr, mono=True)
    bird2, _ = librosa.load("bird2.wav", sr=sr, mono=True)

    # === 2. Erstelle simuliertes Stereo-Mix (Links: Vogel 1, Rechts: Vogel 2) ===
    stereo, left, right = create_stereo_mix(bird1, bird2, sr)
    sf.write("stereo_mix.wav", stereo.T, sr)  # Speichern Stereo-Mix
    sf.write("original_bird1.wav", bird1, sr)
    sf.write("original_bird2.wav", bird2, sr)

    # === 3. STFT ===
    n_fft = 1024
    hop_length = 512
    S_left = librosa.stft(left, n_fft=n_fft, hop_length=hop_length)
    S_right = librosa.stft(right, n_fft=n_fft, hop_length=hop_length)

    mag_left = np.abs(S_left)
    mag_right = np.abs(S_right)
    phase_left = np.angle(S_left)
    phase_right = np.angle(S_right)

    ILD = 20 * np.log10((mag_left + 1e-6) / (mag_right + 1e-6))
    phase_diff = np.unwrap(phase_left - phase_right, axis=0)

    # === 4. Mel-Features + Clustering ===
    mel_left = librosa.feature.melspectrogram(S=mag_left**2, sr=sr, n_mels=40)
    log_mel_left = librosa.power_to_db(mel_left)

    n_frames = log_mel_left.shape[1]
    feature_vectors = []
    for i in range(n_frames):
        mel = log_mel_left[:, i]
        ild = np.mean(ILD[:, i])
        phase = np.mean(phase_diff[:, i])
        vec = np.concatenate([mel, [ild, phase]])
        feature_vectors.append(vec)
    X = np.stack(feature_vectors)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    labels = kmeans.labels_

    # === 5. Maskenbildung und Rücktransformation ===
    S_mix = S_left
    masks = []
    for cluster_id in range(2):
        mask = np.zeros_like(S_mix, dtype=np.float32)
        for t_idx in range(len(labels)):
            if labels[t_idx] == cluster_id:
                mask[:, t_idx] = 1.0
        masks.append(mask)

    # === 6. Rekonstruiere getrennte Quellen ===
    sources = []
    for i, mask in enumerate(masks):
        S_source = mask * S_mix
        x_source = librosa.istft(S_source, hop_length=hop_length)
        sources.append(x_source)
        sf.write(f"separated_source_{i+1}.wav", x_source, sr)

    # === 7. Plotten ===
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(left)
    plt.title("Stereo Left Channel (Mix)")

    for i, x in enumerate(sources):
        plt.subplot(3, 1, i + 2)
        plt.plot(x)
        plt.title(f"Getrennte Quelle {i + 1}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
