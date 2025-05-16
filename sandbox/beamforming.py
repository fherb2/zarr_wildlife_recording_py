import numpy as np
import soundfile as sf
import pyroomacoustics as pra

# === Parameter ===
sr = 16000
room_dim = [6, 4]  # Raumgröße in Metern
mic_locs = np.array([[2.0, 2.0], [2.1, 2.0]]).T  # 2 Mikrofone (Stereo), 10 cm Abstand
source_1_loc = [1.0, 1.0]  # Vogel 1
source_2_loc = [4.0, 3.0]  # Vogel 2

# === 1. Lade Audiosignale ===
bird1, sr1 = sf.read("bird1.wav")
bird2, sr2 = sf.read("bird2.wav")
assert sr1 == sr2 == sr, "Samplerates müssen übereinstimmen."

min_len = min(len(bird1), len(bird2))
bird1 = bird1[:min_len]
bird2 = bird2[:min_len]

# === 2. Raum mit 2 Quellen und 2 Mikrofonen ===
room = pra.ShoeBox(room_dim, fs=sr, absorption=0.4, max_order=3)

# Mikrofone
room.add_microphone_array(pra.MicrophoneArray(mic_locs, room.fs))

# Quellen mit Signalen
room.add_source(source_1_loc, signal=bird1)
room.add_source(source_2_loc, signal=bird2)

# === 3. Simulation ===
room.simulate()

# Ergebnis: multikanaliges Mikrofonarray-Signal (2 x N)
array_signals = room.mic_array.signals
sf.write("simulated_stereo_mix.wav", array_signals.T, sr)

# === 4. Beamforming mit MVDR auf Quelle 1 ===
R = pra.linear_2D_array([2.05, 2.0], 2, 0.1, 0)  # Referenzpunkt: zwischen den Mics
room.mic_array = pra.Beamformer(mic_locs, room.fs, N=1024, Lg=256)
room.mic_array.rake_mvdr_filters(room.sources[:1], Rn=None)

# Anwendung des Beamformers
output = room.mic_array.process()
sf.write("beamformed_mvdr_output.wav", output, sr)

print("Fertig! Es wurden geschrieben:")
print("- simulated_stereo_mix.wav (Simulierter Mix)")
print("- beamformed_mvdr_output.wav (Gefilterte Quelle 1)")
