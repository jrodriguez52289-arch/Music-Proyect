import sys
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

def dbfs(x, eps=1e-12):
    # convierte amplitud RMS a dBFS (0 dBFS = maximo digital)
    rms = np.sqrt(np.mean(x**2) + eps)
    return 20 * np.log10(rms + eps)

def check_wav(path: str, show_plots: bool = True):
    # Leer sin convertir automaticamente
    y, sr = sf.read(path, always_2d=True)
    channels = y.shape[1]

    # Convertir a mono para analisis (promedio)
    y_mono = y.mean(axis=1).astype(np.float32)

    duration = len(y_mono) / sr
    peak = float(np.max(np.abs(y_mono)))
    clipped = int(np.sum(np.abs(y_mono) >= 0.999))
    dc_offset = float(np.mean(y_mono))
    rms_db = float(dbfs(y_mono))

    # silencio: porcentaje de frames con energia muy baja
    frame_len = 2048
    hop = 512
    frames = librosa.util.frame(y_mono, frame_length=frame_len, hop_length=hop)
    frame_rms = np.sqrt(np.mean(frames**2, axis=0) + 1e-12)
    silent_frames = np.mean(frame_rms < 1e-3)  # umbral simple
    silent_pct = float(silent_frames * 100)

    print("\n=== AUDIO CHECK ===")
    print(f"File: {path}")
    print(f"Sample rate: {sr} Hz")
    print(f"Channels: {channels} (analizando en mono)")
    print(f"Duration: {duration:.2f} s")
    print(f"Peak (abs): {peak:.4f}  (ideal < 0.99)")
    print(f"Clipped samples (>=0.999): {clipped}  (ideal = 0)")
    print(f"RMS level: {rms_db:.1f} dBFS  (referencia, no hay 'perfecto')")
    print(f"DC offset (mean): {dc_offset:.6f}  (ideal cercano a 0)")
    print(f"Silence estimate: {silent_pct:.1f}% frames muy bajos (depende del audio)")

    # Reglas rapidas de salud
    warnings = []
    if duration > 30.5:
        warnings.append("Duracion > 30s (recorta a 30s para el MVP).")
    if clipped > 0:
        warnings.append("Hay clipping. Baja la ganancia y vuelve a grabar.")
    if peak < 0.05:
        warnings.append("El audio esta muy bajo. Sube ganancia (sin clipear).")
    if abs(dc_offset) > 0.01:
        warnings.append("DC offset alto. Revisa cadena de grabacion o aplica high-pass muy suave.")
    if silent_pct > 60:
        warnings.append("Mucho silencio. Recorta inicio/fin para mejorar analisis.")

    if warnings:
        print("\n ALERTAS:")
        for w in warnings:
            print(f"- {w}")
    else:
        print("\n Se ve saludable para el dataset.")

    if show_plots:
        # Waveform
        t = np.arange(len(y_mono)) / sr
        plt.figure()
        plt.plot(t, y_mono)
        plt.title("Waveform (mono)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

        # Spectrogram (dB)
        S = np.abs(librosa.stft(y_mono, n_fft=2048, hop_length=512))
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        plt.figure()
        librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="hz")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram (dB)")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python check_audio.py ruta/al/audio.wav")
        sys.exit(1)
    check_wav(sys.argv[1], show_plots=True)
