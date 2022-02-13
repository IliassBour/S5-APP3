import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
np.set_printoptions(suppress=True)

#signal.find_peaks(), height = 5000, distance = 1000

def extract_wave():
    filename = 'note_guitare_LAd.wav'
    fe, data = wavfile.read(filename)

    return data, fe

def sinusoïdes_principales():
    # Extraire le signal et la fréquence d'échantillonage du fichier .wav
    data, fe = extract_wave()
    N = len(data)

    # Transformée de fourrier rapide
    data = np.abs(data)  # Redressage
    fft = np.fft.fft(data * np.hamming(N))  # Application d'une fenêtre de Hamming
    fft = 20 * np.log10(fft)  # Réponse en dB

    # Sinusoïdes principales (>50% de l'amplitude maximale, à chaque 1kHz)
    print(f"Valeur minimale pour considérer un sommet:\n{np.abs(max(fft) * 0.5)}")
    peaks, _ = signal.find_peaks(np.abs(fft), height=np.abs(max(fft) * 0.5), distance=1000)

    plt.figure("Résultat fft")
    plt.plot(np.abs(fft))
    plt.plot(peaks, np.abs(fft)[peaks], 'x')

    print("\nVALEURS m:")
    print(peaks[1:32])  # m des sinusoïdes principales, le peak[0] est le Gain DC

    # Fréquences normalisées
    # f = (m / N) * fe
    peaksNorm = (peaks[1:32] / N) * fe
    print("\nFRÉQUENCES NORMALISÉES:")
    print(peaksNorm)

    # Amplitudes
    print("\nAMPLITUDES:")
    print(np.abs(fft[peaks[1:32]]))

    # Phases
    print("\nPHASES:")
    print(np.angle(fft[peaks[1:32]]))

    # Fichier wav
    plt.figure("Fichier wav")
    plt.plot(data)
    plt.show()

if __name__ == "__main__":
    sinusoïdes_principales()

