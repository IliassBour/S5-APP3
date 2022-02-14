import matplotlib.pyplot as plt
import numpy
import numpy as np
from scipy import signal
from scipy.io import wavfile
np.set_printoptions(suppress=True)

#signal.find_peaks(), height = 5000, distance = 1000


def extract_wave(filename):
    fe, data = wavfile.read(filename)

    return data, fe

def filtre_RIF_passe_bas():
    w = np.pi / 1000  # fréquence normalisé
    N = find_N(w)
    wn = np.arange(0, 2*np.pi, w)


    hn = []
    for index in wn:
        hn.append(moyenne_mobile(N,index))

    plt.figure("Hn")
    plt.stem(hn)
    plt.semilogy()

    return hn

def moyenne_mobile(N, w):
    return 1/N*(1-np.exp(-1j*N*w))/(1-np.exp(-1j*w))

def find_N(w):
    N=1
    while not (np.sqrt(2)/2 - 0.0001 < np.abs(moyenne_mobile(N, w)) < np.sqrt(2)/2 + 0.0001):
            N=N+1
    return N


def sinusoïdes_principales():
    # Extraire le signal et la fréquence d'échantillonage du fichier .wav
    data, fe = extract_wave("note_guitare_LAd.wav")
    N = len(data)
    #hm = [moyenne_mobile(i, K), for i in N]
    #hm = np.pad(hm, (0, np.abs(len(N) - len(data))), "constant")

    # Transformée de fourrier rapide
    data = np.abs(data)  # Redressage
    xm = np.fft.fft(data * np.hamming(N))  # Application d'une fenêtre de Hamming
    #xm = 20 * np.log10(xm)  # Réponse en dB
    #ym = hm * xm
    ym = xm
    ym = 20 * np.log10(ym)  # Réponse en dB

    # Sinusoïdes principales (>50% de l'amplitude maximale, à chaque 1kHz)
    print(f"Valeur minimale pour considérer un sommet:\n{np.abs(max(ym) * 0.5)}")
    peaks, _ = signal.find_peaks(np.abs(ym), height=np.abs(max(ym) * 0.5), distance=1000)

    plt.figure("Résultat fft")
    plt.plot(np.abs(ym))
    plt.plot(peaks, np.abs(ym)[peaks], 'x')

    print("\nVALEURS m:")
    print(peaks[1:32])  # m des sinusoïdes principales, le peak[0] est le Gain DC

    # Fréquences normalisées
    # f = (m / N) * fe
    peaksNorm = (peaks[1:32] / N) * fe
    print("\nFRÉQUENCES NORMALISÉES:")
    print(peaksNorm)

    # Amplitudes
    print("\nAMPLITUDES:")
    print(np.abs(ym[peaks[1:32]]))

    # Phases
    print("\nPHASES:")
    print(np.angle(ym[peaks[1:32]]))

    # Fichier wav
    plt.figure("Fichier wav")
    plt.plot(data)
    plt.show()

def son_corrompu():
    data, fe = extract_wave("note_basson_plus_sinus_1000_Hz.wav")
    #data = np.abs(data) #Redréssage
    n_e = len(data) #Nombre d'échantillons
    print(n_e)
    print(f"fe:{fe}")

    #Filtre passe-bas d'ordre N=6000, fc = 40Hz
    N = 6000 #Ordre du filtre
    K = (2*40*N/fe)+1
    print(f"K:{K}")
    h_n_bas = lambda n: (1/N) * np.sin(np.pi * n * K/N) / np.sin(np.pi * n/N) if n else K/N

    #Filtre coupe-bande: 960-1040 Hz
    dirac = lambda n: 1 if n == 0 else 0  # Impulsion de Dirac
    omega_0 = 1000
    omega_1 = 40

    h_n = lambda n: dirac(n) - 2 * h_n_bas(n) * np.cos(omega_0 * n)

    #Signal filtrée
    repImp = []
    for n in data:
        repImp.append(h_n(n))
    signal_filtree = numpy.convolve(data, repImp)

    #Répétition
    for i in range(1):
        repImp = []
        for n in signal_filtree:
            repImp.append(h_n(n))
        signal_filtree = numpy.convolve(signal_filtree, repImp)

    plt.figure("Signal filtrée")
    plt.plot(signal_filtree)

    # Créer nouveau fichier .wav
    scaled = np.int16(signal_filtree / np.max(np.abs(signal_filtree)) * 32767)
    wavfile.write('test.wav', 44100, scaled)

    # Fichier .wav
    plt.figure("Fichier wav")
    plt.plot(data)
    plt.show()


if __name__ == "__main__":
    sinusoïdes_principales()
    filtre_RIF_passe_bas()
    plt.show()
    #son_corrompu()

