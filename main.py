import matplotlib.pyplot as plt
import numpy
import numpy as np
from scipy import signal
from scipy.io import wavfile
np.set_printoptions(suppress=True)
FACT_DO = 0.561
FACT_RE = 0.629
FACT_MI = 0.707
FACT_FA = 0.749
FACT_SOL = 0.841
FACT_LA = 0.943
FACT_SI = 1.06


def extract_wave(filename):
    fe, data = wavfile.read(filename)

    return data, fe

def filtre_moyenne_mobile(data_length):
    w = np.pi / 1000  # fréquence normalisé
    N = find_N(w)
    wn = np.arange(0, np.pi, np.pi/data_length)

    hm = []
    for index in wn:
        hm.append(moyenne_mobile(N,index))

    return hm

def moyenne_mobile(N, w):
    if w:
        return 1/N*(1-np.exp(-1j*N*w))/(1-np.exp(-1j*w))
    else:
        return 1/N

def find_N(w):
    N = 1
    while not (np.sqrt(2)/2 - 0.0001 < np.abs(moyenne_mobile(N, w)) < np.sqrt(2)/2 + 0.0001):
            N = N+1
    return N

def somme_des_sinus(peaks, ym, facteur):
    somme = 0
    for peak in peaks:
        somme += ym[peak]
    return facteur * somme

def creer_note(f, fact, fe, amp, phi, env):
    somme = np.zeros(len(env))
    n = np.arange(0, len(env))
    for i in range(len(amp)):
        somme = somme + amp[i] * np.sin(2 * np.pi * f[i] * fact * n/fe + phi[i])

    somme = somme/max(amp)
    #plt.figure("somme")
    #plt.plot(somme)
    #plt.show()
    return (env * somme)[:35000]


def sinusoïdes_principales():
    ####################################################################################################################
    # Extraire le signal et la fréquence d'échantillonage du fichier .wav
    ####################################################################################################################
    data, fe = extract_wave("note_guitare_LAd.wav")
    N = len(data)  #Nombre d'échantillons
    K = find_N(np.pi / 1000)  #Nombre de coefficients du filtre
    hm = filtre_moyenne_mobile(N)

    ####################################################################################################################
    # Transformée de fourrier rapide
    ####################################################################################################################
    data = np.abs(data)  # Redressage
    xm = np.fft.fft(data * np.hamming(N))  # Application d'une fenêtre de Hamming
    ym = hm * xm
    ym = 20 * np.log10(ym)  # Réponse en dB

    ####################################################################################################################
    # Sinusoïdes principales (>42% de l'amplitude maximale, à chaque 1kHz)
    ####################################################################################################################
    min_sommet = np.abs(max(ym) * 0.42)
    print(f"Valeur minimale pour considérer un sommet:\n{min_sommet}")
    peaks, _ = signal.find_peaks(np.abs(ym), height=min_sommet, distance=1000)

    ####################################################################################################################
    # Affichage
    ####################################################################################################################
    plt.figure("Résultat fft")
    freq = [(i / N) * fe for i in range(len(ym))]
    plt.plot(freq, np.abs(ym))
    plt.plot(peaks * fe / len(ym), np.abs(ym)[peaks], 'x')

    print(f"\nNombre de peaks:\n{len(peaks)}")

    print("\nVALEURS m:")
    print(peaks[1:33])  # m des sinusoïdes principales, le peak[0] est le Gain DC

    ### Fréquences normalisées
    # f = (m / N) * fe
    peaksNorm = (peaks[1:33] / N) * fe
    print("\nFRÉQUENCES NORMALISÉES:")
    print(peaksNorm)

    ### Amplitudes
    print("\nAMPLITUDES:")
    print(np.abs(ym[peaks[1:33]]))

    ### Phases
    print("\nPHASES:")
    print(np.angle(ym[peaks[1:33]]))

    ####################################################################################################################
    # Enveloppe
    ####################################################################################################################
    #hn = np.fft.ifft(hm * np.hamming(len(hm)))
    hn = np.full(K, 1/K)

    plt.figure("hn")
    plt.plot(hn)

    enveloppe = np.convolve(data, hn)

    plt.figure("Enveloppe")
    plt.plot(enveloppe)

    ####################################################################################################################
    # Synthèse des sons
    ####################################################################################################################

    sinus_do = creer_note(peaksNorm, FACT_DO, fe, np.abs(ym[peaks[1:33]]), np.angle(ym[peaks[1:33]]), enveloppe)
    sinus_re = creer_note(peaksNorm, FACT_RE, fe, np.abs(ym[peaks[1:33]]), np.angle(ym[peaks[1:33]]), enveloppe)
    sinus_mi = creer_note(peaksNorm, FACT_MI, fe, np.abs(ym[peaks[1:33]]), np.angle(ym[peaks[1:33]]), enveloppe)
    sinus_fa = creer_note(peaksNorm, FACT_FA, fe, np.abs(ym[peaks[1:33]]), np.angle(ym[peaks[1:33]]), enveloppe)
    sinus_sol = creer_note(peaksNorm, FACT_SOL, fe, np.abs(ym[peaks[1:33]]), np.angle(ym[peaks[1:33]]), enveloppe)
    sinus_la = creer_note(peaksNorm, FACT_LA, fe, np.abs(ym[peaks[1:33]]), np.angle(ym[peaks[1:33]]), enveloppe)
    sinus_si = creer_note(peaksNorm, FACT_SI, fe, np.abs(ym[peaks[1:33]]), np.angle(ym[peaks[1:33]]), enveloppe)
    silence = np.zeros(len(enveloppe))[:15000]

    #symphonie = [sinus_sol, sinus_sol, sinus_sol, sinus_mi, silence, sinus_fa, sinus_fa, sinus_fa, sinus_re].join()
    #symphonie = sinus_sol + sinus_sol + sinus_sol + sinus_mi + silence + sinus_fa + sinus_fa + sinus_fa + sinus_re
    symphonie = numpy.concatenate([sinus_sol, sinus_sol, sinus_sol, sinus_mi, silence, sinus_fa, sinus_fa, sinus_fa, sinus_re], axis=None)

    plt.figure("Symphonie")
    plt.plot(symphonie)

    # Créer nouveaux fichier .wav
    scaled = np.int16(sinus_do / np.max(np.abs(sinus_do)) * 32767)
    wavfile.write('do.wav', 44100, scaled)
    scaled = np.int16(sinus_re / np.max(np.abs(sinus_re)) * 32767)
    wavfile.write('re.wav', 44100, scaled)
    scaled = np.int16(sinus_mi / np.max(np.abs(sinus_mi)) * 32767)
    wavfile.write('mi.wav', 44100, scaled)
    scaled = np.int16(sinus_fa / np.max(np.abs(sinus_fa)) * 32767)
    wavfile.write('fa.wav', 44100, scaled)
    scaled = np.int16(sinus_sol / np.max(np.abs(sinus_sol)) * 32767)
    wavfile.write('sol.wav', 44100, scaled)
    scaled = np.int16(sinus_la / np.max(np.abs(sinus_la)) * 32767)
    wavfile.write('la.wav', 44100, scaled)
    scaled = np.int16(sinus_si / np.max(np.abs(sinus_si)) * 32767)
    wavfile.write('si.wav', 44100, scaled)
    scaled = np.int16(symphonie / np.max(np.abs(symphonie)) * 32767)
    wavfile.write('symph.wav', 44100, scaled)

    # Fichier wav
    plt.figure("Fichier wav")
    plt.plot(data)
    plt.show()

def son_corrompu():
    data, fe = extract_wave("note_basson_plus_sinus_1000_Hz.wav")

    #Filtre passe-bas N=6000, fc = 40Hz
    N = 6000 #Ordre du filtre
    fc=40
    K = np.ceil((2*fc*N/fe)+1)

    h_n_bas = lambda n: (1/N)*np.sin(np.pi * n * K/N) / np.sin(np.pi * n/N) if n else K/N

    #Filtre coupe-bande: 960-1040 Hz
    omega_0 = 1000

    dirac = lambda n: 1 if n == 0 else 0
    h_n = lambda n: dirac(n) - h_n_bas(n)*np.cos(2*np.pi*omega_0 /fe*n)

    #Signal filtrée
    repImp = []
    for n in range(int(-N/2), int(N/2)):
        repImp.append(h_n(n))

    signal_filtree = np.convolve(data*np.hamming(len(data)), repImp)

    #Répétition
    for n in range(6):
        signal_filtree = np.convolve(signal_filtree, repImp)

    # Créer nouveau fichier .wav
    wavfile.write('basson.wav', fe, signal_filtree.astype(np.int16))

    #Réponse en fréquence du filtre-coupe bande
    repFreq = np.fft.fft(repImp[int(len(repImp)/2):])
    freq=np.arange(int(len(repFreq)))*fe/N*2

    plt.figure("Réponse en fréquence du filtre coupe-bande")
    plt.subplot(2, 1, 1).set_title("Amplitude")
    plt.plot(freq[:int(len(freq)/8)], np.abs(repFreq[:int(len(repFreq)/8)]))
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("H(m) (dB)")

    plt.subplot(2, 1, 2).set_title("Phase")
    plt.plot(freq[:int(len(freq)/8)], np.angle(repFreq[:int(len(repFreq)/8)]))
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Phase (radian)")

    #Signal de synthèse
    signal_filtree_db = 20*np.log10(signal_filtree)
    plt.figure("Basson synthèse")
    plt.plot(signal_filtree_db)
    plt.xlabel("fréquence (Hz)")
    plt.ylabel("Magnitude (dB)")

    #Signal original
    data_db = 20*np.log10(data)
    plt.figure("Basson original")
    plt.plot(data_db)
    plt.xlabel("fréquence (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.show()


if __name__ == "__main__":
    #sinusoïdes_principales()
    #filtre_RIF_passe_bas()
    son_corrompu()
    plt.show()
    #son_corrompu()

