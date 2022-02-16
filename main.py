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

def creer_note(f, fact, fe, amp, phi, env):
    somme = np.zeros(len(env))
    n = np.arange(0, len(env))
    for i in range(len(amp)):
        somme = somme + amp[i] * np.sin(2 * np.pi * f[i] * fact * n/fe + phi[i])

    somme = somme/max(amp)
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
    ym_og = ym
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
    plt.figure(1)
    freq = [(i / N) * fe for i in range(len(ym))]
    plt.plot(freq, np.abs(ym))
    plt.plot(peaks * fe / len(ym), np.abs(ym)[peaks], 'x')
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Magnitude (dB)")

    print(f"\nNombre de peaks:\n{len(peaks)}")

    print("\nVALEURS m:")
    print(peaks[1:33])  # m des sinusoïdes principales, le peak[0] est le Gain DC

    ### Fréquences normalisées
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
    hn = np.full(K, 1/K)

    plt.figure("Fonction de réponse")
    hm = np.fft.fft(hn)
    plt.plot(20*np.log10(np.abs(hm)))
    plt.ylabel("Amplitude (dB)")

    enveloppe = np.convolve(data, hn)

    temps = [i/fe for i in range(len(enveloppe))]
    plt.figure("Enveloppe")
    plt.plot(temps, enveloppe)
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")

    ####################################################################################################################
    # Synthèse des sons
    ####################################################################################################################

    sinus_do = creer_note(peaksNorm, FACT_DO, fe, np.abs(ym_og[peaks[1:33]]), np.angle(ym_og[peaks[1:33]]), enveloppe)
    sinus_re = creer_note(peaksNorm, FACT_RE, fe, np.abs(ym_og[peaks[1:33]]), np.angle(ym_og[peaks[1:33]]), enveloppe)
    sinus_mi = creer_note(peaksNorm, FACT_MI, fe, np.abs(ym_og[peaks[1:33]]), np.angle(ym_og[peaks[1:33]]), enveloppe)
    sinus_fa = creer_note(peaksNorm, FACT_FA, fe, np.abs(ym_og[peaks[1:33]]), np.angle(ym_og[peaks[1:33]]), enveloppe)
    sinus_sol = creer_note(peaksNorm, FACT_SOL, fe, np.abs(ym_og[peaks[1:33]]), np.angle(ym_og[peaks[1:33]]), enveloppe)
    sinus_la = creer_note(peaksNorm, FACT_LA, fe, np.abs(ym_og[peaks[1:33]]), np.angle(ym_og[peaks[1:33]]), enveloppe)
    sinus_si = creer_note(peaksNorm, FACT_SI, fe, np.abs(ym_og[peaks[1:33]]), np.angle(ym_og[peaks[1:33]]), enveloppe)
    sinus_lad = creer_note(peaksNorm, 1, fe, np.abs(ym_og[peaks[1:33]]), np.angle(ym_og[peaks[1:33]]), enveloppe)
    silence = np.zeros(len(enveloppe))[:15000]

    symphonie = numpy.concatenate([sinus_sol, sinus_sol, sinus_sol, sinus_mi, silence, sinus_fa, sinus_fa, sinus_fa, sinus_re], axis=None)

    plt.figure("Symphonie")
    plt.plot(symphonie)

    ### LA dièse
    la_d_synth = np.fft.fft(sinus_lad)
    freq_synth = [(i / int(len(la_d_synth))/2) * fe for i in range(int(len(la_d_synth)/2))]
    plt.figure("LA# Synth")
    plt.plot(freq_synth, 20*np.log10(np.abs(la_d_synth)[:int(len(la_d_synth)/2)]))
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Magnitude (dB)")
    ###


    # Créer nouveaux fichier .wav
    wavfile.write('do.wav', fe, sinus_do.astype(np.int16))
    wavfile.write('re.wav', fe, sinus_re.astype(np.int16))
    wavfile.write('mi.wav', fe, sinus_mi.astype(np.int16))
    wavfile.write('fa.wav', fe, sinus_fa.astype(np.int16))
    wavfile.write('sol.wav', fe, sinus_sol.astype(np.int16))
    wavfile.write('la.wav', fe, sinus_la.astype(np.int16))
    wavfile.write('si.wav', fe, sinus_si.astype(np.int16))
    wavfile.write('symph.wav', fe, symphonie.astype(np.int16))
    wavfile.write('la_d.wav', fe, sinus_lad.astype(np.int16))

    # Fichier wav
    plt.figure("Fichier wav")
    plt.plot(data)

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
    repImp = [] # h[n]
    for n in range(int(-N/2), int(N/2)):
        repImp.append(h_n(n))

    signal_filtree = np.convolve(data*np.hamming(len(data)), repImp)

    #Répétition
    for n in range(6):
        signal_filtree = np.convolve(signal_filtree, repImp)

    # Créer nouveau fichier .wav
    wavfile.write('basson_synthese.wav', fe, signal_filtree.astype(np.int16))

    #Réponse impulsionnelle
    plt.figure("Réponse impulsionnelle coupe-bande")
    plt.plot(range(int(-N/2), int(N/2)), repImp)
    plt.xlabel("Nombre d'échantillon")
    plt.ylabel("Amplitude de h(n)")

    #Réponse à une sinusoïde de 1 kHz
    n_sin = np.arange(0, 1024)
    sin_1kHz = np.sin(2*np.pi*1000*n_sin)
    rep_1kHz = np.convolve(sin_1kHz, repImp)

    for n in range(4):
        rep_1kHz = np.convolve(rep_1kHz, repImp)
    freq_rep_1kHz = np.arange(int(len(n_sin)+5*6000-5)) #nombre de répétition * N_hn - nombre de répétition +N_sin

    plt.figure("Réponse sinusoïde de 1 kHz")
    plt.plot(freq_rep_1kHz, rep_1kHz)
    plt.xlabel("Échantillon")
    plt.ylabel("Amplitude")

    #Réponse en fréquence du filtre-coupe bande
    repFreq = np.fft.fft(repImp[int(len(repImp) / 2):]) # H[m]
    freq=np.arange(int(len(repFreq)))*fe/N*2

    plt.figure("Amplitude de la réponse en fréquence du filtre coupe-bande")
    plt.plot(freq[:int(len(freq)/8)], np.abs(repFreq[:int(len(repFreq)/8)]))
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude de H(m) (dB)")

    plt.figure("Phase de la réponse en fréquence du filtre coupe-bande")
    plt.plot(freq[:int(len(freq)/8)], np.angle(repFreq[:int(len(repFreq)/8)]))
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Phase de H(m) (radian)")

    #Signal de synthèse
    signal_filtree_db = 20*np.log10(signal_filtree)
    plt.figure("Signaux du basson après filtrage")
    plt.plot(signal_filtree_db)
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Magnitude (dB)")

    #Signal original
    data_db = 20*np.log10(data)
    plt.figure("Signaux du basson avant filtrage")
    plt.plot(data_db)
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Magnitude (dB)")


if __name__ == "__main__":
    sinusoïdes_principales()
    son_corrompu()
    plt.show()
