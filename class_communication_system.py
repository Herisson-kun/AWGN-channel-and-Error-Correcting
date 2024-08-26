import numpy as np
import matplotlib.pyplot as plt

class Communication_System:
    def __init__(self, SNR_dB, n=7, k=4):
        self.snr_db = SNR_dB
        self.n = n
        self.k = k
        self.code_words = self.hamming_code_systematic_7_4()

    def hamming_code_systematic_7_4(self):
        # Matrice génératrice systématique pour Hamming(7,4)
        # Les bits d'information sont placés en premier, suivis des bits de parité
        G = np.array([[1, 0, 0, 0, 1, 1, 0],
                    [0, 1, 0, 0, 1, 0, 1],
                    [0, 0, 1, 0, 1, 1, 1],
                    [0, 0, 0, 1, 0, 1, 1]])

        # Générer tous les mots d'information possibles (2^k pour k=4)
        information_words = np.array([list(np.binary_repr(i, width=self.k)) for i in range(2**self.k)]).astype(int)
        
        # Générer les mots de code systématiques
        code_words = np.mod(np.dot(information_words, G), 2)  # Modulo 2 pour les opérations binaires
        
        return code_words


    # Fonction pour simuler l'envoi et la réception sur un canal AWGN
    def simulate_awgn_channel(self):
        random_int = np.random.randint(0, len(self.code_words)-1)
        message = self.code_words[random_int]

        snr_linear = 10**(self.snr_db / 10.0)
        signal_power = 1
        noise_power = signal_power / snr_linear
        
        noise = np.sqrt(noise_power) * np.random.randn(len(message))
        
        received_signal = message + noise
        
        return message, received_signal


    def hard_decoding(self, received_signal): 
        naive_guess = []
        for value in received_signal:
            if value < 0.5:
                naive_guess.append(0)
            else:
                naive_guess.append(1)
            
        hamming_distances = []
        for code_word in self.code_words:
            hamming_distance = np.sum(np.abs(received_signal - code_word))
            hamming_distances.append(hamming_distance)
        guess = np.array(self.code_words[np.argmin(hamming_distances)])

        return guess, naive_guess, hamming_distances

    def N_best_hypothesis(self, N, hamming_distances):

        if N%2 != 0:
            raise ValueError("N doit être pair")
        
        arr = np.array(hamming_distances)
        indices = np.argsort(arr)
        smallest_indices = indices[:N]  # Obtenir les indices des deux plus petites valeurs
        
        best_hypothesis = []
        for index in smallest_indices:
            best_hypothesis.append(self.code_words[index])
        
        return best_hypothesis


    def likelihood(self, data, mean, variance=1):
        return (1 / (variance * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data - mean) / variance) ** 2)

    def partial_soft_decoding(self, H0, H1, received_signal):
        MAPs = np.array([])
        for i in range(len(received_signal)):
            MAP = self.likelihood(received_signal[i], mean=H0[i]) / self.likelihood(received_signal[i], mean=H1[i])
            MAPs = np.append(MAPs, MAP)
        MAPn = np.prod(MAPs)
        return MAPn

    def soft_decoding(self, best_hypothesis, received_signal):
        H = best_hypothesis[0]
        best_RAPn = None
        for i in range(1, len(best_hypothesis)):
            RAPn = self.partial_soft_decoding(H, best_hypothesis[i], received_signal)
            if RAPn < 1:
                H = best_hypothesis[i]
                best_RAPn = RAPn
        return H, best_RAPn