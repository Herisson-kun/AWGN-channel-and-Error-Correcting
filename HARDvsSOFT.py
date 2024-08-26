from class_communication_system import Communication_System
import numpy as np
import matplotlib.pyplot as plt

# Paramètres de simulation
N = 7  # Taille du message
snr_dB = 10  # Rapport signal sur bruit en dB
CS = Communication_System(snr_dB, n=7, k=4)
print("Mots de code générés:", CS.code_words, "\n")

# Simuler le canal
message, received_signal = CS.simulate_awgn_channel()
print("Message envoyé :", message)
print("Signal reçu bruité :", np.round(received_signal, 2), "\n")

# Décodage
noise_values = message - received_signal

guess, naive_guess, hamming_distances = CS.hard_decoding(received_signal)

# Calcul des erreurs
print("erreur quadratique moyenne:", np.mean(noise_values**2), "Erreur max:", np.max(np.abs(noise_values)), "at", np.argmax(np.abs(noise_values)))
naive_error = message - naive_guess
naive_errors = []
index = 0
for index in naive_error:
    if noise_values[index] != 0:
        naive_errors.append((message[index], received_signal[index]))
print("Nombre d'erreurs:", len(naive_errors),"\n")

print("Hamming distances:", np.round(hamming_distances, 2))
best_hypothesis = CS.N_best_hypothesis(16, hamming_distances)
print("Meilleures hypothèses:", best_hypothesis, "\n")
soft_decoding_guess, RAPn = CS.soft_decoding(best_hypothesis, received_signal)
print("Message:", message, "Soft Decoding Guess:", soft_decoding_guess, RAPn, "Hard Decoding Guess:", guess)



# Tracé du message et du signal bruité
plt.figure(figsize=(10, 6))
plt.stem(message[:100], linefmt='b-', markerfmt='bo', basefmt='r-', label='Message envoyé')
plt.stem(received_signal[:100], linefmt='g-', markerfmt='go', basefmt='r-', label='Signal reçu avec AWGN')
plt.legend()
plt.title('Comparaison du message envoyé et du signal reçu avec bruit AWGN')
plt.xlabel('Indice du symbole')
plt.ylabel('Valeur du symbole')
plt.grid(True)
plt.show()
