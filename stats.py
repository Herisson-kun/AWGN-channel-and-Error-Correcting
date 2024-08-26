from class_communication_system import Communication_System
from tqdm import tqdm
import numpy as np

N = 100000
CS = Communication_System(SNR_dB=15, n=7, k=4)
sample = {}
error_naive = 0
error_hard = 0
error_soft = 0

for i in tqdm(range (0, N)):
    message, received_signal = CS.simulate_awgn_channel()
    hard_decoding_guess, naive_guess, hamming_distances = CS.hard_decoding(received_signal)
    best_hypothesis = CS.N_best_hypothesis(2, hamming_distances)
    soft_decoding_guess, RAPn = CS.soft_decoding(best_hypothesis, received_signal)
    if not np.array_equal(message, naive_guess):
        error_naive += 1
    if not np.array_equal(message, hard_decoding_guess):
        error_hard += 1
    if not np.array_equal(message, soft_decoding_guess):
        error_soft += 1
    sample.update({i: {"message": message, "received_signal": received_signal, "naive_guess": naive_guess, "hard_decoding_guess": hard_decoding_guess, "soft_decoding_guess": soft_decoding_guess, "RAPn": RAPn}})

print("Errors : Naive:", error_naive/N, "Hard:", error_hard/N, "Soft:", error_soft/N)
