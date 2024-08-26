from class_communication_system import Communication_System
import numpy as np
from multiprocessing import Pool, cpu_count
from time import time

def process_chunk(start, end, CS):
    # Initialize error counters for this chunk
    error_naive = 0
    error_hard = 0
    error_soft = 0
    
    for i in range(start, end):
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
    
    return error_naive, error_hard, error_soft

def main():
    N = 100000
    N_process = cpu_count()  # Number of processes to use
    print("Using", N_process, "processes")
    chunk_size = N // N_process
    CS = Communication_System(SNR_dB=10, n=7, k=4)
    
    # Create a list of chunks to process
    chunks = [(i * chunk_size, (i + 1) * chunk_size if i != N_process - 1 else N, CS) for i in range(N_process)]

    with Pool(processes=N_process) as pool:
        results = pool.starmap(process_chunk, chunks)

    
    # Combine results
    error_naive = sum(result[0] for result in results)
    error_hard = sum(result[1] for result in results)
    error_soft = sum(result[2] for result in results)
    
    print("Errors : Naive:", error_naive / N, "Hard:", error_hard / N, "Soft:", error_soft / N)

if __name__ == "__main__":
    print("Starting...")
    time_start = time()
    main()
    time_end = time()
    print("Execution time:", time_end - time_start)
