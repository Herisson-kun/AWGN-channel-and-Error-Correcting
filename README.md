# Communication System Simulation (CS Simulation)

## Overview

This project simulates a communication system using different decoding techniques to evaluate their performance under noisy conditions. It employs a parallel processing approach to handle a large number of simulations efficiently.

## Features

- **Simulate AWGN Channel**: Generates random messages and simulates the effects of an Additive White Gaussian Noise (AWGN) channel.
- **Decoding Techniques**: Implements various decoding techniques:
  - **Naive Decoding** (simple threshold)
  - **Hard Decoding** (Geometrical distance decision-process)
  - **Soft Decoding** (Probability decision-process)
- **Parallel Processing**: Utilizes Python's `multiprocessing` module to speed up computations by processing chunks of data in parallel.
- **Error Rate Calculation**: Computes the error rates for each decoding technique.

## Requirements

- Python 3.x
- `numpy`: For numerical operations
- `multiprocessing`: For parallel processing (included in Python standard library)
- `matplotlib` : For plots

## Usage

Run `stats_multiprocess.py` to get the error rates of the different methods. You can change the SNR by changing the `SNR_dB` argument of the CS object in the main function.
Otherwise I encourage you to check out `HARDvsSOFT.py` for a deeper understanding and more details about the Hard and the Soft decoding methods.
