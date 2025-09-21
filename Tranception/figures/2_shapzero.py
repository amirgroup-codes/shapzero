"""
Script to compute SHAP values and Faith-Shap interactions 
Output is shapzero_values.npy, which contains a num_samples x n matrix of SHAP values
Output is shapzero_fsi.npz, which contains a num_samples x n matrix of q-ary encoded sequences under 'sequences' and a list of dictionaries under 'interactions_sequences' containing the interactions at each position for each sequence
"""
import numpy as np
import pandas as pd
import os 
import sys
sys.path.append('../..')
from gen.shapzero import shapzero
import pickle
from tqdm import tqdm
import time 



q = 20
n = 10
b = 3
properties = ['GFP_Tranception']
current_directory = os.getcwd()
for property in properties:

    data = np.load(f'data/{property}_heldout.npy')

    shapzero_values = np.zeros((data.shape[0], n))
    sequences = []
    interactions_sequences = []

    # Load Fourier coefficients
    with open('../results/q{}_n{}_b{}/qsft_transform_{}.pickle'.format(q, n, b, property), 'rb') as file:
        qsft_transform = pickle.load(file)
    mean_file = f'../results/q{q}_n{n}_b{b}/train/samples/{property}/train_mean.npy'
    mean = np.load(mean_file)
    zeros = [0 for _ in range(n)]
    qsft_transform[tuple(zeros)] = qsft_transform.get(tuple(zeros), 0) + mean
    # qsft_transform = {key: value for key, value in qsft_transform.items() if sum(1 for element in key if element != 0) <= 2}

    # Initialize shapzero class
    shap_zero = shapzero(qsft_transform, q=q, n=n)

    # Measure time to compute all SHAP values and interactions
    time_shap_value = 0
    time_interaction = 0
    for i, sequence in tqdm(enumerate(data), total=data.shape[0], desc="Processing sequences for {}".format(property)):

        """
        SHAP values
        """
        start_time_i = time.time()
        interactions = shap_zero.explain(sequence, explanation='shap_value') # Returns a dictionary containing the SHAP values for each position
        end_time_i = time.time()
        elapsed_time_i = end_time_i - start_time_i
        time_shap_value += elapsed_time_i

        # Save values for plotting
        for key, value in interactions.items():
            shapzero_values[i, key[0]] = np.real(value) 

        
        """
        Interactions
        """
        start_time_i = time.time()
        interactions = shap_zero.explain(sequence, explanation='faith_shap') # Returns a dictionary containing interactions for each position
        end_time_i = time.time()
        elapsed_time_i = end_time_i - start_time_i
        time_interaction += elapsed_time_i

        interactions_sequences.append(interactions)
        sequences.append(sequence)
    average_time_per_sample_shap_value = time_shap_value / data.shape[0]
    average_time_per_sample_interaction = time_interaction / data.shape[0]

    np.save(f'shap_results/time_shapzero_values_{property}.npy', average_time_per_sample_shap_value) 
    np.save(f'shap_results/time_shapzero_fsi_{property}.npy', average_time_per_sample_interaction)
    np.save(f'shap_results/shapzero_values_{property}.npy', shapzero_values)
    np.savez(f'shap_results/shapzero_fsi_{property}.npz', interactions_sequences=interactions_sequences, sequences=sequences)


    




