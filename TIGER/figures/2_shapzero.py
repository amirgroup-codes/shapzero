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



q = 4
n = 26
b = 7
properties = ['on_target', 'titration']
current_directory = os.getcwd()
encoding = {0:'A', 1:'C', 2:'T', 3:'G'}
for property in properties:

    df = pd.read_csv('data/{}_heldout.csv'.format(property))

    shapzero_values = np.zeros((len(df), n))
    sequences = []
    interactions_sequences = []

    # Load Fourier coefficients
    with open('../results/q{}_n{}_b{}/qsft_transform_{}.pickle'.format(q, n, b, property), 'rb') as file:
        qsft_transform = pickle.load(file)
    mean_file = f'../results/q{q}_n{n}_b{b}/train/samples/{property}/train_mean.npy'
    mean = np.load(mean_file)
    zeros = [0 for _ in range(n)]
    qsft_transform[tuple(zeros)] = qsft_transform.get(tuple(zeros), 0) + mean

    # Initialize shapzero class
    shap_zero = shapzero(qsft_transform, q=q, n=n)

    # Measure time to compute all SHAP values and interactions
    time_shap_value = 0
    time_interaction = 0
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing sequences for {}".format(property)):

        """
        SHAP values
        """
        sequence = list(row['Full Sequence'])
        sequence = [list(encoding.keys())[list(encoding.values()).index(char)] for char in sequence] # Convert sequence back to q-ary encoding
        localized_mobius_tf, _ = shap_zero.localize_sample(sequence)
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
    average_time_per_sample_shap_value = time_shap_value / df.shape[0]
    average_time_per_sample_interaction = time_interaction / df.shape[0]

    np.save(f'shap_results/time_shapzero_values_{property}.npy', average_time_per_sample_shap_value) 
    np.save(f'shap_results/time_shapzero_fsi_{property}.npy', average_time_per_sample_interaction)
    np.save(f'shap_results/shapzero_values_{property}.npy', shapzero_values)
    np.savez(f'shap_results/shapzero_fsi_{property}.npz', interactions_sequences=interactions_sequences, sequences=sequences)


    




