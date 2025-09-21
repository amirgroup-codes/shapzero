"""
Script to compute scores for SHAP Zero, linear model, and pairwise model
"""
import numpy as np
import sys
import pandas as pd
sys.path.append('../..')
from gen.utils import run_linear_model_protein, run_pairwise_model_protein, compute_fourier_output
import pickle
from scipy.stats import pearsonr, spearmanr
import os
aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
inv_aa_dict = {v: k for k, v in aa_dict.items()} 
sys.path.append('..')
from src.utils import ModelScorer
import torch

# Set training data to be from DMS
DMS='GFP_AEQVI_Sarkisyan_2016' 
DMS_list = pd.read_csv('../data/ProteinGym_reference_file_substitutions.csv')
DMS_list = DMS_list[DMS_list['DMS_id'] == DMS]
sequence = DMS_list["target_seq"].values[0].upper()
positions = [83, 87, 82, 84, 43, 164, 47, 162, 140, 165] # 0-indexed
scorer = ModelScorer(sequence, positions, DMS, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), property=property)
properties = ['GFP_Tranception']
file_path = ['GFP_Tranception']

# Select training data to come from heldout dataset. Test set is from DMS
for property in properties:
    all_scores = []
    data_train = np.load(f'data/{property}_heldout.npy')
    scores = scorer.compute_scores_from_numpy_arr(data_train)
    seqs_train = [''.join([inv_aa_dict[aa_idx] for aa_idx in seq]) for seq in data_train]

    # Turn into df with columns 'Sequence' and 'Score'
    df = pd.DataFrame(columns=['Sequence', f'{property}'])
    for i in range(len(seqs_train)):
        df.loc[i] = [seqs_train[i], scores[i]]
    df.to_csv('data/{}_train_baselines.csv'.format(property))

for property in properties:
    data_test = np.load(f'data/{property}_train.npy')
    data_scores = np.load(f'data/{property}_train_scores.npy')
    seqs_test = [''.join([inv_aa_dict[aa_idx] for aa_idx in seq]) for seq in data_test]

    # Turn into df with columns 'Sequence' and 'Score'
    df = pd.DataFrame(columns=['Sequence', f'{property}'])
    for i in range(len(seqs_test)):
        df.loc[i] = [seqs_test[i], data_scores[i]]

    # Score with Tranception too
    scores = scorer.compute_scores_from_numpy_arr(data_test)
    df['Tranception'] = scores
    df.to_csv('data/{}_test_baselines.csv'.format(property))



"""
Baselines
"""
q = 20
n = 10
b = 3
df = pd.DataFrame()
index_names = ['Pearson', 'Spearman']
df_corrs = pd.DataFrame(index=index_names)
for f, property in zip(file_path, properties):

    df_train = pd.read_csv('data/{}_train_baselines.csv'.format(property))
    seqs_train = df_train['Sequence'].tolist()
    df_heldout = pd.read_csv('data/{}_test_baselines.csv'.format(property))
    seqs_heldout = df_heldout['Sequence'].tolist()

    # Tranception
    df['Tranception: {}'.format(property)] = df_heldout['Tranception']
    # Get correlation
    pearson_corr, _ = pearsonr(df_heldout[property], df_heldout['Tranception'])
    spearman_corr, _ = spearmanr(df_heldout[property], df_heldout['Tranception'])
    print('{} - Pearson correlation: {:.2f}, Spearman correlation: {:.2f}'.format(property, pearson_corr, spearman_corr))
    df_corrs['Tranception: {}'.format(property)] = [pearson_corr, spearman_corr]

    
    print('SHAP Zero')
    # Load Fourier coefficients
    with open('../results/q{}_n{}_b{}/qsft_transform_{}.pickle'.format(q, n, b, f), 'rb') as file:
        qsft_transform = pickle.load(file)
    mean_file = '../results/q{}_n{}_b{}/train/samples/{}/train_mean.npy'.format(q, n, b, f)
    mean = np.load(mean_file)
    zeros = [0 for _ in range(n)]
    qsft_transform[tuple(zeros)] = qsft_transform.get(tuple(zeros), 0) + mean

    # Convert sequences back to q-ary encoding
    seqs_qary = []
    for seq in seqs_heldout:
        seq = list(seq)
        encoded_seq = [list(inv_aa_dict.keys())[list(inv_aa_dict.values()).index(char)] for char in seq]
        seqs_qary.append(encoded_seq)
    seqs_qary = np.array(seqs_qary)

    # Compute predicted output
    y_pred = np.real(compute_fourier_output(seqs_qary, qsft_transform, q))
    pearson_corr, _ = pearsonr(df_heldout[property], y_pred)
    spearman_corr, _ = spearmanr(df_heldout[property], y_pred)
    print('{} - Pearson correlation: {:.2f}, Spearman correlation: {:.2f}'.format(property, pearson_corr, spearman_corr))
    df['SHAP Zero: {}'.format(property)] = y_pred
    df_corrs['SHAP Zero: {}'.format(property)] = [pearson_corr, spearman_corr]



    print('Linear and pairwise models')    
    # Linear model
    corr, y_pred = run_linear_model_protein(seqs_train, df_train[property].tolist(), seqs_heldout, df_heldout[property].tolist())
    pearson_corr, _ = pearsonr(df_heldout[property], y_pred)
    spearman_corr, _ = spearmanr(df_heldout[property], y_pred)
    y_pred_linear = y_pred.copy()
    print('Linear model {}: Pearson correlation: {:.2f}, Spearman correlation: {:.2f}'.format(property, pearson_corr, spearman_corr))
    df['Linear: {}'.format(property)] = y_pred
    df_corrs['Linear: {}'.format(property)] = [pearson_corr, spearman_corr]

    # Pairwise model
    corr, y_pred = run_pairwise_model_protein(seqs_train, df_train[property].tolist(), seqs_heldout, df_heldout[property].tolist())
    pearson_corr, _ = pearsonr(df_heldout[property], y_pred)
    spearman_corr, _ = spearmanr(df_heldout[property], y_pred)
    y_pred_pairwise = y_pred.copy()
    print('Pairwise model {}: Pearson correlation: {:.2f}, Spearman correlation: {:.2f}'.format(property, pearson_corr, spearman_corr))
    df['Pairwise: {}'.format(property)] = y_pred
    df_corrs['Pairwise: {}'.format(property)] = [pearson_corr, spearman_corr]

    df.to_csv('correlation_results/model_scores.csv')
    df_corrs.to_csv('correlation_results/model_results.csv')
