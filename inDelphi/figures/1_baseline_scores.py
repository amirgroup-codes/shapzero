"""
Script to compute scores for SHAP Zero, linear model, and pairwise model
"""
import numpy as np
import sys
import pandas as pd
sys.path.append('../..')
from gen.utils import run_linear_model, run_pairwise_model, compute_fourier_output
import pickle
from scipy.stats import pearsonr, spearmanr
encoding = {0:'A', 1:'C', 2:'T', 3:'G'}



"""
HEK293 for frameshift and 1-bp-ins
"""
q = 4
n = 40
b = 7
celltype = 'HEK293'
properties = ['Frameshift frequency', 'Highest 1-bp insertion']
file_path = ['frameshift', '1_bp_ins']
df_heldout = pd.read_csv('data/{}_heldout.csv'.format(celltype))
left_seq = df_heldout['Left_seq']
right_seq = df_heldout['Right_seq']
seqs = left_seq + right_seq
seqs = seqs.tolist()



# SHAP Zero
df = pd.DataFrame()
print('SHAP Zero')
for f, property in zip(file_path, properties):
    
    # Load Fourier coefficients
    with open('../results/q{}_n{}_b{}/qsft_transform_{}_{}.pickle'.format(q, n, b,celltype, f), 'rb') as file:
        qsft_transform = pickle.load(file)
    mean_file = '../results/q{}_n{}_b{}/train/samples/{}/{}/train_mean.npy'.format(q, n, b, celltype, f)
    mean = np.load(mean_file)
    zeros = [0 for _ in range(n)]
    qsft_transform[tuple(zeros)] = qsft_transform.get(tuple(zeros), 0) + mean

    # Convert sequences back to q-ary encoding
    seqs_qary = []
    for seq in seqs:
        seq = list(seq)
        encoded_seq = [list(encoding.keys())[list(encoding.values()).index(char)] for char in seq]
        seqs_qary.append(encoded_seq)
    seqs_qary = np.array(seqs_qary)

    # Compute predicted output
    y_pred = np.real(compute_fourier_output(seqs_qary, qsft_transform, q))
    pearson_corr, _ = pearsonr(df_heldout[property], y_pred)
    spearman_corr, _ = spearmanr(df_heldout[property], y_pred)
    print('{} Pearson correlation: {:.2f}, Spearman correlation: {:.2f}'.format(property, pearson_corr, spearman_corr))
    df['SHAP Zero: {}'.format(property)] = y_pred



# Linear and pairwise models
print('Linear and pairwise models')
df_train = pd.read_csv('data/{}_train.csv'.format(celltype))
for property in properties:
    sequences_test = seqs.copy()
    sequences_train = df_train.apply(lambda row: row['Left_seq'] + row['Right_seq'], axis=1).tolist()

    # Linear model
    corr, y_pred = run_linear_model(sequences_train, df_train[property].tolist(), sequences_test, df_heldout[property].tolist())
    pearson_corr, _ = pearsonr(df_heldout[property], y_pred)
    spearman_corr, _ = spearmanr(df_heldout[property], y_pred)
    y_pred_linear = y_pred.copy()
    print('Linear model {}: Pearson correlation: {:.2f}, Spearman correlation: {:.2f}'.format(property, pearson_corr, spearman_corr))
    df['Linear: {}'.format(property)] = y_pred

    # Pairwise model
    corr, y_pred = run_pairwise_model(sequences_train, df_train[property].tolist(), sequences_test, df_heldout[property].tolist())
    pearson_corr, _ = pearsonr(df_heldout[property], y_pred)
    spearman_corr, _ = spearmanr(df_heldout[property], y_pred)
    y_pred_pairwise = y_pred.copy()
    print('Pairwise model {}: Pearson correlation: {:.2f}, Spearman correlation: {:.2f}'.format(property, pearson_corr, spearman_corr))
    df['Pairwise: {}'.format(property)] = y_pred

df.to_csv('correlation_results/model_scores_{}.csv'.format(celltype))