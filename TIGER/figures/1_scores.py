"""
Script to compute scores for SHAP Zero, TIGER, linear model, and pairwise model
"""
import numpy as np
import sys
import pandas as pd
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../tiger')
sys.path.append('../tiger/hugging_face/')
from gen.utils import run_linear_model, run_pairwise_model, compute_fourier_output
import pickle
from tiger_trainer import score_tiger, return_scoring, measure_performance
from tiger_class import calibrate_predictions, score_predictions
from utils import measure_performance # From the tiger directory
encoding = {0:'A', 1:'C', 2:'T', 3:'G'}
sat_quant_active = 0.05
sat_quant_inactive = 0.95
from src.utils import batch_prediction, batch_load, process_titration, process_on_target
from tiger_class import TranscriptProcessor
import time
from tqdm import tqdm
batch_size = 512
postprocess = True
tiger_dir = '../tiger/hugging_face/'



"""
on_target and titration
"""
q = 4
n = 26
b = 7
properties = ['on_target', 'titration']
file_path = ['on_target', 'titration']

# SHAP Zero
df_on_target = pd.DataFrame()
df_titration = pd.DataFrame()
index_names = ['Pearson', 'Spearman', 'AUROC', 'AUPRC']
df_corrs = pd.DataFrame(index=index_names)
print('SHAP Zero')
for f, property in zip(file_path, properties):

    df_heldout = pd.read_csv('data/{}_heldout.csv'.format(property))
    seqs = df_heldout['Full Sequence'].tolist()
    
    # Load Fourier coefficients
    with open('../results/q{}_n{}_b{}/qsft_transform_{}.pickle'.format(q, n, b, f), 'rb') as file:
        qsft_transform = pickle.load(file)
    mean_file = '../results/q{}_n{}_b{}/train/samples/{}/train_mean.npy'.format(q, n, b, f)
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
    df_heldout['predicted_lfc'] = y_pred 
    scoring_params = score_tiger(df_heldout, sat_quant_active, sat_quant_inactive)
    df_heldout, scoring_df = return_scoring(df_heldout, scoring_params)
    y_pred = df_heldout['predicted_lfc'].to_list()
    print('{} - Pearson correlation: {:.2f}, Spearman correlation: {:.2f}, AUROC: {:.2f}, AUPRC: {:.2f}'.format(property, scoring_df['Pearson'][0], scoring_df['Spearman'][0], scoring_df['AUROC'][0], scoring_df['AUPRC'][0]))
    corr = scoring_df['Pearson'][0]
    corr_error = scoring_df['Pearson err'][0]
    spearman = scoring_df['Spearman'][0]
    spearman_error = scoring_df['Spearman err'][0]
    AUROC = scoring_df['AUROC'][0]
    AUROC_error = scoring_df['AUROC err'][0]
    AUPRC = scoring_df['AUPRC'][0]
    AUPRC_error = scoring_df['AUPRC err'][0]
    scores = [corr, spearman, AUROC, AUPRC]
    errors = [corr_error, spearman_error, AUROC_error, AUPRC_error]
    if property == 'on_target':
        df_on_target['SHAP Zero: {}'.format(property)] = y_pred
    else:
        df_titration['SHAP Zero: {}'.format(property)] = y_pred
    df_corrs['SHAP Zero: {}'.format(property)] = scores
    df_corrs['SHAP Zero error: {}'.format(property)] = errors



# TIGER
print('TIGER')
for property in properties:
    all_scores = []
    df_heldout = pd.read_csv('data/{}_heldout.csv'.format(property))
    seqs = df_heldout['Full Sequence'].tolist()

    if property == 'on_target':
        mode = 'all'
        processor = TranscriptProcessor(tiger_dir=tiger_dir, postprocess=postprocess, mode=mode)
    elif property == 'titration':
        mode = 'titration'
        processor = TranscriptProcessor(tiger_dir=tiger_dir, postprocess=postprocess, mode=mode)

    processor = TranscriptProcessor(tiger_dir=tiger_dir, postprocess=postprocess, mode=mode)
    start_time = time.time()   
    scores = []
    with tqdm(total=len(seqs), desc="Processing experimental samples") as pbar:
        for seq_batch in batch_load(seqs, batch_size):
            if property == 'on_target':
                df_batch, _, _ = batch_prediction(seq_batch, processor=processor)
                on_target_scores_batch = process_on_target(df_batch)
                all_scores = all_scores + on_target_scores_batch
            elif property == 'titration':
                _, df_batch, _ = batch_prediction(seq_batch, processor=processor)
                titration_scores_batch = process_titration(df_batch)
                all_scores = all_scores + titration_scores_batch
            pbar.update(len(seq_batch))

    end_time = time.time()  
    total_time = end_time - start_time
    average_time_per_sample = total_time / len(seqs)
    print("Total time:", total_time)
    print("Average time per sample:", average_time_per_sample)
    np.save('shap_results/time_per_sample_{}.npy'.format(property), average_time_per_sample) # Use this for time complexity of SHAP Zero

    # Compute predicted output
    y_pred = all_scores.copy()
    df_heldout['predicted_lfc'] = y_pred 
    scoring_params = score_tiger(df_heldout, sat_quant_active, sat_quant_inactive)
    df_heldout, scoring_df = return_scoring(df_heldout, scoring_params)
    y_pred = df_heldout['predicted_lfc'].to_list()
    print('{} - Pearson correlation: {:.2f}, Spearman correlation: {:.2f}, AUROC: {:.2f}, AUPRC: {:.2f}'.format(property, scoring_df['Pearson'][0], scoring_df['Spearman'][0], scoring_df['AUROC'][0], scoring_df['AUPRC'][0]))
    corr = scoring_df['Pearson'][0]
    corr_error = scoring_df['Pearson err'][0]
    spearman = scoring_df['Spearman'][0]
    spearman_error = scoring_df['Spearman err'][0]
    AUROC = scoring_df['AUROC'][0]
    AUROC_error = scoring_df['AUROC err'][0]
    AUPRC = scoring_df['AUPRC'][0]
    AUPRC_error = scoring_df['AUPRC err'][0]
    scores = [corr, spearman, AUROC, AUPRC]
    errors = [corr_error, spearman_error, AUROC_error, AUPRC_error]
    if property == 'on_target':
        df_on_target['Model: {}'.format(property)] = y_pred
    else:
        df_titration['Model: {}'.format(property)] = y_pred
    df_corrs['Model: {}'.format(property)] = scores
    df_corrs['Model: {}'.format(property)] = errors



# Linear and pairwise models (skip score_tiger and return_scoring since we're already predicting observed_lfc directly)
print('Linear and pairwise models')
for property in properties:

    df_train = pd.read_csv('data/{}_train.csv'.format(property))
    df_heldout = pd.read_csv('data/{}_heldout.csv'.format(property))
    sequences_test = df_heldout['Full Sequence'].tolist()
    sequences_train = df_train['Full Sequence'].tolist()

    # Linear model
    corr, y_pred = run_linear_model(sequences_train, df_train['observed_lfc'].tolist(), sequences_test, df_heldout['observed_lfc'].tolist())
    df_heldout['predicted_lfc'] = y_pred 
    scoring_df = measure_performance(df_heldout, silence=False)
    y_pred = df_heldout['predicted_lfc'].to_list()
    print('Linear {} - Pearson correlation: {:.2f}, Spearman correlation: {:.2f}, AUROC: {:.2f}, AUPRC: {:.2f}'.format(property, scoring_df['Pearson'][0], scoring_df['Spearman'][0], scoring_df['AUROC'][0], scoring_df['AUPRC'][0]))
    corr = scoring_df['Pearson'][0]
    corr_error = scoring_df['Pearson err'][0]
    spearman = scoring_df['Spearman'][0]
    spearman_error = scoring_df['Spearman err'][0]
    AUROC = scoring_df['AUROC'][0]
    AUROC_error = scoring_df['AUROC err'][0]
    AUPRC = scoring_df['AUPRC'][0]
    AUPRC_error = scoring_df['AUPRC err'][0]
    scores = [corr, spearman, AUROC, AUPRC]
    errors = [corr_error, spearman_error, AUROC_error, AUPRC_error]
    if property == 'on_target':
        df_on_target['Linear: {}'.format(property)] = y_pred
    else:
        df_titration['Linear: {}'.format(property)] = y_pred
    df_corrs['Linear: {}'.format(property)] = scores
    df_corrs['Linear error: {}'.format(property)] = errors

    # Pairwise model
    corr, y_pred = run_pairwise_model(sequences_train, df_train['observed_lfc'].tolist(), sequences_test, df_heldout['observed_lfc'].tolist())
    df_heldout['predicted_lfc'] = y_pred 
    scoring_df = measure_performance(df_heldout, silence=False)
    y_pred = df_heldout['predicted_lfc'].to_list()
    print('Pairwise {} - Pearson correlation: {:.2f}, Spearman correlation: {:.2f}, AUROC: {:.2f}, AUPRC: {:.2f}'.format(property, scoring_df['Pearson'][0], scoring_df['Spearman'][0], scoring_df['AUROC'][0], scoring_df['AUPRC'][0]))
    corr = scoring_df['Pearson'][0]
    corr_error = scoring_df['Pearson err'][0]
    spearman = scoring_df['Spearman'][0]
    spearman_error = scoring_df['Spearman err'][0]
    AUROC = scoring_df['AUROC'][0]
    AUROC_error = scoring_df['AUROC err'][0]
    AUPRC = scoring_df['AUPRC'][0]
    AUPRC_error = scoring_df['AUPRC err'][0]
    scores = [corr, spearman, AUROC, AUPRC]
    errors = [corr_error, spearman_error, AUROC_error, AUPRC_error]
    if property == 'on_target':
        df_on_target['Pairwise: {}'.format(property)] = y_pred
    else:
        df_titration['Pairwise: {}'.format(property)] = y_pred
    df_corrs['Pairwise: {}'.format(property)] = scores
    df_corrs['Pairwise error: {}'.format(property)] = errors

df_on_target.to_csv('correlation_results/model_scores_on_target.csv'.format(property))
df_titration.to_csv('correlation_results/model_scores_titration.csv'.format(property))
df_corrs.to_csv('correlation_results/model_results.csv')