import os
import argparse
import json
import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast
import tranception
from tranception import config, model_pytorch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



"""
General functions
"""
def compute_fourier_output(seqs_qary, qsft_transform, q):
    """
    Given a list of q-ary encoded sequence and the qsft transform, compute the predicted Fourier output.
    Parameters:
        seqs_qary (np.ndarray): Array of q-ary encoded sequences
        qsft_transform (dict): Dictionary mapping frequency vectors to Fourier coefficients
        q (int): Size of alphabet (e.g. 4 for DNA/RNA sequences)
        
    Returns:
        y_hat (np.ndarray): Predicted outputs for each sequence using the Fourier transform
    """
    batch_size = 10000
    beta_keys = list(qsft_transform.keys())
    beta_values = list(qsft_transform.values())
    y_hat = []
    for i in range(0, len(seqs_qary), batch_size):
        seqs_qary_batch = seqs_qary[i:i + batch_size, :]
        freqs = np.array(seqs_qary_batch) @ np.array(beta_keys).T
        H = np.exp(2j * np.pi * freqs / q)
        y_hat.append(H @ np.array(beta_values))

    y_hat = np.concatenate(y_hat)

    return y_hat


def filter_by_positions(df, positions):
    """
    Given DMS data and a list of positions (0-indexed), filter the DMS to only include rows where the mutant column contains a mutation at one of the specified positions.
    """
    positions_copy = positions.copy()
    for i in range(len(positions_copy)):
        positions_copy[i] += 1 # 1-indexed
    return df[df['mutant'].apply(lambda x: any(int(m[1:-1]) in positions_copy for m in x.split(':')))]


"""
Tranception functions
"""
def load_tranception(model_dir_path, data_dir_path, inference_time_retrieval=True, model='Tranception_Medium', DMS='GFP_AEQVI_Sarkisyan_2016', device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Load Tranception model with inference time retrieval
    """
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=model_dir_path + '../tranception/utils/tokenizers/Basic_tokenizer',
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]"
        )
    
    DMS_list = pd.read_csv(data_dir_path + 'ProteinGym_reference_file_substitutions.csv')
    DMS_list = DMS_list[DMS_list['DMS_id'] == DMS]
    target_seq = DMS_list["target_seq"].values[0].upper()
    if inference_time_retrieval:
        MSA_data_file = data_dir_path + 'MSA_files/' + DMS_list["MSA_filename"].values[0]
        MSA_weight_file_name = data_dir_path + 'MSA_weights/' + DMS_list["weight_file_name"].values[0]
        MSA_start = int(DMS_list["MSA_start"]) - 1 # MSA_start typically based on 1-indexing
        MSA_end = int(DMS_list["MSA_end"].values[0])

    config = json.load(open(model_dir_path + model + '/config.json'))
    config = tranception.config.TranceptionConfig(**config)
    config.attention_mode="tranception"
    config.position_embedding="grouped_alibi"
    config.tokenizer = tokenizer
    config.scoring_window = 'optimal'
    retrieval_inference_weight = 0.6

    if inference_time_retrieval:
        config.retrieval_aggregation_mode = "aggregate_substitution"
        config.MSA_filename=MSA_data_file
        config.full_protein_length=len(target_seq)
        config.MSA_weight_file_name=MSA_weight_file_name
        config.retrieval_inference_weight=retrieval_inference_weight
        config.MSA_start = MSA_start
        config.MSA_end = MSA_end
    else:
        config.retrieval_aggregation_mode = None

    model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path=model_dir_path + model, config=config)
    if torch.cuda.is_available():
        model.to(device)
        # model.cuda()
    model.eval()

    return model


def apply_mutations(wt, seq, pos):
    wt_list = list(wt) 
    for seq_idx, wt_idx in enumerate(pos):
        wt_list[wt_idx] = seq[seq_idx] 
    return "".join(wt_list)


def log_mutations(wt, seq):
    """
    Records mutations in WT:position:mutated form
    """
    mutations = []
    for i, (wt_char, mut_char) in enumerate(zip(wt, seq), 1):  # 1-indexed
        if wt_char != mut_char:
            mutations.append(f"{wt_char}{i}{mut_char}")
    return ":".join(mutations) if mutations else "WT"


aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
num_encoding = {v: k for k, v in aa_dict.items()}
def compute_scores_from_numpy_arr(samples, model, positions=None, target_seq=None, batch_size=64):
    """
    Scores sequences using Tranception
    """
    # Format mutations
    indices_nucleotides = [[num_encoding[num] for num in row] for row in samples]
    indices_nucleotides = [list(map(str, row)) for row in indices_nucleotides]
    seqs = [''.join(row) for row in indices_nucleotides]
    df = pd.DataFrame({'original_sequence': seqs})
    df['mutated_sequence'] = df['original_sequence'].apply(lambda seq: apply_mutations(target_seq, seq, positions))
    df['mutant'] = df['mutated_sequence'].apply(lambda seq: log_mutations(target_seq, seq))
    df.drop(columns=['original_sequence'], inplace=True)

    # Check if WT row exists. If it does, remove it, then re-add it back after Tranception scores
    if 'WT' in df['mutant'].values:
        wt_mask = df['mutant'] == 'WT'
        df_no_wt = df[~wt_mask]

        # Score unique mutants only
        df_unique = df_no_wt.drop_duplicates(subset='mutant')
        num_workers = 10
        scored_mutants = model.score_mutants(
            DMS_data=df_unique, 
            target_seq=target_seq, 
            scoring_mirror=False, 
            batch_size_inference=batch_size,  
            num_workers=num_workers, 
            indel_mode=False
        )

        # Add back in the wildtype with the score of 0
        # Ensure the scored DataFrame has a 'mutant' column to merge on
        if 'mutant' not in scored_mutants.columns:
            scored_mutants = df_unique[['mutant']].reset_index(drop=True).join(scored_mutants)
        df = df.copy()
        df = df.merge(scored_mutants[['mutant', 'avg_score']], on='mutant', how='left')
        df['avg_score'] = df['avg_score'].fillna(0.0)  # fill WT or unmatched with 0.0
        all_scores = df

    else:
        # Score sequences
        num_workers = 10
        all_scores = model.score_mutants(
            DMS_data=df, 
            target_seq=target_seq, 
            scoring_mirror=False, 
            batch_size_inference=batch_size,  
            num_workers=num_workers, 
            indel_mode=False
            )

    # Tranception automatically removes duplicates. If we have duplicates, we want to keep them in the dataframe.
    if len(df) != len(all_scores):
        all_scores = df.merge(all_scores, on=["mutated_sequence"], how="left")

    scores = all_scores['avg_score'].tolist()

    return scores


import sys
sys.path.append('../..')
from gen.utils import load_data
class ModelScorer:

    def __init__(self, target_seq, positions, DMS, device='cpu', property=None):
        self.model = load_tranception(model_dir_path='../model/', data_dir_path='../data/', inference_time_retrieval=True, model='Tranception_Large', DMS=DMS, device=device)
        self.target_seq = target_seq
        self.positions = positions
        self.M = 3
        self.D = 3
        self.property = property



    def compute_scores_from_numpy_arr(self, samples):
        return np.array(compute_scores_from_numpy_arr(samples=samples, model=self.model, positions=self.positions, target_seq=self.target_seq))

    def load_training_scores(self, samples):
        all_scores = self.compute_scores_from_numpy_arr(samples)
        all_samples = list(map(tuple, samples))
        self.scores = dict(zip(all_samples, all_scores))

    def compute_scores_fastshap(self, X):

        if type(X) == torch.Tensor:
            X = X.cpu().numpy()

        X = list(map(tuple, X))
        scores = np.array([self.scores[sample] for sample in X])
        return scores

