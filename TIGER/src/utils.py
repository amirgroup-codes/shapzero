import sys
sys.path.append('..')
sys.path.append('../tiger/hugging_face/')
import os
import numpy as np
import shap
import pandas as pd
from tqdm import tqdm
encoding = {0:'A', 1:'C', 2:'T', 3:'G'}

from tiger_class import TranscriptProcessor
UNIT_INTERVAL_MAP = 'sigmoid'


def find_target_sequence(indices_nucleotides, n, target_length=23, with_context=26):
    """
    Given a numpy array with a sequence encoding, find the target sequence, assuming the sequence must be 26 nt long
    """
    encoding = {0:'A', 1:'C', 2:'T', 3:'G'}
    indices_nucleotides = np.array(indices_nucleotides)
    if indices_nucleotides.ndim == 1:
        indices_nucleotides = indices_nucleotides[np.newaxis, :]
    indices_nucleotides = [[encoding[num] for num in row] for row in indices_nucleotides]
    indices_nucleotides = [list(map(str, row)) for row in indices_nucleotides]
    if n is not with_context:
         raise ValueError(f'Sequence is not {with_context} nt long')
    else:
        seqs_target = [
                ''.join(row[3:])
                for row in indices_nucleotides
            ]
        seqs = [''.join(row) for row in indices_nucleotides]
    return seqs, seqs_target


def batch_load(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def batch_prediction(seq_batch, processor=None):
    """
    Calls tiger_class.py to run TIGER
    Takes in a list of strings 
    """
    df = pd.DataFrame(seq_batch, columns=['transcript_seq'])
    processor.process_batches(df_seqs=df)
    df_on_target, df_titration, df_off_target = processor.get_results()

    return df_on_target, df_titration, df_off_target


def process_on_target(df_on_target):
    return df_on_target['Guide Score'].values.tolist()


def process_titration(df_titration):
    """
    Compute the mean of guide scores for each target sequence
    """
    return df_titration.groupby('Transcript ID')['Guide Score'].mean().tolist()


def get_predictions_shap_ontarget(X):
    """
    Takes in a matrix X (# samples x # features) and runs TIGER. Output is a numpy array length # samples
    """
    tiger_dir = '../tiger/hugging_face/'
    postprocess = True
    mode = 'all'
    processor = TranscriptProcessor(tiger_dir=tiger_dir, postprocess=postprocess, mode=mode, gpu_num=0)
    batch_size = 512
    indices_nucleotides = [[encoding[num] for num in row] for row in X]
    indices_nucleotides = [list(map(str, row)) for row in indices_nucleotides]
    indices_nucleotides = [''.join(row) for row in indices_nucleotides]
    all_samples = []
    """
    Process experimental samples
    """
    with tqdm(total=len(indices_nucleotides), desc="Processing experimental samples") as pbar:
        for seq_batch in batch_load(indices_nucleotides, batch_size):
            df_on_target, _, _ = batch_prediction(seq_batch, processor=processor)
            on_target_scores_batch = process_on_target(df_on_target)
            all_samples = all_samples + on_target_scores_batch
            pbar.update(len(seq_batch))

    all_samples = np.array(all_samples) 
    return all_samples


def get_predictions_shap_titration(X):
    """
    Takes in a matrix X (# samples x # features) and runs TIGER. Output is a numpy array length # samples
    """
    tiger_dir = '../tiger/hugging_face/'
    postprocess = True
    mode = 'titration'
    processor = TranscriptProcessor(tiger_dir=tiger_dir, postprocess=postprocess, mode=mode, gpu_num=0)
    batch_size = 512

    indices_nucleotides = [[encoding[num] for num in row] for row in X]
    indices_nucleotides = [list(map(str, row)) for row in indices_nucleotides]
    indices_nucleotides = [''.join(row) for row in indices_nucleotides]
    all_samples = []

    """
    Process experimental samples
    """
    with tqdm(total=len(indices_nucleotides), desc="Processing experimental samples") as pbar:
        for seq_batch in batch_load(indices_nucleotides, batch_size):
            _, df_titration, _ = batch_prediction(seq_batch, processor=processor)
            titration_scores_batch = process_titration(df_titration)
            all_samples = all_samples + titration_scores_batch
            pbar.update(len(seq_batch))

    all_samples = np.array(all_samples) 
    return all_samples
