"""
Create training datasets. Since we don't have the experimental values of the training set, we will use inDelphi predictions for the training set.
Use the inDelphi environment
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from src.utils import split_sequence, get_indelphi_from_str



n = 40
celltype = 'HEK293'
properties = ['Frameshift frequency', 'Precision', 'Insertion %', 'Expected indel length', 'MH del frequency', "Highest 1-bp insertion"]
file_path = ['frameshift', 'precision', 'insertion', 'indel_length', 'MH_del_freq', '1_bp_ins']
df_names = pd.read_csv('data/names-libA.txt', header=None, names=['_Experiment'])
df_sequences = pd.read_csv('data/targets-libA.txt', header=None, names=['Sequence'])
df_train = pd.concat([df_names, df_sequences], axis=1).head(1872) # All the training sequences
df_train[['Left_seq', 'Right_seq']] = df_train['Sequence'].apply(lambda x: pd.Series(split_sequence(x, num_keep=int(n/2))))
left_seq = df_train['Left_seq']
right_seq = df_train['Right_seq']
seqs = left_seq + right_seq
seqs = seqs.tolist()
all_samples = get_indelphi_from_str(celltype, seqs, properties)
for i, file_name in enumerate(file_path):
    df_train[file_name] = all_samples[:, i]
df_train.to_csv('data/{}_train.csv'.format(celltype))