"""
Script to preprocess the heldout sequences and generate the training and validation sets. Files are saved in the data folder.
Use the fshap-codes environment
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from src.utils import split_sequence



"""
Create frameshift heldout dataset from the main text.
Create a CSV with the experimental scores, the full sequence, and the n=40 sequence.
"""
# Available at https://www.nature.com/articles/s41586-018-0686-x#Sec21 Source Data Fig. 3
df = pd.read_excel('data/41586_2018_686_MOESM5_ESM.xlsx', sheet_name='Fig. 3d')
df = df[df['Frame'].isin([1, 2])]
df = df.groupby('_Experiment').head(2)
df = df.groupby('_Experiment')['Obs'].sum()
df = df.reset_index(name='Frameshift frequency')
# For each experiment, add the sequences into the dataframe. 
df_names = pd.read_csv('data/names-libA.txt', header=None, names=['_Experiment'], on_bad_lines='skip')
df_sequences = pd.read_csv('data/targets-libA.txt', header=None, names=['Sequence'])
df_names_sequences = pd.concat([df_names, df_sequences], axis=1)
df = df.merge(df_names_sequences, on='_Experiment')

# Split the sequence into left and right halves. Only keep n nucleotides around the cutsite. 
n = 40
df[['Left_seq', 'Right_seq']] = df['Sequence'].apply(lambda x: pd.Series(split_sequence(x, num_keep=int(n/2))))
df_frameshift = df.copy()



"""
SUPPLEMENTARY DATA: Highest 1-bp insertion % for HEK293 cells
"""
df = pd.read_excel('data/41586_2018_686_MOESM5_ESM.xlsx', sheet_name='Fig. 3e')
df = df[['_Experiment', 'ins_1bp.1']].copy()
# Rename the 'ins_1bp.1' column to 'Highest 1-bp insertion'
df = df.rename(columns={'ins_1bp.1': 'Highest 1-bp insertion'})
df_names = pd.read_csv('data/names-libA.txt', header=None, names=['_Experiment'], on_bad_lines='skip')
df_sequences = pd.read_csv('data/targets-libA.txt', header=None, names=['Sequence'])
df_names_sequences = pd.concat([df_names, df_sequences], axis=1)
df = df.merge(df_names_sequences, on='_Experiment', suffixes=(None, '_new'))
df = df.dropna(subset=['Highest 1-bp insertion', '_Experiment'])

# Split the sequence into left and right halves. Only keep n nucleotides around the cutsite. 
df_1_bp_ins = df.copy()#.to_csv('data/HEK293_frameshift.csv')
df_1_bp_ins = df_1_bp_ins.filter(items=['_Experiment', 'Highest 1-bp insertion'])
df = pd.merge(df_frameshift, df_1_bp_ins, on='_Experiment', how='outer')
df = df.dropna(subset=['Frameshift frequency']).reset_index(drop=True)
df.to_csv('data/HEK293_heldout.csv')



"""
SUPPLEMENTARY DATA: Highest 1-bp insertion % for U2OS cells
"""






