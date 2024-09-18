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
df = df.reset_index(name='Observed Frameshift Frequency')

# For each experiment, add the sequences into the dataframe. 
df_names = pd.read_csv('data/names-libA.txt', header=None, names=['_Experiment'], on_bad_lines='skip')
df_sequences = pd.read_csv('data/targets-libA.txt', header=None, names=['Sequence'])
df_names_sequences = pd.concat([df_names, df_sequences], axis=1)
df = df.merge(df_names_sequences, on='_Experiment')

# Split the sequence into left and right halves. Only keep n nucleotides around the cutsite. 
n = 40
df[['Left_seq', 'Right_seq']] = df['Sequence'].apply(lambda x: pd.Series(split_sequence(x, num_keep=int(n/2))))
df.to_csv('data/HEK293_frameshift.csv')



"""
SUPPLEMENTARY DATA: Insertion % 
"""




