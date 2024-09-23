import numpy as np
import random
from tqdm import tqdm
random.seed(42)
import os
import sys
indelphi_folder = os.path.dirname(os.path.abspath(__file__)) + '/../inDelphi-model'
sys.path.append(indelphi_folder)
import inDelphi
import pickle
import zlib

# Define q-ary encoding 
encoding = {0:'A', 1:'C', 2:'T', 3:'G'}

# Add nucleotide padding to fit inDelphi's minimum length requirement
nucleotide_padding = [random.randint(0, 3) for _ in range(40)]
nucleotide_padding = [encoding[num] for num in nucleotide_padding]
n = 40 # Set to 40 for the largest qsft setup
cut = int(len(nucleotide_padding) / 2)
n_2 = int(n/2)


def get_predictions_shap(X, celltype='HEK293'): 
    """
    Takes in a matrix X (# samples x # features) and runs inDelphi. Output is a matrix (# samples x properties)
    properties: properties = ['Frameshift frequency', 'Precision', 'Insertion %', 'Expected indel length', 'MH del frequency', "Highest 1-bp insertion"]
    """
    inDelphi.init_model(celltype = celltype)
    ind = range(X.shape[0])
    properties = ['Frameshift frequency', 'Precision', 'Insertion %', 'Expected indel length', 'MH del frequency', "Highest 1-bp insertion"]
    all_samples = np.zeros((np.shape(X)[0], len(properties)))

    indices_nucleotides = [[encoding[num] for num in row] for row in X]
    indices_nucleotides = [list(map(str, row)) for row in indices_nucleotides]
    left_seqs = [''.join(nucleotide_padding[:cut]) + ''.join(row[:n_2]) for row in indices_nucleotides]
    right_seqs = [''.join(row[n_2:]) + ''.join(nucleotide_padding[cut:]) for row in indices_nucleotides]

    with tqdm(total=len(left_seqs), desc="Computing samples") as pbar:
        for (i, left_seq, right_seq) in zip(ind, left_seqs, right_seqs):
            seq = left_seq + right_seq
            cutsite = len(left_seq)
            pred_df, stats = inDelphi.predict(seq, cutsite)
            stats['Insertion %'] = pred_df.loc[pred_df['Category'] == 'ins', 'Predicted frequency'].sum()
            stats['Highest 1-bp insertion'] = pred_df[(pred_df['Category'] == 'ins') & (pred_df['Length'] == 1)]['Predicted frequency'].max()
            for j, property in enumerate(properties):
                all_samples[i, j] = stats[property]
            pbar.update()
    return all_samples


def get_predictions_fsi_HEK293_frameshift(X, celltype='HEK293', property='Frameshift frequency'): 
    """
    Takes in a matrix X (# samples x # features) and runs inDelphi. Output is a list of # samples with respect to one property
    """
    inDelphi.init_model(celltype = celltype)
    ind = range(X.shape[0])
    all_samples = np.zeros(np.shape(X)[0])

    indices_nucleotides = [[encoding[num] for num in row] for row in X]
    indices_nucleotides = [list(map(str, row)) for row in indices_nucleotides]
    left_seqs = [''.join(nucleotide_padding[:cut]) + ''.join(row[:n_2]) for row in indices_nucleotides]
    right_seqs = [''.join(row[n_2:]) + ''.join(nucleotide_padding[cut:]) for row in indices_nucleotides]

    with tqdm(total=len(left_seqs), desc="Computing samples") as pbar:
        for (i, left_seq, right_seq) in zip(ind, left_seqs, right_seqs):
            seq = left_seq + right_seq
            cutsite = len(left_seq)
            pred_df, stats = inDelphi.predict(seq, cutsite)
            stats['Insertion %'] = pred_df.loc[pred_df['Category'] == 'ins', 'Predicted frequency'].sum()
            stats["Highest 1-bp insertion"] = pred_df[(pred_df['Category'] == 'ins') & (pred_df['Length'] == 1)]['Predicted frequency'].max()
            all_samples[i] = stats[property]
            pbar.update()
    return all_samples


def get_predictions_fsi_HEK293_1bpins(X, celltype='HEK293', property='Highest 1-bp insertion'): 
    """
    Takes in a matrix X (# samples x # features) and runs inDelphi. Output is a list of # samples with respect to one property
    """
    inDelphi.init_model(celltype = celltype)
    ind = range(X.shape[0])
    all_samples = np.zeros(np.shape(X)[0])

    indices_nucleotides = [[encoding[num] for num in row] for row in X]
    indices_nucleotides = [list(map(str, row)) for row in indices_nucleotides]
    left_seqs = [''.join(nucleotide_padding[:cut]) + ''.join(row[:n_2]) for row in indices_nucleotides]
    right_seqs = [''.join(row[n_2:]) + ''.join(nucleotide_padding[cut:]) for row in indices_nucleotides]

    with tqdm(total=len(left_seqs), desc="Computing samples") as pbar:
        for (i, left_seq, right_seq) in zip(ind, left_seqs, right_seqs):
            seq = left_seq + right_seq
            cutsite = len(left_seq)
            pred_df, stats = inDelphi.predict(seq, cutsite)
            stats['Insertion %'] = pred_df.loc[pred_df['Category'] == 'ins', 'Predicted frequency'].sum()
            stats["Highest 1-bp insertion"] = pred_df[(pred_df['Category'] == 'ins') & (pred_df['Length'] == 1)]['Predicted frequency'].max()
            all_samples[i] = stats[property]
            pbar.update()
    return all_samples


def get_predictions_fsi_U2OS_1bpins(X, celltype='U2OS', property='Highest 1-bp insertion'): 
    """
    Takes in a matrix X (# samples x # features) and runs inDelphi. Output is a list of # samples with respect to one property
    """
    inDelphi.init_model(celltype = celltype)
    ind = range(X.shape[0])
    all_samples = np.zeros(np.shape(X)[0])

    indices_nucleotides = [[encoding[num] for num in row] for row in X]
    indices_nucleotides = [list(map(str, row)) for row in indices_nucleotides]
    left_seqs = [''.join(nucleotide_padding[:cut]) + ''.join(row[:n_2]) for row in indices_nucleotides]
    right_seqs = [''.join(row[n_2:]) + ''.join(nucleotide_padding[cut:]) for row in indices_nucleotides]

    with tqdm(total=len(left_seqs), desc="Computing samples") as pbar:
        for (i, left_seq, right_seq) in zip(ind, left_seqs, right_seqs):
            seq = left_seq + right_seq
            cutsite = len(left_seq)
            pred_df, stats = inDelphi.predict(seq, cutsite)
            stats['Insertion %'] = pred_df.loc[pred_df['Category'] == 'ins', 'Predicted frequency'].sum()
            stats["Highest 1-bp insertion"] = pred_df[(pred_df['Category'] == 'ins') & (pred_df['Length'] == 1)]['Predicted frequency'].max()
            all_samples[i] = stats[property]
            pbar.update()
    return all_samples


def df_str_to_encoding(df):
    """
    Takes in a dataframe and returns a matrix of nucleotide encodings without padding
    """
    seq_list = [] 
    for _, row in df.iterrows():

        left_seq = row['Left_seq']
        right_seq = row['Right_seq']
        seq = left_seq + right_seq
        encoded_seq = [list(encoding.keys())[list(encoding.values()).index(char)] for char in seq]
        seq_list.append(encoded_seq)
    
    seq_array = np.array(seq_list)
    return seq_array
    

def get_indelphi_from_numpy_arr(cell_type, all_query_indices, properties):
    
    inDelphi.init_model(celltype = cell_type)
    
    # Preprocess input
    if isinstance(all_query_indices, list):
        all_query_indices = np.array(all_query_indices)
    if all_query_indices.ndim == 1:
        all_query_indices = all_query_indices.reshape(1, -1)
    indices_nucleotides = [[encoding[num] for num in row] for row in all_query_indices]
    indices_nucleotides = [list(map(str, row)) for row in indices_nucleotides]
    left_seqs = [''.join(nucleotide_padding[:cut]) + ''.join(row[:n_2]) for row in indices_nucleotides]
    right_seqs = [''.join(row[n_2:]) + ''.join(nucleotide_padding[cut:]) for row in indices_nucleotides]
    
    num_samples, _ = np.shape(all_query_indices)
    all_samples = np.zeros((num_samples, len(properties)))
    
    # Compute samples
    with tqdm(total=len(left_seqs), desc='Processing Samples') as pbar:
        for i, (left_seq, right_seq) in enumerate(zip(left_seqs, right_seqs)):
            seq = left_seq + right_seq
            cutsite = len(left_seq)
            pred_df, stats = inDelphi.predict(seq, cutsite)
            
            stats['Insertion %'] = pred_df.loc[pred_df['Category'] == 'ins', 'Predicted frequency'].sum()
            stats['Highest 1-bp insertion'] = pred_df[(pred_df['Category'] == 'ins') & (pred_df['Length'] == 1)]['Predicted frequency'].max()
            
            for idx, property in enumerate(properties):
                all_samples[i, idx] = stats[property] 
            pbar.update()
            
    return all_samples


def get_indelphi_from_str(cell_type, all_query_indices_str, properties):
    seqs_encoding = [[list(encoding.keys())[list(encoding.values()).index(char)] for char in seq] for seq in all_query_indices_str]
    all_samples = get_indelphi_from_numpy_arr(cell_type, seqs_encoding, properties)
    return all_samples


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(zlib.decompress(f.read()))
    return data


def save_data(data, filename):
    with open(filename, 'wb') as f:
        f.write(zlib.compress(pickle.dumps(data, protocol=4), 9))


def split_sequence(sequence, num_keep = 10):
    # Calculate the midpoint and split into left and right seqs
    midpoint = len(sequence) // 2
    first_half = sequence[:midpoint]
    second_half = sequence[midpoint:]

    # Take nucleotides around the cutsite
    first_half = first_half[-num_keep:]
    second_half = second_half[:num_keep]

    return first_half, second_half