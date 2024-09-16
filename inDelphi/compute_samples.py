"""
Model-specific script to compute samples given the q-ary indices from get_qary_indices.py
For inDelphi: computes the following summary statistics in one shot: 'frameshift', 'precision', 'insertion', 'indel_length', 'MH_del_freq', '1_bp_ins'
"""
import pickle
import numpy as np
from tqdm import tqdm
import random
import os
random.seed(42)
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.utils import get_indelphi_from_numpy_arr
import argparse
import zlib



def parse_args():
    parser = argparse.ArgumentParser(description="Compute samples.")
    parser.add_argument("--q", required=True)
    parser.add_argument("--n", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--num_subsample", required=True)
    parser.add_argument("--num_repeat", required=True)
    parser.add_argument("--celltype", required=True)
    return parser.parse_args()
def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(zlib.decompress(f.read()))
    return data
def save_data(data, filename):
    with open(filename, 'wb') as f:
        f.write(zlib.compress(pickle.dumps(data, protocol=4), 9))
properties = ['Frameshift frequency', 'Precision', 'Insertion %', 'Expected indel length', 'MH del frequency', 'Highest 1-bp insertion']
file_path = ['frameshift', 'precision', 'insertion', 'indel_length', 'MH_del_freq', '1_bp_ins']

if __name__ == "__main__":
    args = parse_args()


    """
    Set parameters
    """
    # q = Alphabet size
    # n = Length of sequence
    # b = inner dimension of subsampling
    q = int(args.q)
    n = int(args.n)
    N = q ** n
    b = int(args.b)
    len_seq = n

    # num_sample (M) and num_repeat (D) control the amount of samples computed 
    M = int(args.num_subsample)
    D = int(args.num_repeat)

    # inDelphi-specific celltype
    celltype = args.celltype

    current_directory = os.path.join(os.getcwd(), '..', 'results') # Go up one directory into results

    # Define q-ary encoding 
    encoding = {0:'A', 1:'C', 2:'T', 3:'G'}

    # Add nucleotide padding to fit inDelphi's minimum length requirement
    nucleotide_padding = [random.randint(0, 3) for _ in range(40)]
    nucleotide_padding = [encoding[num] for num in nucleotide_padding]
    cut = int(len(nucleotide_padding) / 2)
    n_2 = int(n/2)



    """
    Initialize files
    """
    folder_path = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        os.makedirs(os.path.join(folder_path, "train"))
        os.makedirs(os.path.join(folder_path, "train", "samples"))
        os.makedirs(os.path.join(folder_path, "test"))
    celltype_folder_path = os.path.join(folder_path, "train", "samples", "{}".format(celltype))
    if not os.path.exists(celltype_folder_path):
        os.makedirs(celltype_folder_path)
        for file in file_path:
            if not os.path.exists(os.path.join(celltype_folder_path, file)):
                os.makedirs(os.path.join(celltype_folder_path, file))
    celltype_folder_path_mean = os.path.join(folder_path, "train", "samples_mean", "{}".format(celltype))
    if not os.path.exists(celltype_folder_path_mean):
        os.makedirs(celltype_folder_path_mean)
        for file in file_path:
            if not os.path.exists(os.path.join(celltype_folder_path_mean, file)):
                os.makedirs(os.path.join(celltype_folder_path_mean, file))
    celltype_folder_path_test = os.path.join(folder_path, "test", "{}".format(celltype))
    for file in file_path:
        if not os.path.exists(os.path.join(celltype_folder_path_test, file)):
            os.makedirs(os.path.join(celltype_folder_path_test, file))



    """
    Compute samples needed for F-SHAP
    """
    for i in range(M):
        for j in range(D):
            query_indices_file = os.path.join(folder_path, "train", "samples", "M{}_D{}_queryindices.pickle".format(i, j))
            query_indices = load_data(query_indices_file)
            flag = True

            # Loop through all files and check if they exist
            for file in file_path:
                sample_file = os.path.join(celltype_folder_path, file, "M{}_D{}.pickle".format(i, j))
                sample_file_mean = os.path.join(celltype_folder_path_mean, file, "M{}_D{}.pickle".format(i, j))
                if os.path.isfile(sample_file):
                    flag = False
                    
            if flag: 
                block_length = len(query_indices[0])
                samples = [np.zeros((len(query_indices), block_length), dtype=complex) for _ in range(len(file_path))]

                all_query_indices = np.concatenate(query_indices)
                all_samples = get_indelphi_from_numpy_arr(celltype, all_query_indices, properties)         

                for file, sample, arr in zip(file_path, samples, all_samples.T):
                    sample_file = os.path.join(celltype_folder_path, file, "M{}_D{}.pickle".format(i, j))
                    sample_file_mean = os.path.join(celltype_folder_path_mean, file, "M{}_D{}.pickle".format(i, j))
                    for k in range(len(query_indices)):
                        sample[k] = arr[k * block_length: (k+1) * block_length]
                    sample = sample.T
                    save_data(sample, sample_file)
                    save_data(sample, sample_file_mean)
                        
                    

    # Save the empirical mean separately
    folder_path = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b))
    for file in file_path:
        mean_file = os.path.join(celltype_folder_path, file, "train_mean.npy") 

        if not os.path.isfile(mean_file):
            all_samples = []
            for i in range(M):
                for j in range(D):
                    sample_file = os.path.join(celltype_folder_path_mean, file, "M{}_D{}.pickle".format(i, j))
                    samples = load_data(sample_file)
                    samples = np.concatenate(samples)
                    all_samples = np.concatenate([all_samples, samples])
            all_samples_mean = np.mean(all_samples)
            np.save(mean_file, all_samples_mean)
        else:
            all_samples_mean = np.load(mean_file)
        for i in range(M):
            for j in range(D):
                sample_file_zeromean = os.path.join(celltype_folder_path, file, "M{}_D{}.pickle".format(i, j))
                sample_file = os.path.join(celltype_folder_path_mean, file, "M{}_D{}.pickle".format(i, j))
                samples = load_data(sample_file)
                samples_zeromean = samples - all_samples_mean
                save_data(samples_zeromean, sample_file_zeromean)



    """
    Testing samples to compute NMSE and R^2
    """
    query_indices_file = os.path.join(celltype_folder_path_test, "..", "signal_t_queryindices.pickle")
    query_indices = load_data(query_indices_file)

    query_qaryindices_file = os.path.join(celltype_folder_path_test, "..", "signal_t_query_qaryindices.pickle")
    query_qaryindices = load_data(query_qaryindices_file)

    # Loop through all files and check if they exist
    for file in file_path:
        sample_file = os.path.join(celltype_folder_path_test, file, "signal_t.pickle")
        sample_file_mean = os.path.join(celltype_folder_path_test, file, "signal_t_mean.pickle")
        flag = True

        if os.path.isfile(sample_file):
            flag = False

    if flag:
        block_length = len(query_indices[0])
        samples = np.zeros((len(query_indices), block_length), dtype=complex)

        all_query_indices = query_indices
        all_samples = get_indelphi_from_numpy_arr(celltype, all_query_indices, properties)

        for file, arr in zip(file_path, all_samples.T):
            sample_file = os.path.join(celltype_folder_path_test, file, "signal_t.pickle")
            sample_file_mean = os.path.join(celltype_folder_path_test, file, "signal_t_mean.pickle")

            samples_dict = dict(zip(query_qaryindices, arr))
            save_data(samples_dict, sample_file)
            save_data(samples_dict, sample_file_mean)

    # Remove empirical mean
    for file in file_path:
        mean_file = os.path.join(celltype_folder_path, file, "train_mean.npy")
        all_samples_mean = np.load(mean_file)

        sample_file_mean = os.path.join(celltype_folder_path_test, file, "signal_t_mean.pickle")
        sample_file = os.path.join(celltype_folder_path_test, file, "signal_t.pickle")
        samples_dict = load_data(sample_file_mean)

        all_values = list(samples_dict.values())
        all_values = np.array(all_values, dtype=complex) - all_samples_mean
        samples_dict = {key: value for key, value in zip(samples_dict.keys(), all_values)}
        save_data(samples_dict, sample_file)