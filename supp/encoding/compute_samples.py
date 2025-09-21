import pickle
import numpy as np
from tqdm import tqdm
import random
import os
random.seed(42)
import matplotlib.pyplot as plt
import sys
import argparse
import zlib
sys.path.append('../..')
from gen.utils import save_data, load_data
import argparse
import pandas as pd
import torch
from torch import nn
from encoding_utils import infer, MLP



def parse_args():
    parser = argparse.ArgumentParser(description="Run q-sft.")
    parser.add_argument("--q", required=True)
    parser.add_argument("--n", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--num_subsample", required=True)
    parser.add_argument("--num_repeat", required=True)
    parser.add_argument("--param", required=True)
    parser.add_argument("--encoding", required=True)
    return parser.parse_args()
def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False

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
    file_path = [args.param]
    encoding = int(args.encoding)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=36).to(device)
    model.load_state_dict(torch.load("model/mlp_model.pt", map_location=device))
    model.eval()
    
    current_directory = os.path.join(os.getcwd(), 'results') # Go up one directory into results



    """
    Initialize files
    """
    folder_path = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        os.makedirs(os.path.join(folder_path, "train"))
        os.makedirs(os.path.join(folder_path, "train", "samples"))
        os.makedirs(os.path.join(folder_path, "test"))
    folder_path = os.path.join(folder_path, "train", "samples")
    for file in file_path:
        if not os.path.exists(os.path.join(folder_path, file)):
            os.makedirs(os.path.join(folder_path, file))
    folder_path = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "train", "samples_mean")
    for file in file_path:
        if not os.path.exists(os.path.join(folder_path, file)):
            os.makedirs(os.path.join(folder_path, file))
    folder_path = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "test")
    for file in file_path:
        if not os.path.exists(os.path.join(folder_path, file)):
            os.makedirs(os.path.join(folder_path, file))



    """
    Compute samples needed
    """
    for i in range(M):
        for j in range(D):
            query_indices_file = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "train", "samples", "M{}_D{}_queryindices.pickle".format(i, j))
            query_indices = load_data(query_indices_file)
            flag = True

            # Loop through all files and check if they exist
            for file in file_path:
                sample_file = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "train", "samples", file, "M{}_D{}.pickle".format(i, j))
                sample_file_mean = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "train", "samples_mean", file, "M{}_D{}.pickle".format(i, j))
                if os.path.isfile(sample_file):
                    flag = False

            if flag:
                all_query_indices = np.concatenate(query_indices)

                all_samples = np.zeros((np.shape(all_query_indices)[0], len(file_path)))
                block_length = len(query_indices[0])
                samples = [np.zeros((len(query_indices), block_length), dtype=complex) for _ in range(len(file_path))]
                print('Computing samples for M{}_D{}:'.format(i, j))
                all_samples[:,0] = infer(all_query_indices, model, device=device, change_encoding=encoding)

                for file, sample, arr in zip(file_path, samples, all_samples.T):
                    for k in range(len(query_indices)):
                        sample[k] = arr[k * block_length: (k+1) * block_length]
                    sample = sample.T
                    save_data(sample, sample_file)
                    save_data(sample, sample_file_mean)



    # Save the empirical mean separately
    folder_path = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b))
    for file in file_path:
        mean_file = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "train", "samples", file, "train_mean.npy") 

        if not os.path.isfile(mean_file):
            all_samples = []
            for i in range(M):
                for j in range(D):
                    sample_file = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "train", "samples_mean", file, "M{}_D{}.pickle".format(i, j))
                    samples = load_data(sample_file)
                    samples = np.concatenate(samples)
                    all_samples = np.concatenate([all_samples, samples])
            all_samples_mean = np.mean(all_samples)
            np.save(mean_file, all_samples_mean)
        else:
            all_samples_mean = np.load(mean_file)
        
        for i in range(M):
            for j in range(D):
                sample_file_zeromean = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "train", "samples", file, "M{}_D{}.pickle".format(i, j))
                sample_file = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "train", "samples_mean", file, "M{}_D{}.pickle".format(i, j))
                samples = load_data(sample_file)
                samples_zeromean = samples - all_samples_mean
                save_data(samples_zeromean, sample_file_zeromean)



    """
    Testing samples to compute NMSE and R^2
    """
    query_indices_file = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "test", "signal_t_queryindices.pickle")
    query_indices = load_data(query_indices_file)

    query_qaryindices_file = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "test", "signal_t_query_qaryindices.pickle")
    query_qaryindices = load_data(query_qaryindices_file)

    # Loop through all files and check if they exist
    for file in file_path:
        sample_file = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "test", file, "signal_t.pickle")
        sample_file_mean = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "test", file, "signal_t_mean.pickle")
        flag = True

        if os.path.isfile(sample_file):
            flag = False

    if flag:
        all_query_indices = query_indices

        all_samples = np.zeros((np.shape(all_query_indices)[0], len(file_path)))
        block_length = len(query_indices[0])
        samples = [np.zeros((len(query_indices), block_length), dtype=complex) for _ in range(len(file_path))]
        print('Computing test samples:')
        all_samples[:,0] = infer(all_query_indices, model, device=device, change_encoding=encoding)
    
        for file, arr in zip(file_path, all_samples.T):
            sample_file = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "test", file, "signal_t.pickle")
            sample_file_mean = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "test", file, "signal_t_mean.pickle")
            samples_dict = dict(zip(query_qaryindices, arr))
            save_data(samples_dict, sample_file)
            save_data(samples_dict, sample_file_mean)

        # Remove empirical mean
        for file in file_path:
            mean_file = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "train", "samples", file, "train_mean.npy")
            all_samples_mean = np.load(mean_file)

            sample_file_mean = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "test", file, "signal_t_mean.pickle")
            sample_file = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "test", file, "signal_t.pickle")
            samples_dict = load_data(sample_file_mean)

            all_values = list(samples_dict.values())
            all_values = np.array(all_values, dtype=complex) - all_samples_mean
            samples_dict = {key: value for key, value in zip(samples_dict.keys(), all_values)}
            save_data(samples_dict, sample_file)