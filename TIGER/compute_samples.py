"""
Model-specific script to compute samples given the q-ary indices from get_qary_indices.py
For TIGER: computes both on_target and titration scores in one shot. 
Note: titration scores take significantly longer to compute than on_target scores. For only on_target scores, we recommend commenting out the titration score parts of the code.
"""
import pickle
import numpy as np
from tqdm import tqdm
import random
import os
random.seed(42)
import matplotlib.pyplot as plt
import sys
sys.path.append('../tiger/hugging_face/')
sys.path.append('../..')
from src.utils import batch_prediction, batch_load, process_titration, find_target_sequence, process_on_target
from tiger_class import TranscriptProcessor
from gen.utils import save_data, load_data
import argparse
import zlib



def parse_args():
    parser = argparse.ArgumentParser(description="Run q-sft.")
    parser.add_argument("--q", required=True)
    parser.add_argument("--n", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--num_subsample", required=True)
    parser.add_argument("--num_repeat", required=True)
    parser.add_argument("--param", required=True)
    parser.add_argument("--gpu", required=True)
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

    # TIGER-specific arguments
    gpu = str_to_bool(args.gpu)
    param = args.param
    batch_size = 512
    postprocess = True
    tiger_dir = '../tiger/hugging_face/'
    if param == 'all':
        file_path = ['on_target', 'titration']
        mode = 'titration'
    elif param == 'on_target':
        file_path = ['on_target']
        mode = 'all' # all = PM/on_target setting in TIGER 
    elif param == 'titration':
        file_path = ['titration']
        mode = 'titration'
    if gpu:
        gpu_num = 0
        processor = TranscriptProcessor(tiger_dir=tiger_dir, postprocess=postprocess, mode=mode, gpu_num=gpu_num)
    else:
        processor = TranscriptProcessor(tiger_dir=tiger_dir, postprocess=postprocess, mode=mode)

    current_directory = os.path.join(os.getcwd(), '..', 'results') # Go up one directory into results

    # Define q-ary encoding 
    encoding = {0:'A', 1:'C', 2:'T', 3:'G'}
    


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
    Compute samples needed for F-SHAP
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
                # Preprocess and find target sequence
                all_query_indices = np.concatenate(query_indices)
                seqs, _ = find_target_sequence(all_query_indices, n)

                # Compute samples
                all_samples = np.zeros((len(seqs), len(file_path)))
                block_length = len(query_indices[0])
                samples = [np.zeros((len(query_indices), block_length), dtype=complex) for _ in range(len(file_path))]
                if param == 'all':
                    on_target_scores = []
                    titration_scores = []
                    with tqdm(total=len(seqs), desc="Processing M{} D{}".format(i, j)) as pbar:
                        for seq_batch in batch_load(seqs, batch_size):
                            df_on_target, df_titration, _ = batch_prediction(seq_batch, processor=processor)
                            on_target_scores_batch = process_on_target(df_on_target)
                            titration_scores_batch = process_titration(df_titration)
                            on_target_scores = on_target_scores + on_target_scores_batch
                            titration_scores = titration_scores + titration_scores_batch
                            pbar.update(len(seq_batch))
                    all_samples[:,0] = on_target_scores
                    all_samples[:,1] = titration_scores
                elif param == 'on_target':
                    on_target_scores = []
                    with tqdm(total=len(seqs), desc="Processing M{} D{}".format(i, j)) as pbar:
                        for seq_batch in batch_load(seqs, batch_size):
                            df_on_target, _, _ = batch_prediction(seq_batch, processor=processor)
                            on_target_scores_batch = process_on_target(df_on_target)
                            on_target_scores = on_target_scores + on_target_scores_batch
                            pbar.update(len(seq_batch))
                    all_samples[:,0] = on_target_scores
                elif param == 'titration':
                    titration_scores = []
                    with tqdm(total=len(seqs), desc="Processing M{} D{}".format(i, j)) as pbar:
                        for seq_batch in batch_load(seqs, batch_size):
                            _, df_titration, _ = batch_prediction(seq_batch, processor=processor)
                            titration_scores_batch = process_titration(df_titration)
                            titration_scores = titration_scores + titration_scores_batch
                            pbar.update(len(seq_batch))
                    all_samples[:,1] = titration_scores

                for file, sample, arr in zip(file_path, samples, all_samples.T):
                    sample_file = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "train", "samples", file, "M{}_D{}.pickle".format(i, j))
                    sample_file_mean = os.path.join(current_directory, "q{}_n{}_b{}".format(q, n, b), "train", "samples_mean", file, "M{}_D{}.pickle".format(i, j))
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
        # Preprocess and find target sequence
        all_query_indices = query_indices
        seqs, _ = find_target_sequence(all_query_indices, n)
        
        # Compute samples
        all_samples = np.zeros((len(seqs), len(file_path)))
        block_length = len(query_indices[0])
        samples = [np.zeros((len(query_indices), block_length), dtype=complex) for _ in range(len(file_path))]
        if param == 'all':
            on_target_scores = []
            titration_scores = []
            with tqdm(total=len(seqs), desc="Processing M{} D{}".format(i, j)) as pbar:
                for seq_batch in batch_load(seqs, batch_size):
                    df_on_target, df_titration, _ = batch_prediction(seq_batch, processor=processor)
                    on_target_scores_batch = process_on_target(df_on_target)
                    titration_scores_batch = process_titration(df_titration)
                    on_target_scores = on_target_scores + on_target_scores_batch
                    titration_scores = titration_scores + titration_scores_batch
                    pbar.update(len(seq_batch))
            all_samples[:,0] = on_target_scores
            all_samples[:,1] = titration_scores
        elif param == 'on_target':
            on_target_scores = []
            with tqdm(total=len(seqs), desc="Processing M{} D{}".format(i, j)) as pbar:
                for seq_batch in batch_load(seqs, batch_size):
                    df_on_target, _, _ = batch_prediction(seq_batch, processor=processor)
                    on_target_scores_batch = process_on_target(df_on_target)
                    on_target_scores = on_target_scores + on_target_scores_batch
                    pbar.update(len(seq_batch))
            all_samples[:,0] = on_target_scores
        elif param == 'titration':
            titration_scores = []
            with tqdm(total=len(seqs), desc="Processing M{} D{}".format(i, j)) as pbar:
                for seq_batch in batch_load(seqs, batch_size):
                    _, df_titration, _ = batch_prediction(seq_batch, processor=processor)
                    titration_scores_batch = process_titration(df_titration)
                    titration_scores = titration_scores + titration_scores_batch
                    pbar.update(len(seq_batch))
            all_samples[:,1] = titration_scores

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