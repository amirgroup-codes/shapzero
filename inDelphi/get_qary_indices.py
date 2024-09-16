"""
Script to generate q-ary indices for a given q, n, b, num_subsample, and num_repeat 
Not model specific
"""
import numpy as np
import sys
from pathlib import Path
sys.path.append("..")
from src.helper import Helper
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="Get q-ary indices.")
    parser.add_argument("--q", required=True)
    parser.add_argument("--n", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--num_subsample", required=True)
    parser.add_argument("--num_repeat", required=True)
    return parser.parse_args()

def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    
if __name__ == "__main__":
    np.random.seed(20)
    args = parse_args()


    """
    q-sft parameters
    """
    # q = Alphabet size
    # n = Length of sequence
    # b = inner dimension of subsampling
    q = int(args.q)
    n = int(args.n)
    N = q ** n
    b = int(args.b)
    len_seq = n

    # num_sample and num_repeat control the amount of samples computed 
    num_subsample = int(args.num_subsample)
    num_repeat = int(args.num_repeat)


    """
    Initialization
    """
    current_directory = Path(__file__).resolve().parent
    # Select folder
    folder = current_directory / "results" / f"q{q}_n{n}_b{b}"
    folder.mkdir(parents=True, exist_ok=True)

    query_args = {
        "query_method": "generate_samples",
        "method": "generate_samples",
        "num_subsample": num_subsample,
        "num_repeat": num_repeat,
        "b": b,
        "folder": folder 
    }
    signal_args = {
                    "n":n,
                    "q":q,
                    "query_args":query_args,
                    "len_seq":len_seq
                    }
    test_args = {
            "n_samples": 10000,
            "method": "generate_samples"
        }
    file_path = os.path.join(folder, 'train', 'samples')
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
    file_path = os.path.join(folder, 'test')
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)

    print('Generating train and test samples')
    helper = Helper(signal_args=signal_args, methods=["qsft"], subsampling_args=query_args, test_args=test_args, exp_dir=folder)