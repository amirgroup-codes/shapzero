"""
Script to compute Fourier coefficients for a given q, n, b, num_subsample, and num_repeat 
Not model specific
"""
import numpy as np
import sys
from pathlib import Path
sys.path.append("..")
from src.helper import Helper
from gen.utils import summarize_results
import pickle
import time
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="Run q-sft.")
    parser.add_argument("--q", required=True)
    parser.add_argument("--n", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--noise_sd", required=True)
    parser.add_argument("--num_subsample", required=True)
    parser.add_argument("--num_repeat", required=True)
    parser.add_argument("--hyperparam", required=False, default=False)
    parser.add_argument("--hyperparam_range", required=False, default=False)
    parser.add_argument("--param", required=False, default=None)
    return parser.parse_args()

def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    
if __name__ == "__main__":
    np.random.seed(20)
    start_time = time.time()
    args = parse_args()



    """
    q-sft parameters
    """
    # q = Alphabet size
    # n = Length of sequence
    # b = inner dimension of subsampling
    # noise_sd = hyperparameter: proxy for the amount of additive noise in the signal
    q = int(args.q)
    n = int(args.n)
    N = q ** n
    b = int(args.b)
    noise_sd = float(args.noise_sd)

    # num_sample and num_repeat control the amount of samples computed 
    num_subsample = int(args.num_subsample)
    num_repeat = int(args.num_repeat)

    # hyperparam: hyperparameter tune noise_sd
    hyperparam = str_to_bool(args.hyperparam)

    # Other q-sft parameters - leave as default
    t = 3
    delays_method_source = "identity" 
    delays_method_channel = "nso"


    """
    Initialization
    """
    current_directory = Path(__file__).resolve().parent
    folder = current_directory / "results" / f"q{q}_n{n}_b{b}" 
    folder.mkdir(parents=True, exist_ok=True)
    query_args = {
        "query_method": "complex",
        "num_subsample": num_subsample,
        "delays_method_source": delays_method_source,
        "subsampling_method": "qsft",
        "delays_method_channel": delays_method_channel,
        "num_repeat": num_repeat,
        "b": b,
        "t": t,
        "folder": folder 
    }
    signal_args = {
                    "n":n,
                    "q":q,
                    "noise_sd":noise_sd,
                    "query_args":query_args,
                    "t": t,
                    "type": str(args.param)
                    }
    test_args = {
            "n_samples": 10000
        }



    """
    Recover Fourier coefficients and get summary statistics
    """
    print('----------')
    print("Sampling from model")
    start_time_sampling = time.time()
    helper = Helper(signal_args=signal_args, methods=["qsft"], subsampling_args=query_args, test_args=test_args, exp_dir=folder)
    end_time_sampling = time.time()
    elapsed_time_sampling = end_time_sampling - start_time_sampling
    print(f"Sampling time: {elapsed_time_sampling} seconds")

    model_kwargs = {}
    model_kwargs["num_subsample"] = num_subsample
    model_kwargs["num_repeat"] = num_repeat
    model_kwargs["b"] = b
    test_kwargs = {}
    model_kwargs["n_samples"] = num_subsample * (helper.q ** b) * num_repeat * (helper.n + 1)

    if hyperparam:
        print('Hyperparameter tuning noise_sd:')
        start_time_hyperparam = time.time()
        range_values = [float(x) for x in args.hyperparam_range.strip('[]').split(', ')]
        noise_sd = np.arange(range_values[0], range_values[1], range_values[2]).round(2)
        nmse_entries = []
        r2_entries = []

        for noise in noise_sd:
            signal_args.update({
                "noise_sd": noise
            })
            model_kwargs["noise_sd"] = noise
            model_result = helper.compute_model(method="qsft", model_kwargs=model_kwargs, report=True, verbosity=0)
            test_kwargs["beta"] = model_result.get("gwht")
            nmse, r2 = helper.test_model("qsft", **test_kwargs)
            gwht = model_result.get("gwht")
            locations = model_result.get("locations")
            n_used = model_result.get("n_samples")
            avg_hamming_weight = model_result.get("avg_hamming_weight")
            max_hamming_weight = model_result.get("max_hamming_weight")
            nmse_entries.append(nmse)
            r2_entries.append(r2)
            print(f"noise_sd: {noise} - NMSE: {nmse}, R2: {r2}")

        end_time_hyperparam= time.time()
        elapsed_time_hyperparam = end_time_hyperparam - start_time_hyperparam
        min_nmse_ind = nmse_entries.index(min(nmse_entries))
        min_nmse = nmse_entries[min_nmse_ind]
        print('----------')
        print(f"Hyperparameter tuning time: {elapsed_time_hyperparam} seconds")
        print(f"noise_sd: {noise_sd[min_nmse_ind]} - Min NMSE: {min_nmse}")

        # Recompute qsft with the best noise_sd
        signal_args.update({
            "noise_sd": noise_sd[min_nmse_ind]
        })
        model_kwargs["noise_sd"] = noise_sd[min_nmse_ind]
        model_result = helper.compute_model(method="qsft", model_kwargs=model_kwargs, report=True, verbosity=0)
        test_kwargs["beta"] = model_result.get("gwht")
        nmse, r2_value = helper.test_model("qsft", **test_kwargs)
        gwht = model_result.get("gwht")
        locations = model_result.get("locations")
        n_used = model_result.get("n_samples")
        avg_hamming_weight = model_result.get("avg_hamming_weight")
        max_hamming_weight = model_result.get("max_hamming_weight")

        plt.figure()
        plt.title(f'q{q}_n{n}_b{b}')
        plt.plot(noise_sd, nmse_entries[:], marker='o', linestyle='-', color='b')
        plt.scatter(noise_sd[min_nmse_ind], nmse_entries[min_nmse_ind], color='red', marker='x', label='Min NMSE')
        plt.text(noise_sd[min_nmse_ind], nmse_entries[min_nmse_ind], f'noise_sd: {noise_sd[min_nmse_ind]} - Min NMSE: {min_nmse:.2f}', ha='right', va='top')
        plt.xlabel('noise_sd')
        plt.ylabel('NMSE')
        plt.savefig(str(folder) + '/nmse.png')  
        df = pd.DataFrame({'noise_sd': noise_sd, 'nmse': nmse_entries})
        df.to_csv(str(folder) + '/nmse.csv', index=False)

    else:
        print('Running q-sft')
        model_kwargs["noise_sd"] = noise_sd
        start_time_qsft = time.time()
        model_result = helper.compute_model(method="qsft", model_kwargs=model_kwargs, report=True, verbosity=0)
        test_kwargs["beta"] = model_result.get("gwht")
        nmse, r2_value = helper.test_model("qsft", **test_kwargs)
        gwht = model_result.get("gwht")
        locations = model_result.get("locations")
        n_used = model_result.get("n_samples")
        avg_hamming_weight = model_result.get("avg_hamming_weight")
        max_hamming_weight = model_result.get("max_hamming_weight")
        end_time_qsft = time.time()
        elapsed_time_qsft = end_time_qsft - start_time_qsft
        print('----------')
        print(f"q-sft time: {elapsed_time_qsft} seconds")
        print(f"R^2 is {r2_value}")
        print(f"NMSE is {nmse}")
        
    if args.param:
        qsft_path = f'qsft_transform_{str(args.param)}.pickle'
        qsft_indices = f'qsft_indices_{str(args.param)}.pickle'
        param_path = args.param.replace('/', '_')
    with open(str(folder) + "/" + "qsft_transform_" + str(param_path) + ".pickle", "wb") as pickle_file:
        pickle.dump(gwht, pickle_file)

    if args.param:
        param_path = args.param.replace('/', '_')
        os.rename(folder / 'test_samples.png', folder / 'test_samples_{}.png'.format(param_path))

    summarize_results(locations, gwht, q, n, b, noise_sd, n_used, r2_value, nmse, avg_hamming_weight, max_hamming_weight, folder, args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time: {elapsed_time} seconds")
    print('----------')