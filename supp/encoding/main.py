import subprocess
import sys
import numpy as np
import pandas as pd
sys.path.append('../../..')
sys.path.append('../../TIGER')

change_encoding = np.arange(4)
df = pd.read_csv('mpsa.csv')

q = 4
n = 9
b = 5

df_results = pd.DataFrame()

for enc in change_encoding:

    # Get q-ary indices
    cmd = [
        "python", 
        "get_qary_indices.py",
        "--q", f"{q}",
        "--n", f"{n}",
        "--b", f"{b}",
        "--num_subsample", "3",
        "--num_repeat", "3"
    ]
    subprocess.run(cmd, check=True)

    # Compute samples
    cmd = [
        "python", 
        "compute_samples.py",
        "--q", f"{q}",
        "--n", f"{n}",
        "--b", f"{b}",
        "--num_subsample", "3",
        "--num_repeat", "3",
        "--param", f"encoding_{enc}",
        "--encoding", str(enc)
    ]
    subprocess.run(cmd, check=True)

    # Run q-SFT
    cmd = [
        "python", 
        "run_helper.py",
        "--q", f"{q}",
        "--n", f"{n}",
        "--b", f"{b}",
        "--num_subsample", "3",
        "--num_repeat", "3",
        "--param", f"encoding_{enc}",
        "--noise_sd", "0.725",
        "--hyperparam", "True",
        "--hyperparam_range", "[0, 1, 0.1]"
    ]
    subprocess.run(cmd, check=True)

    # Extract encoding and R^2
    file_path = f"results/q4_n9_b5/helper_results_encoding_{enc}.txt"
    with open(file_path, 'r') as f:
        for line in f:
            if "R^2 =" in line:
                r2_value = float(line.strip().split('=')[1])

    og = np.array([0, 1, 2, 3])
    encoding = {'A': 0, 'C': 1, 'U': 2, 'G': 3}
    new_encoding = {}
    if enc > 3:
        np.random.seed(enc)
        remapped = np.random.permutation(4) 
    else:
        remapped = (og + enc) % 4

    index_to_base = {v: k for k, v in encoding.items()}
    new_encoding = {base: remapped[idx] for idx, base in index_to_base.items()}

    # Get sparsity and max order
    gwht = np.load('results/q4_n9_b5/qsft_transform_encoding_{}.pickle'.format(enc), allow_pickle=True)
    s = len(gwht)
    max_order = max(sum(1 for x in k if x != 0) for k in gwht.keys())

    df_results = df_results.append({
        'encoding': new_encoding,
        'r2': r2_value,
        'sparsity': s,
        'max_order': max_order
    }, ignore_index=True)

# Compute mean and std of s and R^2
mean = df_results['sparsity'].mean()
std = df_results['sparsity'].std()
print('Mean sparsity = {}'.format(mean))
print('Std sparsity = {}'.format(std))

mean = df_results['r2'].mean()
std = df_results['r2'].std()
print('Mean R^2 = {}'.format(mean))
print('Std R^2 = {}'.format(std))

df_results.to_csv('results/encoding_results.csv')