import numpy as np
import random
random.seed(42)
np.random.seed(42)
import os
import pandas as pd
import shap
import sys
sys.path.append('..')
from src.utils import df_str_to_encoding, get_predictions_shap, get_predictions_fsi_HEK293_frameshift
import time
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pickle
import shapiq
from tqdm import tqdm
from collections import OrderedDict



"""
Run Kernel SHAP and SHAP-IQ for HEK293
"""
q = 4
n = 40
celltype = 'HEK293'
df = pd.read_csv('data/{}_heldout.csv'.format(celltype))
df_train = pd.read_csv('data/{}_train.csv'.format(celltype))

# # Select a subset of samples 
# num_background_samples = 50
# df_background = df_train.sample(n=num_background_samples, random_state=42)

# # Check converge of KernelSHAP given different number of background samples on a small dataset
# num_background_samples_total = [5, 10, 15, 30, 40, 50]
# num_heldout_samples = 10
# for num_background_samples in num_background_samples_total:
#     df_background_i = df_background.head(num_background_samples)
#     background = df_str_to_encoding(df_background_i)
#     x_valid = df_str_to_encoding(df.head(num_heldout_samples))
#     e = shap.KernelExplainer(get_predictions_shap, background)
#     shap_values = e.shap_values(x_valid)
#     np.save('samples_search/shap_values_{}_{}.npy'.format(celltype, num_background_samples), shap_values)

# last_shap_values = np.load('samples_search/shap_values_{}_{}.npy'.format(celltype, num_background_samples_total[-1]))
# plt.figure()
# r2s = []
# for num_background_samples in num_background_samples_total:
#     shap_values = np.load('samples_search/shap_values_{}_{}.npy'.format(celltype, num_background_samples))
#     r2 = r2_score(last_shap_values.flatten(), shap_values.flatten())
#     r2s.append(r2)

# plt.plot(num_background_samples_total, r2s)
# plt.savefig('samples_search/r2_{}.png'.format(celltype))



# Check convergence of Faith-Shap given different budgets and a fixed background dataset
num_background_samples = 30
df_background = df_train.sample(n=num_background_samples, random_state=42)
background = df_str_to_encoding(df_background)
property = 'frameshift'

budgets = [30000]
num_heldout_samples = 1
max_order = 3
x_valid = df_str_to_encoding(df)
np.random.shuffle(x_valid)
x_valid = np.array(x_valid)[0:num_heldout_samples,:]
for budget in budgets:
    shap_interactions = []
    for x in tqdm(x_valid, desc="Computing interactions"):
        explainer = shapiq.TabularExplainer(
            model=get_predictions_fsi_HEK293_frameshift,
            data=background,
            index="FSII",
            max_order=max_order,
        )
        interaction_values = explainer.explain(x, budget=budget)

        # Save out data
        interactions = interaction_values.interaction_lookup.copy()
        interactions = OrderedDict(sorted(interactions.items(), key=lambda item: item[1]))
        vals = interaction_values.values.copy()
        keys = list(interactions.keys())
        for i, key in enumerate(keys):
            interactions[key] = vals[i]
        shap_interactions.append(interactions)

    with open('samples_search/fsi_{}_{}.pickle'.format(celltype, budget), 'wb') as file:
        pickle.dump(shap_interactions, file)     

