"""
Script to compute SHAP values from Kernel SHAP and Faith-Shap interactions from SHAP-IQ
Output is shap_values.npy, which contains a num_properties x num_samples x n matrix of SHAP values, or a num_samples x n matrix of SHAP values. Time saved in time_shap_values.npy
Output is fsi.pickle, which contains a list of dictionaries containing the interactions at each position for each sequence. q-ary samples are located at fsi_samples.npy. Time saved in time_fsi.npy
"""
import numpy as np
import random
random.seed(42)
np.random.seed(42)
import os
import pandas as pd
import shap
import sys
sys.path.append('..')
from src.utils import df_str_to_encoding, get_predictions_shap, get_predictions_fsi_HEK293_frameshift, get_predictions_fsi_HEK293_1bpins, get_predictions_fsi_U2OS_1bpins
import time
import shapiq
from tqdm import tqdm
import pickle
from collections import OrderedDict

encoding = {0:'A', 1:'C', 2:'T', 3:'G'}
max_order = 3
budget = 20000



"""
Run Kernel SHAP and SHAP-IQ for HEK293
"""
q = 4
n = 40
celltype = 'HEK293'
df = pd.read_csv('data/{}_heldout.csv'.format(celltype))
df_train = pd.read_csv('data/{}_train.csv'.format(celltype))

# Select a subset of samples 
num_background_samples = 30
df_background = df_train.sample(n=num_background_samples, random_state=42)
background = df_str_to_encoding(df_background)

# Compute Shapley values
x_valid = df_str_to_encoding(df)
start_time = time.time()
e = shap.KernelExplainer(get_predictions_shap, background)
shap_values = e.shap_values(x_valid)
end_time = time.time()
elapsed_time = end_time - start_time
average_time_per_sample = elapsed_time / np.shape(x_valid)[0]
np.save('shap_results/time_shap_values_{}.npy'.format(celltype), average_time_per_sample)
np.save('shap_results/shap_values_{}.npy'.format(celltype), shap_values)



# Compute interactions using SHAP-IQ for HEK293 frameshift
property = 'frameshift'
num_iq_valid_samples = 4
np.random.shuffle(x_valid)
x_valid = np.array(x_valid)[0:num_iq_valid_samples,:]
np.save('shap_results/fsi_{}_{}_samples.npy'.format(celltype, property), x_valid)

start_time = time.time()
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

end_time = time.time()
elapsed_time = end_time - start_time
average_time_per_sample = elapsed_time / np.shape(x_valid)[0]
np.save('shap_results/time_fsi_{}_{}.npy'.format(celltype, property), average_time_per_sample)
with open('shap_results/fsi_{}_{}.pickle'.format(celltype, property), 'wb') as file:
    pickle.dump(shap_interactions, file)     



# 1-bp-ins
property = '1bpins'
num_iq_valid_samples = 4
np.random.shuffle(x_valid)
x_valid = np.array(x_valid)[0:num_iq_valid_samples,:]
np.save('shap_results/fsi_{}_{}_samples.npy'.format(celltype, property), x_valid)

start_time = time.time()
shap_interactions = []
for x in tqdm(x_valid, desc="Computing interactions"):
    explainer = shapiq.TabularExplainer(
        model=get_predictions_fsi_HEK293_1bpins,
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

end_time = time.time()
elapsed_time = end_time - start_time
average_time_per_sample = elapsed_time / np.shape(x_valid)[0]
np.save('shap_results/time_fsi_{}_{}.npy'.format(celltype, property), average_time_per_sample)
with open('shap_results/fsi_{}_{}.pickle'.format(celltype, property), 'wb') as file:
    pickle.dump(shap_interactions, file)     



"""
Run Kernel SHAP and SHAP-IQ for U2OS
"""



# # U2OS

# # compute Shapley interactions using SHAP-IQ
# num_iq_valid_samples = 4
# np.random.shuffle(x_valid)
# x_valid = np.array(x_valid)[0:num_iq_valid_samples,:]
# np.save('shap_results/shap_interactions_{}_{}_samples2.npy'.format(celltype, property), x_valid)

# start_time = time.time()
# shap_interactions = []
# for x in tqdm(x_valid, desc="Computing interactions"):
#     explainer = shapiq.TabularExplainer(
#         model=get_predictions_shap_interactions,
#         data=background,
#         index="FSII",
#         max_order=max_order,
#     )
#     interaction_values = explainer.explain(x, budget=budget)

#     # Save out data
#     interactions = interaction_values.interaction_lookup.copy()
#     interactions = OrderedDict(sorted(interactions.items(), key=lambda item: item[1]))
#     vals = interaction_values.values.copy()
#     keys = list(interactions.keys())
#     for i, key in enumerate(keys):
#         interactions[key] = vals[i]
#     shap_interactions.append(interactions)

# end_time = time.time()
# elapsed_time = end_time - start_time
# average_time_per_sample = elapsed_time / np.shape(x_valid)[0]
# np.save('shap_results/time_shap_interactions2.npy', average_time_per_sample)
# with open('shap_results/shap_interactions_{}_{}2.pickle'.format(celltype, property), 'wb') as file:
#     pickle.dump(shap_interactions, file)     
