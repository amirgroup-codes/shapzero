import numpy as np
import sys
import os
sys.path.append('..')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append('../fastshap-main')
import random
import pandas as pd
random.seed(42)
np.random.seed(42)
import shap
import shapiq
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pickle
from src.utils import ModelScorer
import torch



"""
Using experimental sequences, use SHAP to explain their outputs
"""
q = 20
n = 10
properties = ['GFP_Tranception']
max_order = 2

for property in properties:

    data_train = np.load(f'data/{property}_train.npy')
    data_val = np.load(f'data/{property}_heldout.npy')

    num_background_samples = data_train.shape[0]
    positions = [83, 87, 82, 84, 43, 164, 47, 162, 140, 165] # 0-indexed
    DMS='GFP_AEQVI_Sarkisyan_2016' 
    DMS_list = pd.read_csv('../data/ProteinGym_reference_file_substitutions.csv')
    DMS_list = DMS_list[DMS_list['DMS_id'] == DMS]
    sequence = DMS_list["target_seq"].values[0].upper()
    scorer = ModelScorer(sequence, positions, DMS, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), property=property)
    num_background_samples_total = [10, 20, 30, 50, 70, num_background_samples]
    num_heldout_samples = 30
    index = np.random.choice(data_val.shape[0], size=num_heldout_samples, replace=False)
    x_valid = np.array(data_val[index])
    for num_background_samples in num_background_samples_total:
        index = np.random.choice(data_train.shape[0], size=num_background_samples, replace=False)
        background = np.array(data_train[index])
        e = shap.KernelExplainer(scorer.compute_scores_from_numpy_arr, background)
        shap_values = e.shap_values(x_valid)
        np.save('samples_search/shap_values_{}_{}.npy'.format(property, num_background_samples), shap_values)



    # Check convergence of Faith-Shap given different budgets and a fixed background dataset
    background = np.array(data_train.copy())
    budgets = [1000, 10000, 30000, 50000, 60000, 70000]
    num_heldout_samples = 10
    index = np.random.choice(data_val.shape[0], size=num_heldout_samples, replace=False)
    x_valid = np.array(data_val[index])
    for budget in budgets:
        shap_interactions = []
        for x in tqdm(x_valid, desc="Computing interactions"):
            explainer = shapiq.TabularExplainer(
                model=scorer.compute_scores_from_numpy_arr,
                data=background,
                index="FSII",
                max_order=max_order
            )
            interaction_values = explainer.explain(x, budget=budget)

            # Save out data
            interactions = interaction_values.interaction_lookup.copy()
            interactions = dict(sorted(interactions.items(), key=lambda item: item[1]))
            vals = interaction_values.values.copy()
            keys = list(interactions.keys())
            for i, key in enumerate(keys):
                interactions[key] = vals[i]
            shap_interactions.append(interactions)

        with open('samples_search/fsi_{}_{}.pickle'.format(property, budget), 'wb') as file:
            pickle.dump(shap_interactions, file)       
