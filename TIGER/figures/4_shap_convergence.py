import numpy as np
import sys
sys.path.append('..')
sys.path.append('../tiger/hugging_face/')
sys.path.append('../fastshap-main')
import random
import os
import pandas as pd
random.seed(42)
np.random.seed(42)
import shap
import shapiq
from tqdm import tqdm
from src.utils import get_predictions_shap_ontarget, get_predictions_shap_titration, deepshap_batch_explain, get_predictions_fastshap_on_target, get_predictions_fastshap_titration
import time
from tiger_class import TranscriptProcessor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pickle


encoding = {0:'A', 1:'C', 2:'T', 3:'G'}
reverse_encoding = {v: k for k, v in encoding.items()}


def measure_shap(property, data_val, background, max_order, budget, num_iq_valid_samples, function=None):

    x_valid = data_val['Full Sequence'].to_list()
    x_valid = [[reverse_encoding[num] for num in row] for row in x_valid]
    x_valid = np.array(x_valid)
    
    if not os.path.isfile('shap_results/shap_values_{}.npy'.format(property)):
        start_time = time.time()
        e = shap.KernelExplainer(function, background)
        shap_values = e.shap_values(x_valid)
        end_time = time.time()
        elapsed_time = end_time - start_time
        average_time_per_sample = elapsed_time / np.shape(x_valid)[0]
        np.save('shap_results/time_shap_values_{}.npy'.format(property), average_time_per_sample)  
        np.save('shap_results/shap_values_{}.npy'.format(property), shap_values)          
    
    # if not os.path.isfile('shap_results/fsi_{}.pickle'.format(property)):

    #     # SHAP-IQ is too expensive to compute for all samples
    #     np.random.shuffle(x_valid)
    #     x_valid = np.array(x_valid)[0:num_iq_valid_samples,:]
    #     np.save('data/fsi_{}_samples.npy'.format(property), x_valid)
    #     start_time = time.time()
    #     shap_interactions = []
    #     for x in tqdm(x_valid, desc="Computing interactions"):
    #         explainer = shapiq.TabularExplainer(
    #             model=function,
    #             data=background,
    #             index="FSII",
    #             max_order=max_order
    #         )
    #         interaction_values = explainer.explain(x, budget=budget)

    #         # Save out data
    #         interactions = interaction_values.interaction_lookup.copy()
    #         interactions = dict(sorted(interactions.items(), key=lambda item: item[1]))
    #         vals = interaction_values.values.copy()
    #         keys = list(interactions.keys())
    #         for i, key in enumerate(keys):
    #             interactions[key] = vals[i]
    #         shap_interactions.append(interactions)

    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     average_time_per_sample = elapsed_time / np.shape(x_valid)[0]
    #     np.save('shap_results/time_fsi_{}.npy'.format(property), average_time_per_sample)   
    #     with open('shap_results/fsi_{}.pickle'.format(property), 'wb') as file:
    #         pickle.dump(shap_interactions, file)         



"""
Using experimental sequences, use SHAP to explain their outputs
"""
q = 4
n = 26
properties = ['on_target']
max_order = 3

for property in properties:

    data_train = pd.read_csv(f'data/{property}_train.csv')
    data_val = pd.read_csv(f'data/{property}_heldout.csv')
    # num_background_samples = 50

    # # select a subset of samples from qsft to take an expectation over
    # df_background = data_train.sample(n=num_background_samples, random_state=42)
    # background = df_background['Full Sequence'].to_list()
    # background = [[reverse_encoding[num] for num in row] for row in background]
    # background = np.array(background)

    # # Check converge of KernelSHAP given different number of background samples on a small dataset
    # num_background_samples_total = [5, 10, 15, 30, 40, 50]
    # num_heldout_samples = 30
    # for num_background_samples in num_background_samples_total:
    #     background = df_background.head(num_background_samples)['Full Sequence'].to_list()
    #     background = [[reverse_encoding[num] for num in row] for row in background]
    #     background = np.array(background)
        
    #     x_valid = data_val.head(num_heldout_samples)['Full Sequence'].to_list()
    #     x_valid = [[reverse_encoding[num] for num in row] for row in x_valid]
    #     x_valid = np.array(x_valid)
        
    #     e = shap.KernelExplainer(get_predictions_shap_ontarget, background)
    #     shap_values = e.shap_values(x_valid)
    #     np.save('samples_search/shap_values_{}_{}.npy'.format(property, num_background_samples), shap_values)


    # last_shap_values = np.load('samples_search/shap_values_{}_{}.npy'.format(property, num_background_samples_total[-1]))
    # plt.figure()
    # r2s = []
    # for num_background_samples in num_background_samples_total:
    #     shap_values = np.load('samples_search/shap_values_{}_{}.npy'.format(property, num_background_samples))
    #     r2 = r2_score(last_shap_values.flatten(), shap_values.flatten())
    #     r2s.append(r2)

    # plt.plot(num_background_samples_total, r2s)
    # plt.savefig('samples_search/r2_{}_shap.png'.format(property))


    # Check convergence of Faith-Shap given different budgets and a fixed background dataset
    num_background_samples = 30
    df_background = data_train.sample(n=num_background_samples, random_state=42)
    background = df_background['Full Sequence'].to_list()
    background = [[reverse_encoding[num] for num in row] for row in background]
    background = np.array(background)

    budgets = [1000, 10000, 30000, 50000, 60000, 70000]
    num_heldout_samples = 2
    x_valid = data_val.head(num_heldout_samples)['Full Sequence'].to_list()
    x_valid = [[reverse_encoding[num] for num in row] for row in x_valid]
    x_valid = np.array(x_valid)
    np.random.shuffle(x_valid)
    x_valid = np.array(x_valid)[0:num_heldout_samples,:]
    for budget in budgets:
        shap_interactions = []
        for x in tqdm(x_valid, desc="Computing interactions"):
            explainer = shapiq.TabularExplainer(
                model=get_predictions_shap_ontarget,
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

