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
import pickle
import torch
import torch.nn as nn
from fastshap import FastSHAP
from fastshap.utils import MaskLayer1d
from fastshap import Surrogate, KLDivLoss
from sklearn.model_selection import train_test_split

from tiger_class import TranscriptProcessor

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
    
    if not os.path.isfile('shap_results/fsi_{}.pickle'.format(property)):

        # SHAP-IQ is too expensive to compute for all samples
        np.random.shuffle(x_valid)
        x_valid = np.array(x_valid)[0:num_iq_valid_samples,:]
        np.save('data/fsi_{}_samples.npy'.format(property), x_valid)
        start_time = time.time()
        shap_interactions = []
        for x in tqdm(x_valid, desc="Computing interactions"):
            explainer = shapiq.TabularExplainer(
                model=function,
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

        end_time = time.time()
        elapsed_time = end_time - start_time
        average_time_per_sample = elapsed_time / np.shape(x_valid)[0]
        np.save('shap_results/time_fsi_{}.npy'.format(property), average_time_per_sample)   
        with open('shap_results/fsi_{}.pickle'.format(property), 'wb') as file:
            pickle.dump(shap_interactions, file)         



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

    if property == 'on_target':
        budget = 50000
        num_background_samples = 30
        num_iq_valid_samples = 8
    elif property == 'titration':
        budget = 20000
        num_background_samples = 30
        num_iq_valid_samples = 4

    # select a subset of samples from qsft to take an expectation over
    df_background = data_train.sample(n=num_background_samples, random_state=42)
    background = df_background['Full Sequence'].to_list()
    background = [[reverse_encoding[num] for num in row] for row in background]
    background = np.array(background)

    if property == 'on_target':
        measure_shap(property, data_val, background, max_order, budget, num_iq_valid_samples, function=get_predictions_shap_ontarget)
    elif property == 'titration':
        measure_shap(property, data_val, background, max_order, budget, num_iq_valid_samples, function=get_predictions_shap_titration)



    """
    FastSHAP
    """
    # Train surrogate model
    background = data_train['Full Sequence'].to_list()
    background = [[reverse_encoding[num] for num in row] for row in background]
    x_train = np.array(background)
    x_train, x_val = train_test_split(x_train, test_size=0.2, random_state=42)

    x_valid = data_val['Full Sequence'].to_list()
    x_valid = [[reverse_encoding[num] for num in row] for row in x_valid]
    x_valid = np.array(x_valid)
    device = torch.device('cuda')
    if os.path.isfile('shap_results/surrogate_{}.pt'.format(property)):
        print('Loading saved surrogate model')
        surr = torch.load('shap_results/surrogate_{}.pt'.format(property)).to(device)
        surrogate = Surrogate(surr, n)

    else:
        start_time = time.time()
        # Create surrogate model
        surr = nn.Sequential(
            MaskLayer1d(value=0, append=True),
            nn.Linear(2 * n, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 1)).to(device)

        # Set up surrogate object
        surrogate = Surrogate(surr, n)

        # Train
        if property == 'on_target':
            surrogate.train_original_model(
                x_train,
                x_val,
                get_predictions_fastshap_on_target,
                batch_size=256,
                max_epochs=100,
                loss_fn=nn.MSELoss(),
                lookback=5,
                bar=True,
                verbose=True)
        elif property == 'titration':
            surrogate.train_original_model(
                x_train,
                x_val,
                get_predictions_fastshap_titration,
                batch_size=256,
                max_epochs=100,
                loss_fn=nn.MSELoss(),
                lookback=5,
                bar=True,
                verbose=True)

        # Save surrogate
        end_time = time.time()
        elapsed_time = end_time - start_time
        np.save('shap_results/time_fastshap_surrogate_{}.npy'.format(property), elapsed_time)  
        surr.cpu()
        torch.save(surr, 'shap_results/surrogate_{}.pt'.format(property))
        surr.to(device)



    # Train FastSHAP
    # Check for model
    if os.path.isfile('shap_results/explainer_{}.pt'.format(property)):
        print('Loading saved explainer model')
        explainer = torch.load('shap_results/explainer_{}.pt'.format(property)).to(device)
        fastshap = FastSHAP(explainer, surrogate, normalization='additive')

    else:
        start_time = time.time()
        # Create explainer model
        explainer = nn.Sequential(
            nn.Linear(n, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n)).to(device)

        # Set up FastSHAP object
        fastshap = FastSHAP(explainer, surrogate, normalization='additive')

        # Train
        fastshap.train(
            x_train,
            x_val,
            batch_size=256,
            num_samples=len(x_train),
            max_epochs=200,
            eff_lambda=1e-2,
            validation_samples=len(x_val),
            lookback=10,
            bar=True,
            verbose=True)

        # Save explainer
        end_time = time.time()
        elapsed_time = end_time - start_time
        np.save('shap_results/time_fastshap_explainer_{}.npy'.format(property), elapsed_time)  
        explainer.cpu()
        torch.save(explainer, 'shap_results/explainer_{}.pt'.format(property))
        explainer.to(device)



    # Run FastSHAP
    start_time = time.time()
    x_valid = torch.from_numpy(x_valid).float()
    fastshap_values = fastshap.shap_values(x_valid)[:, :, 0]
    end_time = time.time()
    elapsed_time = (end_time - start_time) / len(data_val)
    np.save('shap_results/time_fastshap_inference_{}.npy'.format(property), elapsed_time)  
    np.save('shap_results/fastshap_{}.npy'.format(property), fastshap_values)



    """
    DeepSHAP
    """
    tiger_dir = '../tiger/hugging_face/'
    mode = 'titration'
    gpu_num = 0
    processor = TranscriptProcessor(tiger_dir=tiger_dir, mode=mode, gpu_num=gpu_num)
    start_time = time.time()
    df_shap = deepshap_batch_explain(processor=processor, train_data=data_train, heldout_data=data_val)
    end_time = time.time()
    elapsed_time = (end_time - start_time) / len(data_val)
    np.save('shap_results/time_deepshap_{}.npy'.format(property), elapsed_time)  
    df_shap.to_csv('shap_results/deepshap_{}.csv'.format(property))
