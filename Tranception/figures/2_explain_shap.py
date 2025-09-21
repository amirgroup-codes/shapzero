import pandas as pd
import numpy as np
import sys
import os
sys.path.append('..')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.utils import ModelScorer
import time
import shap
from tqdm import tqdm
import pickle
import shapiq
import torch
import torch.nn as nn
sys.path.append('../../TIGER/fastshap-main')
from fastshap import FastSHAP
from fastshap.utils import MaskLayer1d
from fastshap import Surrogate, KLDivLoss
from sklearn.model_selection import train_test_split


def measure_shap(property, data_val, background, max_order, budget, num_iq_valid_samples, function=None):

    x_valid = data_val.copy()
    
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
q = 20
n = 10
b = 3
properties = ['GFP_Tranception']
max_order = 2

for property in properties:

    data_train = np.load(f'data/{property}_train.npy')
    data_val = np.load(f'data/{property}_heldout.npy')
    budget = 50000
    num_iq_valid_samples = 50

    background = data_train.copy()
    
    positions = [83, 87, 82, 84, 43, 164, 47, 162, 140, 165] # 0-indexed
    DMS='GFP_AEQVI_Sarkisyan_2016' 
    DMS_list = pd.read_csv('../data/ProteinGym_reference_file_substitutions.csv')
    DMS_list = DMS_list[DMS_list['DMS_id'] == DMS]
    sequence = DMS_list["target_seq"].values[0].upper()
    scorer = ModelScorer(sequence, positions, DMS, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), property=property)
    measure_shap(property, data_val, background, max_order, budget, num_iq_valid_samples, function=scorer.compute_scores_from_numpy_arr)



    """
    FastSHAP
    """
    # Train surrogate model
    background = np.load(f'data/{property}_train.npy')
    x_train = np.array(background)
    x_train, x_val = train_test_split(x_train, test_size=0.2, random_state=42)

    x_valid = data_val.copy()
    x_valid = np.array(x_valid)
    device = torch.device('cuda')
    if os.path.isfile('shap_results/surrogate_{}.pt'.format(property)):
        print('Loading saved surrogate model')
        surr = torch.load('shap_results/surrogate_{}.pt'.format(property), weights_only=False).to(device)
        surrogate = Surrogate(surr, n)

    else:
        scorer.load_training_scores(background)
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
        surrogate.train_original_model(
            x_train,
            x_val,
            scorer.compute_scores_fastshap,
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
    if not os.path.isfile('shap_results/explainer_{}.pt'.format(property)):
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
            batch_size=32,
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
