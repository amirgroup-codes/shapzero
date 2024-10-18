import numpy as np
import pandas as pd
import sys
sys.path.append('..')
sys.path.append('../../TIGER/fastshap-main')
import torch
torch.manual_seed(0)
import torch.nn as nn
from fastshap import FastSHAP
from fastshap.utils import MaskLayer1d
from fastshap import Surrogate, KLDivLoss
from sklearn.model_selection import train_test_split
import time
from src.utils import df_str_to_encoding, get_predictions_fastshap_HEK293_frameshift
import os



"""
FastSHAP
"""
q = 4
n = 40
celltype = 'HEK293'
df = pd.read_csv('data/{}_heldout.csv'.format(celltype))
df_train = pd.read_csv('data/{}_train.csv'.format(celltype))
properties  = ['frameshift']

for property in properties:
    # Train surrogate model
    background = df_str_to_encoding(df_train)
    x_train = np.array(background)
    x_train, x_val = train_test_split(x_train, test_size=0.2, random_state=42)

    x_valid = df_str_to_encoding(df)
    x_valid = np.array(x_valid)
    device = torch.device('cuda')
    if os.path.isfile('shap_results/surrogate_{}_{}.pt'.format(celltype, property)):
        print('Loading saved surrogate model')
        surr = torch.load('shap_results/surrogate_{}_{}.pt'.format(celltype, property)).to(device)
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
        if property == 'frameshift':
            surrogate.train_original_model(
                x_train,
                x_val,
                get_predictions_fastshap_HEK293_frameshift,
                batch_size=256,
                max_epochs=100,
                loss_fn=nn.MSELoss(),
                lookback=5,
                bar=True,
                verbose=True)

        # Save surrogate
        end_time = time.time()
        elapsed_time = end_time - start_time
        np.save('shap_results/time_fastshap_surrogate_{}_{}.npy'.format(celltype, property), elapsed_time)
        surr.cpu()
        torch.save(surr, 'shap_results/surrogate_{}_{}.pt'.format(celltype, property))
        surr.to(device)



    # Train FastSHAP
    # Check for model
    if os.path.isfile('shap_results/explainer_{}_{}.pt'.format(celltype, property)):
        print('Loading saved explainer model')
        explainer = torch.load('shap_results/explainer_{}_{}.pt'.format(celltype, property)).to(device)
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
        np.save('shap_results/time_fastshap_explainer_{}_{}.npy'.format(celltype, property), elapsed_time)
        explainer.cpu()
        torch.save(explainer, 'shap_results/explainer_{}_{}.pt'.format(celltype, property))
        explainer.to(device)



    # # Run FastSHAP
    # start_time = time.time()
    # x_valid = torch.from_numpy(x_valid).float()
    # fastshap_values = fastshap.shap_values(x_valid)[:, :, 0]
    # end_time = time.time()
    # elapsed_time = (end_time - start_time) / len(df)
    # np.save('shap_results/time_fastshap_inference_{}_{}.npy'.format(celltype, property), elapsed_time)
    # np.save('shap_results/fastshap_{}_{}.npy'.format(celltype, property), fastshap_values)

    # # Run FastSHAP for the two extra sample sets
    # start_time = time.time()
    # df_extra = pd.read_csv('data/extra_shap_samples_{}.csv'.format(celltype))
    # x_valid = df_str_to_encoding(df_extra)
    # x_valid = np.array(x_valid)
    # x_valid = torch.from_numpy(x_valid).float()
    # fastshap_values = fastshap.shap_values(x_valid)[:, :, 0]
    # end_time = time.time()
    # elapsed_time = (end_time - start_time) / len(df)
    # np.save('shap_results/time_fastshap_inference_extra_{}_{}.npy'.format(celltype, property), elapsed_time)
    # np.save('shap_results/fastshap_extra_{}_{}.npy'.format(celltype, property), fastshap_values)

    # start_time = time.time()
    # df_extra = pd.read_csv('data/extra_shap_samples_{}_x2.csv'.format(celltype))
    # x_valid = df_str_to_encoding(df_extra)
    # x_valid = np.array(x_valid)
    # x_valid = torch.from_numpy(x_valid).float()
    # fastshap_values = fastshap.shap_values(x_valid)[:, :, 0]
    # end_time = time.time()
    # elapsed_time = (end_time - start_time) / len(df)
    # np.save('shap_results/time_fastshap_inference_extra_{}_{}_x2.npy'.format(celltype, property), elapsed_time)
    # np.save('shap_results/fastshap_extra_{}_{}_x2.npy'.format(celltype, property), fastshap_values)

