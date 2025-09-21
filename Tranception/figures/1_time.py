import numpy as np
import sys
import pandas as pd
sys.path.append('..')
sys.path.append('../..')
import pickle
import time
from tqdm import tqdm
import os
from src.utils import ModelScorer
import torch



"""
Time Tranception speeds
"""
properties = ['GFP_Tranception']
file_path = ['GFP_Tranception']

# Since we don't have ground truth data, time Trancepton on the heldout data
DMS='GFP_AEQVI_Sarkisyan_2016' 
DMS_list = pd.read_csv('../data/ProteinGym_reference_file_substitutions.csv')
DMS_list = DMS_list[DMS_list['DMS_id'] == DMS]
sequence = DMS_list["target_seq"].values[0].upper()
positions = [83, 87, 82, 84, 43, 164, 47, 162, 140, 165] # 0-indexed
scorer = ModelScorer(sequence, positions, DMS, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), property=property)

for property in properties:
   
    all_scores = []
    data_heldout = np.load(f'data/{property}_heldout.npy')
    all_samples = np.zeros((data_heldout.shape[0], len(file_path)))
    scores = np.array([])
    start_time = time.time()
    scores = scorer.compute_scores_from_numpy_arr(data_heldout)
    end_time = time.time()  
    total_time = end_time - start_time
    average_time_per_sample = total_time / data_heldout.shape[0]
    print("Total time:", total_time)
    print("Average time per sample:", average_time_per_sample)
    np.save('shap_results/time_per_sample_{}.npy'.format(property), average_time_per_sample) # Use this for time complexity of SHAP Zero