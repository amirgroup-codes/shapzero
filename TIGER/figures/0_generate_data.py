"""
Script to preprocess the heldout sequences and generate the training and validation sets. Files are saved in the data folder.
Use the fshap-codes environment
"""
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('../tiger')
from data import load_data, label_and_filter_data, training_validation_split_targets
from normalization import get_normalization_object
import tensorflow as tf
tf.keras.utils.set_random_seed(12345)
from tiger_trainer import score_tiger, calibrate_tiger
NORMALIZATION = 'UnitInterval'
NORMALIZATION_KWARGS = dict(q_neg=1, q_pos=99, squash=False)



"""
Create perfect match heldout dataset from the main text, and 1 basepair off dataset from the supplementary text.
Create a CSV with the experimental scores, the full sequence, and the n=40 sequence.
"""
curr_dir = os.getcwd()
data = load_data(curr_dir + '/../tiger/data-processed/off-target')
data = label_and_filter_data(*data, method='NoFilter')
data = training_validation_split_targets(data, train_ratio=0.9)
normalizer = get_normalization_object(NORMALIZATION)(data, **NORMALIZATION_KWARGS)
data = normalizer.normalize_targets(data)
data.head()

n = 26
target_length = 23
properties = ['on_target', 'titration']
splits = ['training', 'validation']

# Split data into training and validation
for split in splits:
    data_hold = data.copy()
    data_split = data_hold[data_hold.fold == split]#data_hold.copy()

    # Split data into PM and SM
    for property in properties:
        file_path = property
        guide_type = property

        # Keep only either PM or SM's
        data_split_i = data_split.copy()
        data_split_i = data_split_i[data_split_i['guide_type'] == guide_type]
        data_split_i['unique ID'] = data_split_i['guide_id'].str.split(':').str[0]
        if guide_type == 'SM':
            # Set it to be the mean of all the guide scores
            observed_lfc_mean = data_split_i.groupby('unique ID')['observed_lfc'].mean().reset_index()
            data_merged = data_split_i.drop(columns='observed_lfc').merge(observed_lfc_mean, on='unique ID')
            data_unique_genes = data_merged.drop_duplicates(subset='unique ID')
            data_split_i = data_unique_genes.copy()

        # Keep only the first 3 nt of the 5' context to make a 26 nt sequence
        data_split_i['Full Sequence'] = data_split_i['5p_context'].str[-3:] + data_split_i['target_seq'] 
        data_split_i.to_csv(f'data/experimental_{split}_{guide_type}.csv')