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
from gen.shapzero import shapzero
from gen.utils import load_data
aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
num_aas = len(aa_dict)

DMS='GFP_AEQVI_Sarkisyan_2016' 
DMS_list = pd.read_csv('../data/ProteinGym_reference_file_substitutions.csv')
DMS_list = DMS_list[DMS_list['DMS_id'] == DMS]
sequence = DMS_list["target_seq"].values[0].upper()
positions = [83, 87, 82, 84, 43, 164, 47, 162, 140, 165] # 0-indexed
scorer = ModelScorer(sequence, positions, DMS, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), property=property)
gwht = np.load('/usr/scratch/dtsui/FinalizedCodes/SHAPzero/Tranception/results/q20_n10_b3/qsft_transform_GFP_Tranception.pickle', allow_pickle=True)
positive_interactions = [ '4,6 R,Y', '4,6 K,W', '5,9 K,P', '5,9 D,W', '2,3 F,K']
negative_interactions = [ '4,6 K,E', '2,3 F,W', '4,6 G,F', '5,9 K,K', '7,9 V,P']
shap_zero = shapzero(gwht, q=20, n=10)
top_seqs = 3

samples = np.load('data/GFP_Tranception_heldout.npy')
results_data = []
all_interactions_to_process = [('Positive', positive_interactions), ('Negative', negative_interactions)]

# 1. Loop through Positive then Negative interaction lists
for sign, interaction_list in all_interactions_to_process:
    for interaction_str in interaction_list:
        pos_str, aas_str = interaction_str.split(' ')
        p1_str, p2_str = pos_str.split(',')
        idx1, idx2 = int(p1_str), int(p2_str)
        mut_code1, mut_code2 = aa_dict[aas_str.split(',')[0]], aa_dict[aas_str.split(',')[1]]

        mask = (samples[:, idx1] == mut_code1) & (samples[:, idx2] == mut_code2)
        background_samples = samples[mask]
        num_found = len(background_samples)
        
        print(f"\nAnalyzing '{sign}' interaction '{interaction_str}': Found {num_found} matching backgrounds.")
        if num_found == 0:
            continue

        interaction_epistasis_results, interaction_effect1_results, interaction_effect2_results = [], [], []

        # 2. Loop through each of the found background sequences
        for base_sequence in background_samples:
            localized_mobius_tf, localized_mobius_tf_aa = shap_zero.localize_sample(base_sequence)
            
            qualifying_aa_entries = []
            for (key_tf, value_tf), (key_aa, value_aa) in zip(localized_mobius_tf.items(), localized_mobius_tf_aa.items()):
                # Conditional check based on the sign
                is_positive_match = (sign == 'Positive' and value_tf.real > 0)
                is_negative_match = (sign == 'Negative' and value_tf.real < 0)

                if sum(1 for x in key_tf if x != 0) == 2 and key_tf[idx1] != 0 and key_tf[idx2] != 0 and (is_positive_match or is_negative_match):
                    qualifying_aa_entries.append((value_aa.real, key_aa))

            # Sort to get the highest magnitude (most positive or most negative)
            qualifying_aa_entries.sort(key=lambda x: x[0], reverse=(sign == 'Positive'))
            top_candidates = qualifying_aa_entries[:top_seqs]
            if not top_candidates: continue

            batch_for_scoring = []
            for _, mutant_seq_tuple in top_candidates:
                mutant_seq = np.array(mutant_seq_tuple)
                diff_indices = np.where(mutant_seq != base_sequence)[0]
                if len(diff_indices) != 2: continue
                d1, d2 = diff_indices
                seq_A = base_sequence.copy(); seq_A[d1] = mutant_seq[d1]
                seq_B = base_sequence.copy(); seq_B[d2] = mutant_seq[d2]
                batch_for_scoring.extend([base_sequence, seq_A, seq_B, mutant_seq])
            
            if not batch_for_scoring: continue

            scoring_batch = np.stack(batch_for_scoring)
            all_scores = scorer.compute_scores_from_numpy_arr(scoring_batch)
            structured_scores = all_scores.reshape(-1, 4)
            
            effects_1 = structured_scores[:, 1] - structured_scores[:, 0]
            effects_2 = structured_scores[:, 2] - structured_scores[:, 0]
            epistasis_values = structured_scores[:, 3] - structured_scores[:, 1] - structured_scores[:, 2] + structured_scores[:, 0]
            
            interaction_epistasis_results.append(np.mean(epistasis_values))
            interaction_effect1_results.append(np.mean(effects_1))
            interaction_effect2_results.append(np.mean(effects_2))

        # 3. After processing all backgrounds, compute the final aggregated stats
        if not interaction_epistasis_results:
            print(f"  No qualifying candidates found for any background sequence in '{interaction_str}'.")
            continue

        results_data.append({
            'Sign': sign,
            'background_interaction': interaction_str,
            'num_backgrounds_found': num_found,
            'mean_epistasis': np.mean(interaction_epistasis_results),
            'std_epistasis': np.std(interaction_epistasis_results),
            'mean_effect_1': np.mean(interaction_effect1_results),
            'std_effect_1': np.std(interaction_effect1_results),
            'mean_effect_2': np.mean(interaction_effect2_results),
            'std_effect_2': np.std(interaction_effect2_results),
        })

# 4. Save final results to a CSV file
results_df = pd.DataFrame(results_data)
output_filename = 'epistasis_analysis.csv'
results_df.to_csv(output_filename, index=False)

print("\nFinal Aggregated Results:")
print(results_df.round(6))
