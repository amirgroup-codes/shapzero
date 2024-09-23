"""
Script to compute scores for inDelphi
Note: Kept separate because inDelphi is run on an older environment
"""
import numpy as np
import sys
import time
import pandas as pd
sys.path.append('..')
from src.utils import get_indelphi_from_numpy_arr, get_indelphi_from_str
from scipy.stats import pearsonr, spearmanr



"""
HEK293 for frameshift and 1-bp-ins
"""
q = 4
n = 40
b = 7
celltype = 'HEK293'
properties = ['Frameshift frequency', 'Highest 1-bp insertion']
df_heldout = pd.read_csv('data/{}_heldout.csv'.format(celltype))
left_seq = df_heldout['Left_seq']
right_seq = df_heldout['Right_seq']
seqs = left_seq + right_seq
seqs = seqs.tolist()



# inDelphi
start_time = time.time()
all_samples = get_indelphi_from_str(celltype, seqs, properties)
print('inDelphi')
df = pd.read_csv('correlation_results/model_scores_{}.csv'.format(celltype))
df_corrs = pd.read_csv('correlation_results/model_results_{}.csv'.format(celltype))

# Save each property's results
end_time = time.time()
execution_time = (end_time - start_time) / len(seqs)
np.save('shap_results/time_per_sample_{}.npy'.format(celltype), execution_time) # Use this for time complexity of SHAP Zero
for i, property in enumerate(properties):
    y_pred = all_samples[:, i]
    pearson_corr, _ = pearsonr(df_heldout[property], y_pred)
    spearman_corr, _ = spearmanr(df_heldout[property], y_pred)
    print('{} - Pearson correlation: {:.2f}, Spearman correlation: {:.2f}'.format(property, pearson_corr, spearman_corr))
    df['Model: {}'.format(property)] = y_pred
    df_corrs['Model: {}'.format(property)] = [pearson_corr, spearman_corr]

df = df.drop(columns=['Unnamed: 0'])
df.to_csv('correlation_results/model_scores_{}.csv'.format(celltype))
df_corrs.to_csv('correlation_results/model_results_{}.csv'.format(celltype))

