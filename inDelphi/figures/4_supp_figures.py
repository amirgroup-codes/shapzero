import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
np.random.seed(42)
from scipy.stats import pearsonr
import random
random.seed(42)

import sys
sys.path.append('../..')
from gen.shapzero import plot_shap_values, plot_interactions_summary, top_shap_values, top_interactions, plot_interactions, correlation_shap_values
from gen.utils import  plot_time_complexity

# Import font
import matplotlib.font_manager as fm
font_path = '/usr/scratch/dtsui/Helvetica.ttf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
font_name = font_prop.get_name()
plt.rcParams['font.family'] = font_name



"""
Plotting parameters
"""
from matplotlib import gridspec
font_size = 6
markersize = 2
complexity_markersize = 3
legend_marker_size = 3
top_values_shap = 50
mm = 1/25.4  # mm in inches
plt.rcParams['font.size'] = font_size
linewidth = 0.25
plt.rcParams['axes.linewidth'] = linewidth
overall_fig_width = 183 # Measurements in mm
time_complexity_width = 40.26



"""
FastSHAP
"""
celltype = 'HEK293'
property = 'frameshift'
df_heldout = pd.read_csv(f'data/{celltype}_heldout.csv')
fig = plt.figure(figsize=(overall_fig_width*mm, 50*mm), constrained_layout=True) 
gs = gridspec.GridSpec(1, 2, height_ratios=[2], width_ratios=[4, 2], figure=fig)
seq_length = 40
def str2numpy(string):
    string = string.strip('[]')
    str_list = string.split(',')
    float_list = [float(num) for num in str_list]
    arr = np.array(float_list)
    return arr
# Add extra SHAP samples + their times
df_extra = pd.read_csv(f'data/extra_shap_samples_{celltype}.csv')
df_extra_2 = pd.read_csv(f'data/extra_shap_samples_{celltype}_x2.csv')



# FastSHAP
shap = np.load(f'shap_results/fastshap_{celltype}_{property}.npy')
extra_shap = np.load(f'shap_results/fastshap_extra_{celltype}_{property}.npy')
extra_shap_x2 = np.load(f'shap_results/fastshap_extra_{celltype}_{property}_x2.npy')
shap = np.concatenate((shap, extra_shap, extra_shap_x2), axis=0)
random_indices = random.sample(range(0, len(shap)), top_values_shap)
shap_values = np.zeros((len(shap), seq_length))
for i, shap_val in enumerate(shap):
    for pos in range(seq_length): 
        shap_values[i, pos] = shap_val[pos]

ax1 = fig.add_subplot(gs[0, 0])
sequences = df_heldout['Left_seq'].values + df_heldout['Right_seq'].values
sequences_extra = df_extra['Left_seq'].values + df_extra['Right_seq'].values
sequences_extra_2 = df_extra_2['Left_seq'].values + df_extra_2['Right_seq'].values
sequences = np.concatenate([sequences, sequences_extra, sequences_extra_2])
sequences_trimmed = sequences[random_indices]
shap_values_trimmed = [shap_values[i,:] for i in random_indices]
plot_shap_values(ax1, sequences_trimmed, shap_values_trimmed, x_label='Target sequence position', y_label='SHAP value \n using FastSHAP', font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth, y_limits=(np.min(shap_values_trimmed), np.max(shap_values_trimmed) + 0.3))
top_shap_values(sequences, shap_values, top_values=20, all_values_filename=f'shap_results/inDelphi_fastshap_values')

# Correlation between KernelSHAP
kernelshap = np.load(f'shap_results/shap_values_{celltype}.npy')[0, :, :]
kernelshap_extra = np.load(f'shap_results/shap_values_extra_{celltype}.npy')[0, :, :]
kernelshap_extra_x2 = np.load(f'shap_results/shap_values_extra_{celltype}_x2.npy')[0, :, :] 
kernelshap = np.concatenate((kernelshap, kernelshap_extra, kernelshap_extra_x2), axis=0)
correlation_shap_values(kernelshap, shap_values)



# Time complexity 
# SHAP zero
num_samples = range(1, 500)
shapzero_sample_time = np.load(f'shap_results/time_per_sample_{celltype}.npy') 
shapzero_sample_time = shapzero_sample_time * 3981312 # Number of samples needed for q-sft
shapzero_shap_time = (np.load(f'shap_results/time_shapzero_values_{celltype}_{property}.npy') * len(df_heldout) + np.load(f'shap_results/time_shapzero_values_extra_{celltype}_{property}.npy') * len(df_extra) +  np.load(f'shap_results/time_shapzero_values_extra_{celltype}_{property}_x2.npy') * len(df_extra_2)) / (len(df_heldout) + len(df_extra) + len(df_extra_2))
overall_shapzero_time = [shapzero_sample_time + shapzero_shap_time * i for i in num_samples]

# FastSHAP
fastshap_surrogate_time = np.load(f'shap_results/time_fastshap_surrogate_{celltype}_{property}.npy')
fastshap_explainer_time = np.load(f'shap_results/time_fastshap_explainer_{celltype}_{property}.npy')
fastshap_inference_time = (np.load(f'shap_results/time_fastshap_inference_{celltype}_{property}.npy') * len(df_heldout) + np.load(f'shap_results/time_fastshap_inference_extra_{celltype}_{property}.npy') * len(df_extra) + np.load(f'shap_results/time_fastshap_inference_extra_{celltype}_{property}_x2.npy') * len(df_extra_2)) / (len(df_heldout) + len(df_extra) + len(df_extra_2))
overall_fastshap_time = [fastshap_surrogate_time + fastshap_explainer_time + (fastshap_inference_time / len(df_heldout)) * i for i in num_samples]

ax2 = fig.add_subplot(gs[0, 1])
plot_time_complexity(ax2, overall_fastshap_time, overall_shapzero_time, font_size=font_size, linewidth=linewidth, markersize=complexity_markersize, tot_samples=len(df_heldout) + len(df_extra) + len(df_extra_2), offset_intersection_text=70, legend_loc='upper right', x_label='Number of sequences explained', first_shap_method='FastSHAP') 


plt.savefig('shap_results/SUPP_inDelphi_altshap.pdf', transparent=True, dpi=300)
plt.show()



"""
Interaction plots
"""
fig = plt.figure(figsize=(overall_fig_width*mm, 50*mm), constrained_layout=True) 
gs = gridspec.GridSpec(1, 2, height_ratios=[1], width_ratios=[1, 1], figure=fig)

# [0,1]
ax2 = fig.add_subplot(gs[0, 1])
x_valid = np.load(f'shap_results/fsi_{celltype}_{property}_samples.npy')
encoding = {0:'A', 1:'C', 2:'T', 3:'G'}
sequences = [''.join(encoding[num] for num in row) for row in x_valid]
shap_interactions = np.load(f'shap_results/fsi_{celltype}_{property}.pickle', allow_pickle=True)
plot_interactions(ax2, sequences, shap_interactions, top_values=80, x_label='Target sequence position', y_label='Faith-Shap interaction \n using SHAP-IQ', font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth)
bounds = ax2.get_ylim()

# [0,0]
ax1 = fig.add_subplot(gs[0, 0])
shapzero_interactions = np.load(f'shap_results/shapzero_fsi_{celltype}_{property}.npz', allow_pickle=True)
extra_shapzero_interactions = np.load(f'shap_results/shapzero_fsi_extra_{celltype}_{property}.npz', allow_pickle=True)
extra_shapzero_interactions_x2 = np.load(f'shap_results/shapzero_fsi_extra_{celltype}_{property}_x2.npz', allow_pickle=True)
shapzero_interactions_sequences = shapzero_interactions['interactions_sequences']
shapzero_interactions_sequences_extra = extra_shapzero_interactions['interactions_sequences']
shapzero_interactions_sequences_extra_x2 = extra_shapzero_interactions_x2['interactions_sequences']
shapzero_interactions_sequences = np.concatenate([shapzero_interactions_sequences, shapzero_interactions_sequences_extra, shapzero_interactions_sequences_extra_x2])
shapzero_sequences = shapzero_interactions['sequences']
shapzero_equences_extra = extra_shapzero_interactions['sequences']
shapzero_equences_extra_x2 = extra_shapzero_interactions_x2['sequences']
shapzero_sequences = np.concatenate([shapzero_sequences, shapzero_equences_extra, shapzero_equences_extra_x2])
sequences = [''.join(encoding[num] for num in row) for row in shapzero_sequences]
plot_interactions(ax1, sequences, shapzero_interactions_sequences, top_values=80, x_label='Target sequence position', y_label='Faith-Shap interaction \n using SHAP zero', y_limits=bounds, font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth)
plt.savefig('shap_results/SUPP_inDelphi_interactions.pdf', transparent=True, dpi=300)
plt.show()