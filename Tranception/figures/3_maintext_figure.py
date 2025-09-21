import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
np.random.seed(42)
import random
random.seed(42)

import sys
sys.path.append('../..')
from gen.shapzero import top_shap_values, top_interactions, correlation_shap_values, correlation_interactions, plot_shap_values_protein_top_mutations, plot_pairwise_interaction_heatmap
from gen.utils import  plot_time_complexity

# Import font
import matplotlib.font_manager as fm
font_path = '../../gen/Helvetica.ttf'
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
property = 'GFP_Tranception'
plt.rcParams['font.size'] = font_size
linewidth = 0.25
plt.rcParams['axes.linewidth'] = linewidth
overall_fig_width = 150
time_complexity_width = 45
aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
rev_aa_dict = {v: k for k, v in aa_dict.items()}



"""
[1,0]
"""
# Make time complexity plot first
data_heldout = np.load(f'data/{property}_heldout.npy')
fig = plt.figure(figsize=(time_complexity_width*mm, 60*mm), constrained_layout=True) 
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], width_ratios=[2], figure=fig)

ax2 = fig.add_subplot(gs[1, 0])
num_samples = range(1, 210)
shapzero_sample_time = np.load(f'shap_results/time_per_sample_{property}.npy') 
shapzero_sample_time = shapzero_sample_time * 792000 # Number of samples needed for q-sft
shapzero_shap_time = np.load(f'shap_results/time_shapzero_fsi_{property}.npy')
overall_shapzero_time = [shapzero_sample_time + shapzero_shap_time * i for i in num_samples]
shap_time = np.load(f'shap_results/time_fsi_{property}.npy')
overall_shap_time = [shap_time * i for i in num_samples]
print('Amortized inference time:')
print(f'Interactions - SHAP zero: {overall_shapzero_time[data_heldout.shape[0]]/data_heldout.shape[0]}, SHAP-IQ: {overall_shap_time[data_heldout.shape[0]]/data_heldout.shape[0]}')
print(f'SHAP zero is {overall_shap_time[data_heldout.shape[0]]/overall_shapzero_time[data_heldout.shape[0]]}x faster than SHAP-IQ')
plot_time_complexity(ax2, overall_shap_time, overall_shapzero_time, font_size=font_size, linewidth=linewidth, markersize=complexity_markersize, tot_samples=data_heldout.shape[0], first_shap_method='SHAP-IQ', offset_intersection_text=35, x_label='Number of sequences explained', legend_loc='lower right')
bounds = ax2.get_ylim()



"""
[0,0]
"""
ax1 = fig.add_subplot(gs[0, 0])
shap_time = np.load(f'shap_results/time_shap_values_{property}.npy')
overall_shap_time = [shap_time * i for i in num_samples]
shapzero_shap_time = np.load(f'shap_results/time_shapzero_values_{property}.npy')
overall_shapzero_time = [shapzero_sample_time + shapzero_shap_time * i for i in num_samples]
print(f'SHAP Values - SHAP zero: {overall_shapzero_time[data_heldout.shape[0]]/data_heldout.shape[0]}, KernelSHAP: {overall_shap_time[data_heldout.shape[0]]/data_heldout.shape[0]}')
print(f'SHAP zero is {overall_shap_time[data_heldout.shape[0]]/overall_shapzero_time[data_heldout.shape[0]]}x faster than KernelSHAP')
plot_time_complexity(ax1, overall_shap_time, overall_shapzero_time, font_size=font_size, linewidth=linewidth, markersize=complexity_markersize, tot_samples=data_heldout.shape[0], y_limits=bounds, x_label='', offset_intersection_text=35, legend_loc='lower right')
plt.savefig('shap_results/Tranception_time_main.pdf', transparent=True, dpi=300)
plt.show()



"""
[0,1]
"""
# Make SHAP plots
fig = plt.figure(figsize=((overall_fig_width - time_complexity_width)*mm, 60*mm), constrained_layout=True) 
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[1, 1], figure=fig)
shap_values = np.load(f'shap_results/shap_values_{property}.npy')
random_indices = random.sample(range(0, shap_values.shape[0]), top_values_shap)
ax2 = fig.add_subplot(gs[0, 1])
sequences = [''.join(rev_aa_dict[num] for num in row) for row in data_heldout]
plot_shap_values_protein_top_mutations(ax2, sequences, shap_values, colorbar_label='KernelSHAP \n value')
top_shap_values(sequences, shap_values, top_values=20, top_values_filename=f'shap_results/{property}_top_kernelshap_values', all_values_filename=f'shap_results/Tranception_kernelshap_values')



"""
[0,0]
"""
ax1 = fig.add_subplot(gs[0, 0])
shap_zero = np.load(f'shap_results/shapzero_values_{property}.npy')
plot_shap_values_protein_top_mutations(ax1, sequences, shap_zero, colorbar_label='SHAP zero \n value')
top_shap_values(sequences, shap_zero, top_values=20, top_values_filename=f'shap_results/{property}_top_shapzero_values', all_values_filename=f'shap_results/Tranception_shapzero_values')
correlation_shap_values(shap_zero, shap_values)



"""
[1,0]
"""
ax3 = fig.add_subplot(gs[1, 0])
shapzero_interactions = np.load(f'shap_results/shapzero_fsi_{property}.npz', allow_pickle=True)
shapzero_interactions_sequences = shapzero_interactions['interactions_sequences']
shapzero_sequences = shapzero_interactions['sequences']
shapzero_sequences = [''.join(rev_aa_dict[num] for num in row) for row in shapzero_sequences]
print('SHAP zero:')
plot_pairwise_interaction_heatmap(
    ax3,
    shapzero_sequences,         
    shapzero_interactions_sequences, 
    font_size=6,
    colorbar=True,
    colorbar_label='SHAP zero interaction'
)
top_interactions(shapzero_sequences, shapzero_interactions_sequences, top_values=20, top_interactions_filename=f'shap_results/{property}_top_shapzero_fsi', all_interactions_filename=f'shap_results/Tranception_shapzero_fsi')



"""
[1,1]
"""
ax4 = fig.add_subplot(gs[1, 1])
x_valid = np.load(f'data/fsi_{property}_samples.npy')
sequences = [''.join(rev_aa_dict[num] for num in row) for row in x_valid]
shap_interactions = np.load(f'shap_results/fsi_{property}.pickle', allow_pickle=True)
print('SHAP-IQ:')
plot_pairwise_interaction_heatmap(
    ax4,
    sequences,         
    shap_interactions,
    font_size=6,
    colorbar=True,
    colorbar_label='SHAP-IQ interaction'
)
correlation_interactions(shapzero_interactions_sequences, shapzero_sequences, shap_interactions, sequences, encoding=rev_aa_dict)
top_interactions(sequences, shap_interactions, top_values=20, top_interactions_filename=f'shap_results/{property}_top_shapiq_fsi', all_interactions_filename=f'shap_results/Tranception_shapiq_fsi')
plt.savefig('shap_results/Tranception_shap_main.pdf', transparent=True, dpi=300)
plt.show()
