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
from gen.shapzero import plot_shap_values, plot_interactions_summary, top_shap_values, top_interactions, correlation_shap_values, correlation_interactions
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
property = 'on_target'
plt.rcParams['font.size'] = font_size
linewidth = 0.25
plt.rcParams['axes.linewidth'] = linewidth
overall_fig_width = 183 # Measurements in mm
time_complexity_width = 40.26



"""
[1,0]
"""
# Make time complexity plot first
df_heldout = pd.read_csv(f'data/{property}_heldout.csv')
fig = plt.figure(figsize=(time_complexity_width*mm, 90*mm), constrained_layout=True) 
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], width_ratios=[2], figure=fig)

ax2 = fig.add_subplot(gs[1, 0])
num_samples = range(1, 1100)
shapzero_sample_time = np.load(f'shap_results/time_per_sample_{property}.npy') 
shapzero_sample_time = shapzero_sample_time * 3981312 # Number of samples needed for q-sft
shapzero_shap_time = np.load(f'shap_results/time_shapzero_fsi_{property}.npy')
overall_shapzero_time = [shapzero_sample_time + shapzero_shap_time * i for i in num_samples]
shap_time = np.load(f'shap_results/time_fsi_{property}.npy')
overall_shap_time = [shap_time * i for i in num_samples]
print('Amortized inference time:')
print(f'Interactions - SHAP zero: {overall_shapzero_time[len(df_heldout)]/len(df_heldout)}, SHAP-IQ: {overall_shap_time[len(df_heldout)]/len(df_heldout)}')
print(f'SHAP zero is {overall_shap_time[len(df_heldout)]/overall_shapzero_time[len(df_heldout)]}x faster than SHAP-IQ')
plot_time_complexity(ax2, overall_shap_time, overall_shapzero_time, font_size=font_size, linewidth=linewidth, markersize=complexity_markersize, tot_samples=len(df_heldout), first_shap_method='SHAP-IQ', offset_intersection_text=25, x_label='Number of sequences explained')
bounds = ax2.get_ylim()



"""
[0,0]
"""
ax1 = fig.add_subplot(gs[0, 0])
shap_time = np.load(f'shap_results/time_shap_values_{property}.npy')
overall_shap_time = [shap_time * i for i in num_samples]
shapzero_shap_time = np.load(f'shap_results/time_shapzero_values_{property}.npy')
overall_shapzero_time = [shapzero_sample_time + shapzero_shap_time * i for i in num_samples]
print(f'SHAP Values - SHAP zero: {overall_shapzero_time[len(df_heldout)]/len(df_heldout)}, KernelSHAP: {overall_shap_time[len(df_heldout)]/len(df_heldout)}')
print(f'SHAP zero is {overall_shap_time[len(df_heldout)]/overall_shapzero_time[len(df_heldout)]}x faster than KernelSHAP')
plot_time_complexity(ax1, overall_shap_time, overall_shapzero_time, font_size=font_size, linewidth=linewidth, markersize=complexity_markersize, tot_samples=len(df_heldout), y_limits=bounds, x_label='', offset_intersection_text=70)
plt.savefig('shap_results/tiger_time_main.pdf', transparent=True, dpi=300)
plt.show()



"""
[0,1]
"""
# Make SHAP plots
fig = plt.figure(figsize=((overall_fig_width - time_complexity_width)*mm, 63*mm), constrained_layout=True) 
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[4, 4], figure=fig)
shap = np.load(f'shap_results/shap_values_{property}.npy')
random_indices = random.sample(range(0, len(shap)), top_values_shap)

seq_length = 26
shap_values = np.zeros((len(shap), seq_length))
for i, shap_val in enumerate(shap):
    # Loop through all positions in the sequence and calculate the SHAP value for each position for each guide:target pairing
    for pos in range(seq_length): 
        shap_values[i, pos] = shap_val[pos]

ax2 = fig.add_subplot(gs[0, 1])
sequences = df_heldout['Full Sequence'].values
sequences_trimmed = sequences[random_indices]
shap_values_trimmed = [shap_values[i,:] for i in random_indices]
plot_shap_values(ax2, sequences_trimmed, shap_values_trimmed, x_label=None, y_label='SHAP value \n using KernelSHAP', font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth)
top_shap_values(sequences, shap_values, top_values=20, top_values_filename=f'shap_results/{property}_top_kernelshap_values', all_values_filename=f'shap_results/TIGER_kernelshap_values')
bounds = ax2.get_ylim()



"""
[0,0]
"""
ax1 = fig.add_subplot(gs[0, 0])
shap_zero = np.load(f'shap_results/shapzero_values_{property}.npy')
shap_zero_trimmed = [shap_zero[i,:] for i in random_indices]
plot_shap_values(ax1, sequences_trimmed, shap_zero_trimmed, x_label=None, y_label='SHAP value \n using SHAP zero', y_limits=bounds, font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth)
top_shap_values(sequences, shap_zero, top_values=20, top_values_filename=f'shap_results/{property}_top_shapzero_values', all_values_filename=f'shap_results/TIGER_shapzero_values')
correlation_shap_values(shap_zero, shap)

# Calculate what % of top positive interactions are C-G and in the seed region
top_vals = pd.read_csv(f'shap_results/{property}_top_shapzero_values.csv')
pos_df = top_vals[top_vals['Sign'] == 'Positive']
ratio = pos_df[(pos_df['Position'].between(6, 12)) & (pos_df['Feature'].isin(['C', 'G']))]['Average value'].abs().sum() / pos_df['Average value'].abs().sum()
print('Percentage of top C, G positive interactions in the seed region for SHAP zero:', ratio*100)
top_vals = pd.read_csv(f'shap_results/{property}_top_kernelshap_values.csv')
pos_df = top_vals[top_vals['Sign'] == 'Positive']
ratio = pos_df[(pos_df['Position'].between(6, 12)) & (pos_df['Feature'].isin(['C', 'G']))]['Average value'].abs().sum() / pos_df['Average value'].abs().sum()
print('Percentage of top C, G positive interactions in the seed region for KernelSHAP:', ratio*100)



"""
[1,0]
"""
ax3 = fig.add_subplot(gs[1, 0])
shapzero_interactions = np.load(f'shap_results/shapzero_fsi_{property}.npz', allow_pickle=True)
shapzero_interactions_sequences = shapzero_interactions['interactions_sequences']
shapzero_sequences = shapzero_interactions['sequences']
encoding = {0:'A', 1:'C', 2:'T', 3:'G'}
shapzero_sequences = [''.join(encoding[num] for num in row) for row in shapzero_sequences]
print('SHAP zero:')
plot_interactions_summary(ax3, shapzero_sequences, shapzero_interactions_sequences,  x_label='Target sequence position', y_label='Faith-Shap interaction \n using SHAP zero', font_size=font_size, legend=False, linewidth=linewidth, print_ratio='tiger')
top_interactions(shapzero_sequences, shapzero_interactions_sequences, top_values=20, top_interactions_filename=f'shap_results/{property}_top_shapzero_fsi', all_interactions_filename=f'shap_results/TIGER_shapzero_fsi')

# Calculate what % of top positive interactions have an A 
top_vals = pd.read_csv(f'shap_results/{property}_top_shapzero_fsi.csv')
pos_df = top_vals[top_vals['Sign'] == 'Positive']
ratio = pos_df[pos_df['Position'].str.contains(r'\b(?:6|7|8|9|10|11|12)\b') & pos_df['Feature'].str.contains('A') & pos_df['Feature'].str.contains('C|G')]['Average value'].abs().sum() / pos_df['Average value'].abs().sum()
print('Percentage of top A positive interactions for SHAP zero:', ratio*100)



"""
[1,1]
"""
ax4 = fig.add_subplot(gs[1, 1])
x_valid = np.load(f'shap_results/fsi_{property}_samples.npy')
sequences = [''.join(encoding[num] for num in row) for row in x_valid]
shap_interactions = np.load(f'shap_results/fsi_{property}.pickle', allow_pickle=True)
print('SHAP-IQ:')
plot_interactions_summary(ax4, sequences, shap_interactions,  x_label='Target sequence position', y_label='Faith-Shap interaction \n using SHAP-IQ', font_size=font_size, legend=False, linewidth=linewidth, print_ratio='tiger')
correlation_interactions(shapzero_interactions_sequences, shapzero_sequences, shap_interactions, sequences)
top_interactions(sequences, shap_interactions, top_values=20, top_interactions_filename=f'shap_results/{property}_top_shapiq_fsi', all_interactions_filename=f'shap_results/TIGER_shapiq_fsi')
plt.savefig('shap_results/tiger_shap_main.pdf', transparent=True, dpi=300)
plt.show()

