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
from gen.shapzero import plot_shap_values, plot_interactions_summary, top_shap_values, top_interactions, correlation_shap_values, correlation_interactions
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
property = 'frameshift'
celltype = 'HEK293'
plt.rcParams['font.size'] = font_size
linewidth = 0.25
plt.rcParams['axes.linewidth'] = linewidth
overall_fig_width = 183 # Measurements in mm
time_complexity_width = 40.26



"""
[1,0]
"""
# Make time complexity plot first
df_heldout = pd.read_csv(f'data/{celltype}_heldout.csv')
fig = plt.figure(figsize=(time_complexity_width*mm, 90*mm), constrained_layout=True) 
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], width_ratios=[2], figure=fig)

ax2 = fig.add_subplot(gs[1, 0])
num_samples = range(1, 500)
shapzero_sample_time = np.load(f'shap_results/time_per_sample_{celltype}.npy') 
shapzero_sample_time = shapzero_sample_time * 6045696 # Number of samples needed for q-sft
shapzero_shap_time = np.load(f'shap_results/time_shapzero_fsi_{celltype}_{property}.npy')
overall_shapzero_time = [shapzero_sample_time + shapzero_shap_time * i for i in num_samples]
shap_time = np.load(f'shap_results/time_fsi_{celltype}_{property}.npy')
overall_shap_time = [shap_time * i for i in num_samples]
print('Amortized inference time:')
print(f'Interactions - SHAP zero: {overall_shapzero_time[len(df_heldout)]/len(df_heldout)}, SHAP-IQ: {overall_shap_time[len(df_heldout)]/len(df_heldout)}')
plot_time_complexity(ax2, overall_shap_time, overall_shapzero_time, font_size=font_size, linewidth=linewidth, markersize=complexity_markersize, tot_samples=len(df_heldout), first_shap_method='SHAP-IQ', offset_intersection_text=25)
bounds = ax2.get_ylim()



"""
[0,0]
"""
ax1 = fig.add_subplot(gs[0, 0])
shap_time = np.load(f'shap_results/time_shap_values_{celltype}.npy')
overall_shap_time = [shap_time * i for i in num_samples]
shapzero_shap_time = np.load(f'shap_results/time_shapzero_values_{celltype}_{property}.npy')
overall_shapzero_time = [shapzero_sample_time + shapzero_shap_time * i for i in num_samples]
print(f'SHAP Values - SHAP zero: {overall_shapzero_time[len(df_heldout)]/len(df_heldout)}, KernelSHAP: {overall_shap_time[len(df_heldout)]/len(df_heldout)}')
plot_time_complexity(ax1, overall_shap_time, overall_shapzero_time, font_size=font_size, linewidth=linewidth, markersize=complexity_markersize, tot_samples=len(df_heldout), y_limits=bounds, x_label='', offset_intersection_text=70)
plt.savefig('shap_results/inDelphi_time_main.pdf', transparent=True, dpi=300)
plt.show()
print('test ', shap_time * 84)


"""
[0,1]
"""
# Make SHAP plots
fig = plt.figure(figsize=((overall_fig_width - time_complexity_width)*mm, 63*mm), constrained_layout=True) 
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[4, 4], figure=fig)
shap = np.load(f'shap_results/shap_values_{celltype}.npy')[0, :, :]
random_indices = random.sample(range(0, len(shap)), top_values_shap)

seq_length = 40
shap_values = np.zeros((len(shap), seq_length))
for i, shap_val in enumerate(shap):
    # Loop through all positions in the sequence and calculate the SHAP value for each position for each guide:target pairing
    for pos in range(seq_length): 
        shap_values[i, pos] = shap_val[pos]

ax2 = fig.add_subplot(gs[0, 1])
sequences = df_heldout['Left_seq'].values + df_heldout['Right_seq'].values
sequences_trimmed = sequences[random_indices]
shap_values_trimmed = [shap_values[i,:] for i in random_indices]
plot_shap_values(ax2, sequences_trimmed, shap_values_trimmed, x_label=None, y_label='SHAP value \n using KernelSHAP', font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth)
top_shap_values(sequences, shap_values, top_values=20, filename=f'shap_results/{celltype}_{property}_kernelshap_values')
bounds = ax2.get_ylim()



"""
[0,0]
"""
ax1 = fig.add_subplot(gs[0, 0])
shap_zero = np.load(f'shap_results/shapzero_values_{celltype}_{property}.npy')
shap_zero_trimmed = [shap_zero[i,:] for i in random_indices]
plot_shap_values(ax1, sequences_trimmed, shap_zero_trimmed, x_label=None, y_label='SHAP value \n using SHAP zero', y_limits=bounds, font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth)
top_shap_values(sequences, shap_zero, top_values=20, filename=f'shap_results/{celltype}_{property}_shapzero_values')
correlation_shap_values(shap_zero, shap)

# Calculate what % of top interactions are +/- 3 nts away from the cutsite 
top_vals = pd.read_csv(f'shap_results/{celltype}_{property}_shapzero_values.csv')
ratio = top_vals[top_vals['Position'].between(18, 23)]['Average value'].abs().sum() / top_vals['Average value'].abs().sum()
print('Percentage of top interactions +/- 3 nts away from the cutsite for SHAP zero:', ratio*100)
top_vals = pd.read_csv(f'shap_results/{celltype}_{property}_kernelshap_values.csv')
ratio = top_vals[top_vals['Position'].between(18, 23)]['Average value'].abs().sum() / top_vals['Average value'].abs().sum()
print('Percentage of top interactions +/- 3 nts away from the cutsite for KernelSHAP:', ratio*100)



"""
[1,0]
"""
ax3 = fig.add_subplot(gs[1, 0])
shapzero_interactions = np.load(f'shap_results/shapzero_fsi_{celltype}_{property}.npz', allow_pickle=True)
shapzero_interactions_sequences = shapzero_interactions['interactions_sequences']
shapzero_sequences = shapzero_interactions['sequences']
encoding = {0:'A', 1:'C', 2:'T', 3:'G'}
shapzero_sequences = [''.join(encoding[num] for num in row) for row in shapzero_sequences]
print('SHAP zero:')
plot_interactions_summary(ax3, shapzero_sequences, shapzero_interactions_sequences,  x_label='Target sequence position', y_label='Faith-Shap interaction \n using SHAP zero', font_size=font_size, legend=False, linewidth=linewidth, print_ratio='inDelphi')
top_interactions(sequences, shapzero_interactions_sequences, top_values=20, filename=f'shap_results/{celltype}_{property}_shapzero_fsi')

# Calculate what % of top interactions contain C/G and A with C/G
top_vals = pd.read_csv(f'shap_results/{celltype}_{property}_shapzero_fsi.csv')
total_magnitude = top_vals['Average value'].abs().sum()
sum_CG = top_vals[top_vals['Feature'].str.contains('C|G')]['Average value'].abs().sum()
sum_A_with_CG = top_vals[top_vals['Feature'].str.contains('A') & top_vals['Feature'].str.contains('C|G')]['Average value'].abs().sum()
ratio_CG = sum_CG / total_magnitude
ratio_A_with_CG = sum_A_with_CG / total_magnitude
print('Percentage of top interactions with C or G:', ratio_CG*100)
print('Percentage of top interactions with A and C/G:', ratio_A_with_CG*100)



"""
[1,1]
"""
ax4 = fig.add_subplot(gs[1, 1])
x_valid = np.load(f'shap_results/fsi_{celltype}_{property}_samples.npy')
sequences = [''.join(encoding[num] for num in row) for row in x_valid]
shap_interactions = np.load(f'shap_results/fsi_{celltype}_{property}.pickle', allow_pickle=True)
print('SHAP-IQ:')
plot_interactions_summary(ax4, sequences, shap_interactions,  x_label='Target sequence position', y_label='Faith-Shap interaction \n using SHAP-IQ', font_size=font_size, legend=False, linewidth=linewidth, print_ratio='inDelphi')
correlation_interactions(shapzero_interactions_sequences, shapzero_sequences, shap_interactions, sequences)
plt.savefig('shap_results/inDelphi_shap_main.pdf', transparent=True, dpi=300)
plt.show()

