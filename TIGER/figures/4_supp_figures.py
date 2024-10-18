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
from gen.shapzero import plot_shap_values, correlation_shap_values, plot_interactions, top_shap_values
from gen.utils import plot_multiple_time_complexity

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
FastSHAP and DeepSHAP
"""
property = 'on_target'
df_heldout = pd.read_csv(f'data/{property}_heldout.csv')
fig = plt.figure(figsize=(overall_fig_width*mm, 50*mm), constrained_layout=True) 
gs = gridspec.GridSpec(1, 3, height_ratios=[2], width_ratios=[4, 4, 2], figure=fig)
seq_length = 26
def str2numpy(string):
    string = string.strip('[]')
    str_list = string.split(',')
    float_list = [float(num) for num in str_list]
    arr = np.array(float_list)
    return arr



# FastSHAP
shap = np.load(f'shap_results/fastshap_{property}.npy')
random_indices = random.sample(range(0, len(shap)), top_values_shap)
shap_values = np.zeros((len(shap), seq_length))
for i, shap_val in enumerate(shap):
    # Loop through all positions in the sequence and calculate the SHAP value for each position for each guide:target pairing
    for pos in range(seq_length): 
        shap_values[i, pos] = shap_val[pos]

ax1 = fig.add_subplot(gs[0, 0])
sequences = df_heldout['Full Sequence'].values
sequences_trimmed = sequences[random_indices]
shap_values_trimmed = [shap_values[i,:] for i in random_indices]
plot_shap_values(ax1, sequences_trimmed, shap_values_trimmed, x_label='Target sequence position', y_label='SHAP value \n using FastSHAP', font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth, y_limits=(np.min(shap_values_trimmed) - 0.02, np.max(shap_values_trimmed) + 0.02))
top_shap_values(sequences, shap_values, top_values=20, all_values_filename=f'shap_results/TIGER_fastshap_values')
bounds = ax1.get_ylim()

# Correlation between KernelSHAP
kernelshap = np.load(f'shap_results/shap_values_{property}.npy')
correlation_shap_values(kernelshap, shap_values)



# DeepSHAP
# Both the guide and target sequences are used to generate the shap values. Therefore, the SHAP value at each position is the summation of the two SHAP values.
df_shap = pd.read_csv(f'shap_results/deepshap_{property}.csv')
pairings = ['A:T', 'C:G', 'G:C', 'T:A']
shap_values = np.zeros((len(df_shap), seq_length))

for ((i, seq), sequence) in zip(df_shap.iterrows(), sequences): 

    sequence = list(sequence)
    # target:A,guide:A,target:C,guide:C,target:G,guide:G,target:T,guide:T
    # Loop through all positions in the sequence and calculate the SHAP value for each position for each guide:target pairing
    for pos in range(seq_length): # Guide RNA is from 3 to 26
        for j, pair in enumerate(pairings):

            guide, target = pair.split(':')

            guide_shap = str2numpy(seq[f'guide:{guide}'])
            target_shap = str2numpy(seq[f'target:{target}'])

            if sequence[pos] == target:
                # Add the extra 3 bp context length to the shap values
                if pos in (0, 1, 2):
                    shap_values[i, pos] = target_shap[pos]
                else:
                    shap_values[i, pos] = target_shap[pos] + guide_shap[pos]

ax2 = fig.add_subplot(gs[0, 1])
shap_values_trimmed = [shap_values[i,:] for i in random_indices]
plot_shap_values(ax2, sequences_trimmed, shap_values_trimmed, x_label='Target sequence position', y_label='SHAP value \n using DeepSHAP', font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth, y_limits=bounds)
top_shap_values(sequences, shap_values, top_values=20, all_values_filename=f'shap_results/TIGER_deepshap_values')



# Time complexity 
# SHAP zero
num_samples = range(1, 1100)
shapzero_sample_time = np.load(f'shap_results/time_per_sample_{property}.npy') 
shapzero_sample_time = shapzero_sample_time * 3981312 # Number of samples needed for q-sft
shapzero_shap_time = np.load(f'shap_results/time_shapzero_values_{property}.npy')
overall_shapzero_time = [shapzero_sample_time + shapzero_shap_time * i for i in num_samples]

# FastSHAP
fastshap_surrogate_time = np.load(f'shap_results/time_fastshap_surrogate_{property}.npy')
fastshap_explainer_time = np.load(f'shap_results/time_fastshap_explainer_{property}.npy')
fastshap_inference_time = np.load(f'shap_results/time_fastshap_inference_{property}.npy')
overall_fastshap_time = [fastshap_surrogate_time + fastshap_explainer_time + fastshap_inference_time * i for i in num_samples]

# DeepSHAP
deepshap_time = np.load(f'shap_results/time_deepshap_{property}.npy')
overall_deepshap_time = [deepshap_time * i for i in num_samples]

ax3 = fig.add_subplot(gs[0, 2])
plot_multiple_time_complexity(ax3, overall_fastshap_time, overall_shapzero_time, overall_deepshap_time, font_size=font_size, linewidth=linewidth, markersize=complexity_markersize, tot_samples=len(df_heldout), offset_intersection_text=70, x_label='Number of sequences explained')

# Find point of convergence for SHAP zero where SHAP zero > DeepSHAP
num_samples = 1
while True:
    overall_shapzero_time = shapzero_sample_time + shapzero_shap_time * num_samples
    overall_deepshap_time = deepshap_time * num_samples
    if overall_deepshap_time > overall_shapzero_time:
        print(f"At {num_samples} samples, overall_deepshap_time exceeds overall_shapzero_time.")
        break
    num_samples += 1



plt.savefig('shap_results/SUPP_tiger_altshap.pdf', transparent=True, dpi=300)
plt.show()



"""
Interaction plots
"""
fig = plt.figure(figsize=(overall_fig_width*mm, 50*mm), constrained_layout=True) 
gs = gridspec.GridSpec(1, 2, height_ratios=[1], width_ratios=[1, 1], figure=fig)

# [0,0]
ax1 = fig.add_subplot(gs[0, 0])
shapzero_interactions = np.load(f'shap_results/shapzero_fsi_{property}.npz', allow_pickle=True)
shapzero_interactions_sequences = shapzero_interactions['interactions_sequences']
shapzero_sequences = shapzero_interactions['sequences']
encoding = {0:'A', 1:'C', 2:'T', 3:'G'}
sequences = [''.join(encoding[num] for num in row) for row in shapzero_sequences]
plot_interactions(ax1, sequences, shapzero_interactions_sequences, top_values=80, x_label='Target sequence position', y_label='Faith-Shap interaction \n using SHAP zero', font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth)
bounds = ax1.get_ylim()



# [0,1]
ax2 = fig.add_subplot(gs[0, 1])
x_valid = np.load(f'shap_results/fsi_{property}_samples.npy')
sequences = [''.join(encoding[num] for num in row) for row in x_valid]
shap_interactions = np.load(f'shap_results/fsi_{property}.pickle', allow_pickle=True)
plot_interactions(ax2, sequences, shap_interactions, top_values=80, x_label='Target sequence position', y_label='Faith-Shap interaction \n using SHAP-IQ', y_limits=bounds, font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth)
plt.savefig('shap_results/SUPP_tiger_interactions.pdf', transparent=True, dpi=300)
plt.show()




# """
# Titration
# """
# property = 'titration'
# df_heldout = pd.read_csv(f'data/{property}_heldout.csv')
# fig = plt.figure(figsize=(time_complexity_width*mm, 63*mm), constrained_layout=True) 
# gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], width_ratios=[2], figure=fig)

# # [1,0]
# ax2 = fig.add_subplot(gs[1, 0])
# num_samples = range(1, 500)
# shapzero_sample_time = np.load(f'shap_results/time_per_sample_{property}.npy') 
# shapzero_sample_time = shapzero_sample_time * 3981312 # Number of samples needed for q-sft
# shapzero_shap_time = np.load(f'shap_results/time_shapzero_fsi_{property}.npy')
# overall_shapzero_time = [shapzero_sample_time + shapzero_shap_time * i for i in num_samples]
# shap_time = np.load(f'shap_results/time_fsi_{property}.npy')
# overall_shap_time = [shap_time * i for i in num_samples]
# plot_time_complexity(ax2, overall_shap_time, overall_shapzero_time, font_size=font_size, linewidth=linewidth, markersize=complexity_markersize, tot_samples=len(df_heldout), first_shap_method='SHAP-IQ', offset_intersection_text=25)
# bounds = ax2.get_ylim()




# # [0,0]
# ax1 = fig.add_subplot(gs[0, 0])
# shap_time = np.load(f'shap_results/time_shap_values_{property}.npy')
# overall_shap_time = [shap_time * i for i in num_samples]
# shapzero_shap_time = np.load(f'shap_results/time_shapzero_values_{property}.npy')
# overall_shapzero_time = [shapzero_sample_time + shapzero_shap_time * i for i in num_samples]
# plot_time_complexity(ax1, overall_shap_time, overall_shapzero_time, font_size=font_size, linewidth=linewidth, markersize=complexity_markersize, tot_samples=len(df_heldout), y_limits=bounds, x_label='', offset_intersection_text=70)
# plt.savefig('shap_results/tiger_titration_time.pdf', transparent=True, dpi=300)
# plt.show()



# # [0,1]
# fig = plt.figure(figsize=((overall_fig_width - time_complexity_width)*mm, 63*mm), constrained_layout=True) 
# gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[4, 4], figure=fig)
# shap = np.load(f'shap_results/shap_values_{property}.npy')
# random_indices = random.sample(range(0, len(shap)), top_values_shap)

# seq_length = 26
# shap_values = np.zeros((len(shap), seq_length))
# for i, shap_val in enumerate(shap):
#     # Loop through all positions in the sequence and calculate the SHAP value for each position for each guide:target pairing
#     for pos in range(seq_length): 
#         shap_values[i, pos] = shap_val[pos]

# ax2 = fig.add_subplot(gs[0, 1])
# sequences = df_heldout['Full Sequence'].values
# sequences_trimmed = sequences[random_indices]
# shap_values_trimmed = [shap_values[i,:] for i in random_indices]
# plot_shap_values(ax2, sequences_trimmed, shap_values_trimmed, x_label=None, y_label='SHAP value \n using KernelSHAP', font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth)
# bounds = ax2.get_ylim()



# # [0,0]
# ax1 = fig.add_subplot(gs[0, 0])
# shap_zero = np.load(f'shap_results/shapzero_values_{property}.npy')
# shap_zero_trimmed = [shap_zero[i,:] for i in random_indices]
# plot_shap_values(ax1, sequences_trimmed, shap_zero_trimmed, x_label=None, y_label='SHAP value \n using SHAP zero', y_limits=bounds, font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth)
# top_shap_values(sequences, shap_zero, top_values=20, filename=f'shap_results/{property}_shap_values')



# # [1,0]
# ax3 = fig.add_subplot(gs[1, 0])
# shapzero_interactions = np.load(f'shap_results/shapzero_fsi_{property}.npz', allow_pickle=True)
# shapzero_interactions_sequences = shapzero_interactions['interactions_sequences']
# shapzero_sequences = shapzero_interactions['sequences']
# encoding = {0:'A', 1:'C', 2:'T', 3:'G'}
# shapzero_sequences = [''.join(encoding[num] for num in row) for row in shapzero_sequences]
# plot_interactions_summary(ax3, shapzero_sequences, shapzero_interactions_sequences,  x_label='Target sequence position', y_label='Faith-Shap interaction \n using SHAP zero', font_size=font_size, legend=False, linewidth=linewidth)
# top_interactions(shapzero_sequences, shapzero_interactions_sequences, top_values=20, filename=f'shap_results/{property}_fsi')



# # [1,1]
# ax4 = fig.add_subplot(gs[1, 1])
# x_valid = np.load(f'shap_results/fsi_{property}_samples.npy')
# sequences = [''.join(encoding[num] for num in row) for row in x_valid]
# shap_interactions = np.load(f'shap_results/fsi_{property}.pickle', allow_pickle=True)
# plot_interactions_summary(ax4, sequences, shap_interactions,  x_label='Target sequence position', y_label='Faith-Shap interaction \n using SHAP-IQ', font_size=font_size, legend=False, linewidth=linewidth)
# plt.savefig('shap_results/tiger_titration_shap.pdf', transparent=True, dpi=300)
# plt.show()



# """
# On-target and titration interaction plots
# """
# property = 'on_target'
# fig = plt.figure(figsize=(overall_fig_width*mm, 63*mm), constrained_layout=True) 
# gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], figure=fig)

# # [0,0]
# ax1 = fig.add_subplot(gs[0, 0])
# shapzero_interactions = np.load(f'shap_results/shapzero_fsi_{property}.npz', allow_pickle=True)
# shapzero_interactions_sequences = shapzero_interactions['interactions_sequences']
# shapzero_sequences = shapzero_interactions['sequences']
# encoding = {0:'A', 1:'C', 2:'T', 3:'G'}
# sequences = [''.join(encoding[num] for num in row) for row in shapzero_sequences]
# plot_interactions(ax1, sequences, shapzero_interactions_sequences, top_values=80, x_label='', y_label='Faith-Shap interaction using \n SHAP zero for PM guide score', font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth)
# bounds = ax1.get_ylim()



# # [0,1]
# ax2 = fig.add_subplot(gs[0, 1])
# x_valid = np.load(f'shap_results/fsi_{property}_samples.npy')
# sequences = [''.join(encoding[num] for num in row) for row in x_valid]
# shap_interactions = np.load(f'shap_results/fsi_{property}.pickle', allow_pickle=True)
# plot_interactions(ax2, sequences, shap_interactions, top_values=80, x_label='', y_label='Faith-Shap interaction using \n SHAP-IQ for PM guide score', y_limits=bounds, font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth)
# bounds = ax1.get_ylim()



# # [1,0]
# property = 'titration'
# ax3 = fig.add_subplot(gs[1, 0])
# shapzero_interactions = np.load(f'shap_results/shapzero_fsi_{property}.npz', allow_pickle=True)
# shapzero_interactions_sequences = shapzero_interactions['interactions_sequences']
# shapzero_sequences = shapzero_interactions['sequences']
# encoding = {0:'A', 1:'C', 2:'T', 3:'G'}
# sequences = [''.join(encoding[num] for num in row) for row in shapzero_sequences]
# plot_interactions(ax3, sequences, shapzero_interactions_sequences, top_values=80, x_label='Target sequence position', y_label='Faith-Shap interaction using \n SHAP zero for SM guide score', font_size=font_size, markersize=markersize, legend=False, linewidth=linewidth)
# bounds = ax3.get_ylim()



# # [1,1]
# ax4 = fig.add_subplot(gs[1, 1])
# x_valid = np.load(f'shap_results/fsi_{property}_samples.npy')
# sequences = [''.join(encoding[num] for num in row) for row in x_valid]
# shap_interactions = np.load(f'shap_results/fsi_{property}.pickle', allow_pickle=True)
# plot_interactions(ax4, sequences, shap_interactions, top_values=80, x_label='Target sequence position', y_label='Faith-Shap interaction using \n SHAP-IQ for SM guide score', y_limits=bounds, font_size=font_size, markersize=markersize, legend=False, linewidth=linewidth)
# plt.savefig('shap_results/tiger_interaction_plots.pdf', transparent=True, dpi=300)
# plt.show()



# """
# DeepSHAP + FastSHAP
# """
# fig = plt.figure(figsize=(overall_fig_width*mm, 63*mm), constrained_layout=True) 
# gs = gridspec.GridSpec(2, 3, height_ratios=[2, 2], width_ratios=[4, 4, 2], figure=fig)
# def str2numpy(string):
#     string = string.strip('[]')
#     str_list = string.split(',')
#     float_list = [float(num) for num in str_list]
#     arr = np.array(float_list)
#     return arr

# # On-target
# property = 'on_target'
# """DeepSHAP"""
# # Both the guide and target sequences are used to generate the shap values. Therefore, the SHAP value at each position is the summation of the two SHAP values.
# df_shap = pd.read_csv(f'shap_results/deepshap_{property}.csv')
# pairings = ['A:T', 'C:G', 'G:C', 'T:A']
# shap_values = np.zeros((len(df_shap), seq_length))
# sequences = df_heldout['Full Sequence'].values

# for ((i, seq), sequence) in zip(df_shap.iterrows(), sequences): 

#     sequence = list(sequence)
#     # target:A,guide:A,target:C,guide:C,target:G,guide:G,target:T,guide:T
#     # Loop through all positions in the sequence and calculate the SHAP value for each position for each guide:target pairing
#     for pos in range(seq_length): # Guide RNA is from 3 to 26
#         for j, pair in enumerate(pairings):

#             guide, target = pair.split(':')

#             guide_shap = str2numpy(seq[f'guide:{guide}'])
#             target_shap = str2numpy(seq[f'target:{target}'])

#             if sequence[pos] == target:
#                 # Add the extra 3 bp context length to the shap values
#                 if pos in (0, 1, 2):
#                     shap_values[i, pos] = target_shap[pos]
#                 else:
#                     shap_values[i, pos] = target_shap[pos] + guide_shap[pos]

# ax1 = fig.add_subplot(gs[0, 0])
# sequences_trimmed = sequences[random_indices]
# shap_values_trimmed = [shap_values[i,:] for i in random_indices]
# plot_shap_values(ax1, sequences_trimmed, shap_values_trimmed, x_label=None, y_label='PM SHAP value \n using DeepSHAP', font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth)



# """FastSHAP"""
# shap = np.load(f'shap_results/fastshap_{property}.npy')
# shap_values = np.zeros((len(shap), seq_length))
# for i, shap_val in enumerate(shap):
#     # Loop through all positions in the sequence and calculate the SHAP value for each position for each guide:target pairing
#     for pos in range(seq_length): 
#         shap_values[i, pos] = shap_val[pos]

# ax2 = fig.add_subplot(gs[0, 1])
# sequences = df_heldout['Full Sequence'].values
# sequences_trimmed = sequences[random_indices]
# shap_values_trimmed = [shap_values[i,:] for i in random_indices]
# plot_shap_values(ax2, sequences_trimmed, shap_values_trimmed, x_label=None, y_label='PM SHAP value \n using FastSHAP', font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth)



# # Titration
# property = 'titration'
# """DeepSHAP"""
# # Both the guide and target sequences are used to generate the shap values. Therefore, the SHAP value at each position is the summation of the two SHAP values.
# df_shap = pd.read_csv(f'shap_results/deepshap_{property}.csv')
# pairings = ['A:T', 'C:G', 'G:C', 'T:A']
# shap_values = np.zeros((len(df_shap), seq_length))
# sequences = df_heldout['Full Sequence'].values

# for ((i, seq), sequence) in zip(df_shap.iterrows(), sequences): 

#     sequence = list(sequence)
#     # target:A,guide:A,target:C,guide:C,target:G,guide:G,target:T,guide:T
#     # Loop through all positions in the sequence and calculate the SHAP value for each position for each guide:target pairing. Since we're taking the average score, take the average of the SHAP values at each guide. 
#     for pos in range(seq_length): # Guide RNA is from 3 to 26
#         for j, pair in enumerate(pairings):

#             guide, target = pair.split(':')

#             target_shap = str2numpy(seq[f'target:{target}'])

#             if sequence[pos] == target:
#                 # Add the extra 3 bp context length to the shap values
#                 if pos in (0, 1, 2):
#                     shap_values[i, pos] = target_shap[pos]
#                 else:
#                     # Take average of the guide shap values
#                     guide_shap_A = str2numpy(seq['guide:A'])
#                     guide_shap_C = str2numpy(seq['guide:C'])
#                     guide_shap_G = str2numpy(seq['guide:G'])
#                     guide_shap_T = str2numpy(seq['guide:T'])
#                     mean_shap = np.mean([
#                         target_shap[pos] + guide_shap_A[pos],
#                         target_shap[pos] + guide_shap_C[pos],
#                         target_shap[pos] + guide_shap_G[pos],
#                         target_shap[pos] + guide_shap_T[pos]
#                     ])
#                     shap_values[i, pos] = mean_shap

# ax3 = fig.add_subplot(gs[1, 0])
# sequences_trimmed = sequences[random_indices]
# shap_values_trimmed = [shap_values[i,:] for i in random_indices]
# plot_shap_values(ax3, sequences_trimmed, shap_values_trimmed, x_label='Target sequence position', y_label='Average SM SHAP value \n using DeepSHAP', font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth)



# """FastSHAP"""
# shap = np.load(f'shap_results/fastshap_{property}.npy')
# shap_values = np.zeros((len(shap), seq_length))
# for i, shap_val in enumerate(shap):
#     # Loop through all positions in the sequence and calculate the SHAP value for each position for each guide:target pairing
#     for pos in range(seq_length): 
#         shap_values[i, pos] = shap_val[pos]

# ax4 = fig.add_subplot(gs[1, 1])
# sequences = df_heldout['Full Sequence'].values
# sequences_trimmed = sequences[random_indices]
# shap_values_trimmed = [shap_values[i,:] for i in random_indices]
# plot_shap_values(ax4, sequences_trimmed, shap_values_trimmed, x_label='Target sequence position', y_label='Average SM SHAP value \n using FastSHAP', font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth)






