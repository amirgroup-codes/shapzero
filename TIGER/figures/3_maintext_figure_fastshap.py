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
from gen.shapzero import plot_shap_values, plot_interactions_summary, top_shap_values, top_interactions
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
font_size = 5
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
plot_time_complexity(ax2, overall_shap_time, overall_shapzero_time, font_size=font_size, linewidth=linewidth, markersize=complexity_markersize, tot_samples=len(df_heldout), first_shap_method='SHAP-IQ Faith-Shap', offset_intersection_text=25)
bounds = ax2.get_ylim()



"""
[0,0]
"""
ax1 = fig.add_subplot(gs[0, 0])
fastshap_surrogate_time = np.load(f'shap_results/time_fastshap_surrogate_{property}.npy')
fastshap_explainer_time = np.load(f'shap_results/time_fastshap_explainer_{property}.npy')
fastshap_inference_time = np.load(f'shap_results/time_fastshap_inference_{property}.npy')
overall_shap_time = [fastshap_surrogate_time + fastshap_explainer_time + fastshap_inference_time * i for i in num_samples]
plot_time_complexity(ax1, overall_shap_time, overall_shapzero_time, y_limits=bounds, font_size=font_size, linewidth=linewidth, markersize=complexity_markersize, tot_samples=len(df_heldout), x_label='', offset_intersection_text=70) # y_limits=bounds
plt.savefig('shap_results/tiger_time_test.pdf', transparent=True, dpi=300)
plt.show()



"""
[0,1]
"""
# Make SHAP plots
fig = plt.figure(figsize=((overall_fig_width - time_complexity_width)*mm, 63*mm), constrained_layout=True) 
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[4, 4], figure=fig)
shap = np.load(f'shap_results/fastshap_{property}.npy')
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
plot_shap_values(ax2, sequences, shap_values_trimmed, x_label=None, y_label='SHAP value \n using Kernel SHAP', font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth)
bounds = ax2.get_ylim()



"""
[0,0]
"""
ax1 = fig.add_subplot(gs[0, 0])
shap_zero = np.load(f'shap_results/shapzero_values_{property}.npy')
shap_zero_trimmed = [shap_zero[i,:] for i in random_indices]
plot_shap_values(ax1, sequences, shap_zero_trimmed, x_label=None, y_label='SHAP value \n using SHAP Zero', font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth) #  y_limits=bounds,
top_shap_values(sequences, shap_zero_trimmed, top_values=20, filename=f'shap_results/{property}_shap_values')



"""
[1,0]
"""
ax3 = fig.add_subplot(gs[1, 0])
shapzero_interactions = np.load(f'shap_results/shapzero_fsi_{property}.npz', allow_pickle=True)
shapzero_interactions_sequences = shapzero_interactions['interactions_sequences']
shapzero_sequences = shapzero_interactions['sequences']
encoding = {0:'A', 1:'C', 2:'T', 3:'G'}
sequences = [''.join(encoding[num] for num in row) for row in shapzero_sequences]
plot_interactions_summary(ax3, sequences, shapzero_interactions_sequences,  x_label='Target sequence position', y_label='Faith-Shap interaction \n using F-SHAP', font_size=font_size, markersize=markersize, legend=False, linewidth=linewidth)
top_interactions(sequences, shapzero_interactions_sequences, top_values=20, filename=f'shap_results/{property}_fsi')



"""
[1,1]
"""
ax4 = fig.add_subplot(gs[1, 1])
x_valid = np.load(f'shap_results/fsi_{property}_samples.npy')
sequences = [''.join(encoding[num] for num in row) for row in x_valid]
shap_interactions = np.load(f'shap_results/fsi_{property}.pickle', allow_pickle=True)
plot_interactions_summary(ax4, sequences, shap_interactions,  x_label='Target sequence position', y_label='Faith-Shap interaction \n using SHAP-IQ', font_size=font_size, markersize=markersize, legend=False, linewidth=linewidth)
plt.savefig('shap_results/tiger_shap_test.pdf', transparent=True, dpi=300)
plt.show()









# DeepSHAP
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
plot_time_complexity(ax2, overall_shap_time, overall_shapzero_time, font_size=font_size, linewidth=linewidth, markersize=complexity_markersize, tot_samples=len(df_heldout), first_shap_method='SHAP-IQ Faith-Shap', offset_intersection_text=25)
bounds = ax2.get_ylim()



"""
[0,0]
"""
ax1 = fig.add_subplot(gs[0, 0])
deepshap_time = np.load(f'shap_results/time_deepshap_{property}.npy') / len(df_heldout)
print(deepshap_time * len(df_heldout))
overall_shap_time = [deepshap_time * i for i in num_samples]
plot_time_complexity(ax1, overall_shap_time, overall_shapzero_time, font_size=font_size, linewidth=linewidth, markersize=complexity_markersize, tot_samples=len(df_heldout), x_label='', offset_intersection_text=70) # y_limits=bounds
plt.savefig('shap_results/tiger_time_test_deepshap.pdf', transparent=True, dpi=300)
plt.show()



"""
[0,1]
"""
# Make SHAP plots
fig = plt.figure(figsize=((overall_fig_width - time_complexity_width)*mm, 63*mm), constrained_layout=True) 
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[4, 4], figure=fig)
shap = np.load(f'shap_results/fastshap_{property}.npy')
random_indices = random.sample(range(0, len(shap)), top_values_shap)

# seq_length = 26
# shap_values = np.zeros((len(shap), seq_length))
# for i, shap_val in enumerate(shap):
#     # Loop through all positions in the sequence and calculate the SHAP value for each position for each guide:target pairing
#     for pos in range(seq_length): 
#         shap_values[i, pos] = shap_val[pos]

def str2numpy(string):
    string = string.strip('[]')
    str_list = string.split(',')
    float_list = [float(num) for num in str_list]
    arr = np.array(float_list)
    return arr

"""
Both the guide and target sequences are used to generate the shap values. Therefore, the SHAP value at each position is the summation of the two SHAP values.
"""
df_shap = pd.read_csv(f'shap_results/deepshap_{property}.csv')
seq_length = 26
pairings = ['A:T', 'C:G', 'G:C', 'T:A']
# shap_values = np.zeros((len(df_shap), seq_length, len(pairings)))
shap_values = np.zeros((len(df_shap), seq_length))
sequences = df_heldout['Full Sequence'].values

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
                    shap_values[i, pos] = target_shap[pos] + guide_shap[pos]# guide_shap[seq_length - pos + 2]

# shap_values = shap_values[:, :, 0]
ax2 = fig.add_subplot(gs[0, 1])
sequences_trimmed = sequences[random_indices]
shap_values_trimmed = [shap_values[i,:] for i in random_indices]
plot_shap_values(ax2, sequences, shap_values_trimmed, x_label=None, y_label='SHAP value \n using Kernel SHAP', font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth)
bounds = ax2.get_ylim()

# for seq, sample in zip(sequences, shap_values):

#         seq = list(seq)
#         color = []
#         marker = []
#         for nt in seq:
#             color.append(colors.get(nt, 'k'))
#             marker.append(markers.get(nt, 'o'))
#         for i, (x, m, c) in enumerate(zip(range(1, seq_length + 1), marker, color)):
#             ax.scatter(x, sample[i], color=c, marker=m, s=markersize, alpha=0.6)


"""
[0,0]
"""
ax1 = fig.add_subplot(gs[0, 0])
shap_zero = np.load(f'shap_results/shapzero_values_{property}.npy')
shap_zero_trimmed = [shap_zero[i,:] for i in random_indices]
plot_shap_values(ax1, sequences, shap_zero_trimmed, x_label=None, y_label='SHAP value \n using SHAP Zero', font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth) #  y_limits=bounds,
top_shap_values(sequences, shap_zero_trimmed, top_values=20, filename=f'shap_results/{property}_shap_values')



"""
[1,0]
"""
ax3 = fig.add_subplot(gs[1, 0])
shapzero_interactions = np.load(f'shap_results/shapzero_fsi_{property}.npz', allow_pickle=True)
shapzero_interactions_sequences = shapzero_interactions['interactions_sequences']
shapzero_sequences = shapzero_interactions['sequences']
encoding = {0:'A', 1:'C', 2:'T', 3:'G'}
sequences = [''.join(encoding[num] for num in row) for row in shapzero_sequences]
plot_interactions_summary(ax3, sequences, shapzero_interactions_sequences,  x_label='Target sequence position', y_label='Faith-Shap interaction \n using F-SHAP', font_size=font_size, markersize=markersize, legend=False, linewidth=linewidth)
top_interactions(sequences, shapzero_interactions_sequences, top_values=20, filename=f'shap_results/{property}_fsi')



"""
[1,1]
"""
ax4 = fig.add_subplot(gs[1, 1])
x_valid = np.load(f'shap_results/fsi_{property}_samples.npy')
sequences = [''.join(encoding[num] for num in row) for row in x_valid]
shap_interactions = np.load(f'shap_results/fsi_{property}.pickle', allow_pickle=True)
plot_interactions_summary(ax4, sequences, shap_interactions,  x_label='Target sequence position', y_label='Faith-Shap interaction \n using SHAP-IQ', font_size=font_size, markersize=markersize, legend=False, linewidth=linewidth)
plt.savefig('shap_results/tiger_shap_test_deepshap.pdf', transparent=True, dpi=300)
plt.show()