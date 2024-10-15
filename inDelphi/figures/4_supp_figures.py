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



# FastSHAP
shap = np.load(f'shap_results/fastshap_{celltype}_{property}.npy')
random_indices = random.sample(range(0, len(shap)), top_values_shap)
shap_values = np.zeros((len(shap), seq_length))
for i, shap_val in enumerate(shap):
    for pos in range(seq_length): 
        shap_values[i, pos] = shap_val[pos]

ax1 = fig.add_subplot(gs[0, 0])
sequences = df_heldout['Left_seq'].values + df_heldout['Right_seq'].values
sequences_trimmed = sequences[random_indices]
shap_values_trimmed = [shap_values[i,:] for i in random_indices]
plot_shap_values(ax1, sequences_trimmed, shap_values_trimmed, x_label='Target sequence position', y_label='SHAP value \n using FastSHAP', font_size=font_size, markersize=markersize, legend=True, legend_marker_size=legend_marker_size, linewidth=linewidth)

# Correlation between KernelSHAP
kernelshap = np.load(f'shap_results/shap_values_{celltype}.npy')[0, :, :]
correlation_shap_values(kernelshap, shap_values)



# Time complexity 
# SHAP zero
num_samples = range(1, 500)
shapzero_sample_time = np.load(f'shap_results/time_per_sample_{celltype}.npy') 
shapzero_sample_time = shapzero_sample_time * 3981312 # Number of samples needed for q-sft
shapzero_shap_time = np.load(f'shap_results/time_shapzero_values_{celltype}_{property}.npy')
overall_shapzero_time = [shapzero_sample_time + shapzero_shap_time * i for i in num_samples]

# FastSHAP
fastshap_surrogate_time = np.load(f'shap_results/time_fastshap_surrogate_{celltype}_{property}.npy')
fastshap_explainer_time = np.load(f'shap_results/time_fastshap_explainer_{celltype}_{property}.npy')
fastshap_inference_time = np.load(f'shap_results/time_fastshap_inference_{celltype}_{property}.npy')
overall_fastshap_time = [fastshap_surrogate_time + fastshap_explainer_time + (fastshap_inference_time / len(df_heldout)) * i for i in num_samples]

ax2 = fig.add_subplot(gs[0, 1])
plot_time_complexity(ax2, overall_fastshap_time, overall_shapzero_time, font_size=font_size, linewidth=linewidth, markersize=complexity_markersize, tot_samples=len(df_heldout), offset_intersection_text=70, legend_loc='upper right') # y_limits=bounds


plt.savefig('shap_results/SUPP_inDelphi_altshap.pdf', transparent=True, dpi=300)
plt.show()


