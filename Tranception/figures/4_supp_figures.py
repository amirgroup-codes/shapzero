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
from gen.shapzero import correlation_shap_values, plot_shap_values_protein_top_mutations
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
plt.rcParams['font.size'] = font_size
linewidth = 0.25
plt.rcParams['axes.linewidth'] = linewidth
overall_fig_width = 183 # Measurements in mm
time_complexity_width = 45
aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
rev_aa_dict = {v: k for k, v in aa_dict.items()}



"""
FastSHAP
"""
property = 'GFP_Tranception'
data_heldout = np.load(f'data/{property}_heldout.npy')
fig = plt.figure(figsize=(overall_fig_width*mm, 50*mm), constrained_layout=True) 
gs = gridspec.GridSpec(1, 2, height_ratios=[2], width_ratios=[4, 3], figure=fig)
seq_length = 10

# FastSHAP
shap = np.load(f'shap_results/fastshap_{property}.npy')
ax1 = fig.add_subplot(gs[0, 0])
sequences = [''.join(rev_aa_dict[num] for num in row) for row in data_heldout]
shap_zero = np.load(f'shap_results/shapzero_values_{property}.npy')
plot_shap_values_protein_top_mutations(ax1, sequences, shap, colorbar_label='FastSHAP \n value')
correlation_shap_values(shap_zero, shap)



# Time complexity 
# SHAP zero
ax2 = fig.add_subplot(gs[0, 1])
num_samples = range(1, 210)
shapzero_sample_time = np.load(f'shap_results/time_per_sample_{property}.npy') 
shapzero_sample_time = shapzero_sample_time * 792000 # Number of samples needed for q-sft
shapzero_shap_time = np.load(f'shap_results/time_shapzero_values_{property}.npy')
overall_shapzero_time = [shapzero_sample_time + shapzero_shap_time * i for i in num_samples]

# FastSHAP
fastshap_surrogate_time = np.load(f'shap_results/time_fastshap_surrogate_{property}.npy')
fastshap_explainer_time = np.load(f'shap_results/time_fastshap_explainer_{property}.npy')
fastshap_inference_time = np.load(f'shap_results/time_fastshap_inference_{property}.npy')
overall_fastshap_time = [fastshap_surrogate_time + fastshap_explainer_time + fastshap_inference_time * i for i in num_samples]

plot_time_complexity(ax2, overall_fastshap_time, overall_shapzero_time, font_size=font_size, linewidth=linewidth, markersize=complexity_markersize, tot_samples=data_heldout.shape[0], offset_intersection_text=70, legend_loc='upper right', x_label='Number of sequences explained', first_shap_method='FastSHAP') 


plt.savefig('shap_results/SUPP_Tranception_altshap.pdf', transparent=True, dpi=300)
plt.show()

