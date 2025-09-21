import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
import matplotlib.font_manager as fm
font_path = '../gen/Helvetica.ttf'
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



"""
Plotting functions
"""
def plot_shap_value_convergence(ax, shap_values_list, x_values, color="#4daf4a", ylabel="$R^2$", vline=None):
    ref = shap_values_list[-1]  # reference SHAP values (e.g. at largest background size)
    num_sequences = ref.shape[0]

    mean_r2 = []
    std_r2 = []

    for shap_values in shap_values_list[:-1]:  # exclude reference
        r2_all = []
        for i in range(num_sequences):
            ref_vals = ref[i].flatten()
            cur_vals = shap_values[i].flatten()

            if len(ref_vals) < 2:
                continue

            r2 = r2_score(ref_vals, cur_vals)
            r2_all.append(max(r2, 0))  # enforce minimum R² = 0

        mean_r2.append(np.nanmean(r2_all))
        std_r2.append(np.nanstd(r2_all))
    
    x_values.append(x_values[-1])
    mean_r2.append(mean_r2[-1])
    std_r2.append(std_r2[-1])

    ax.plot(x_values[:-1], mean_r2, marker='o', markersize=markersize, color=color)
    ax.fill_between(x_values[:-1],
                    np.array(mean_r2) - np.array(std_r2),
                    np.array(mean_r2) + np.array(std_r2),
                    alpha=0.3, color=color)
    # ax.set_xticks(x_values[:-1])
    ax.set_xlabel("Background sample size", fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(top=1)
    if vline is not None:
        ax.axvline(vline, color='black', linestyle='--', linewidth=linewidth)



def plot_shap_interaction_convergence(ax, all_results, budgets, valid_range=None, color="#377eb8", ylabel="$R^2$", vline=None):
    ref = all_results[budgets[-1]]
    num_sequences = len(ref)

    mean_r2 = []
    std_r2 = []

    for budget in budgets[:-1]:
        current = all_results[budget]
        r2_all = []

        for i in range(num_sequences):
            ref_i = ref[i]
            cur_i = current[i]

            if valid_range is not None:
                # Filter keys to be within valid range
                common_keys = {
                    k for k in ref_i.keys() & cur_i.keys()
                    if len(k) > 0 and all(pos in valid_range for pos in k)
                }
            else:
                common_keys = set(ref_i.keys()) & set(cur_i.keys())

            # Group by interaction order
            keys_by_order = {}
            for k in common_keys:
                order = len(k)
                keys_by_order.setdefault(order, []).append(k)

            for order, keys in keys_by_order.items():
                ref_vals = [ref_i[k] for k in keys]
                cur_vals = [cur_i[k] for k in keys]

                # Skip 0th order term
                if len(ref_vals) < 2:
                    continue 

                r2 = r2_score(ref_vals, cur_vals)
                r2_all.append(r2)

        # Make minimum R² equal to 0
        r2_all = [max(r2, 0) for r2 in r2_all]

        mean_r2.append(np.nanmean(r2_all))
        std_r2.append(np.nanstd(r2_all))

    ax.plot(budgets[:-1], mean_r2, marker='o', markersize=markersize, color=color)
    ax.fill_between(budgets[:-1],
                    np.array(mean_r2) - np.array(std_r2),
                    np.array(mean_r2) + np.array(std_r2),
                    alpha=0.3, color=color)
    # ax.set_xticks(budgets[:-1])
    ax.set_xlabel("Budget", fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(top=1)
    if vline is not None:
        ax.axvline(vline, color='black', linestyle='--', linewidth=linewidth)



    
"""
Create figure
"""
fig = plt.figure(figsize=(overall_fig_width*mm, 100*mm), constrained_layout=True) 
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1], figure=fig)

"""TIGER"""
property = "on_target" 
color = "#226e9c"
y_label_shap = "KernelSHAP $R^2$ (TIGER)"
y_label_interaction = "Faith-Shap $R^2$ (TIGER)"
ax1 = fig.add_subplot(gs[0, 0])
num_background_samples_total = [5, 10, 15, 30, 40, 50]
shap_values_list = [
    np.load(f"../TIGER/figures/samples_search/shap_values_{property}_{n}.npy") for n in num_background_samples_total
]
plot_shap_value_convergence(ax1, shap_values_list, num_background_samples_total, color=color, ylabel=y_label_shap, vline=30)

# Faith-Shap interactions
budgets = [1000, 10000, 30000, 50000, 60000, 70000]
path = "../TIGER/figures/samples_search/fsi_{}_{}.pickle"
all_results = {}
for budget in budgets:
    with open(path.format(property, budget), "rb") as f:
        all_results[budget] = pickle.load(f)
ax2 = fig.add_subplot(gs[0, 1])

plot_shap_interaction_convergence(ax2, all_results, budgets, color=color, ylabel=y_label_interaction, vline=50000)



"""inDelphi"""
property = "HEK293" 
color = "#AF588A"
y_label_shap = "KernelSHAP $R^2$ (inDelphi)"
y_label_interaction = "Faith-Shap $R^2$ (inDelphi)"
ax1 = fig.add_subplot(gs[1, 0])
num_background_samples_total = [5, 10, 15, 30, 40, 50]
shap_values_list = [
    np.load(f"../inDelphi/figures/samples_search/shap_values_{property}_{n}.npy") for n in num_background_samples_total
]
plot_shap_value_convergence(ax1, shap_values_list, num_background_samples_total, color=color, ylabel=y_label_shap, vline=30)

# Faith-Shap interactions
budgets = [1000, 5000, 10000, 20000, 25000, 30000]
path = "../inDelphi/figures/samples_search/fsi_{}_{}.pickle"
all_results = {}
for budget in budgets:
    with open(path.format(property, budget), "rb") as f:
        all_results[budget] = pickle.load(f)
ax2 = fig.add_subplot(gs[1, 1])

plot_shap_interaction_convergence(ax2, all_results, budgets, color=color, ylabel=y_label_interaction, valid_range=set(range(16, 23)), vline=20000)



"""Tranception"""
property = "GFP_Tranception" 
color = "#F6921E"
y_label_shap = "KernelSHAP $R^2$ (Tranception)"
y_label_interaction = "Faith-Shap $R^2$ (Tranception)"
ax3 = fig.add_subplot(gs[2, 0])
num_background_samples_total = [10, 20, 30, 50, 70, 79]
shap_values_list = [
    np.load(f"../Tranception/figures/samples_search/shap_values_{property}_{n}.npy") for n in num_background_samples_total
]
plot_shap_value_convergence(ax3, shap_values_list, num_background_samples_total, color=color, ylabel=y_label_shap, vline=79)

# Faith-Shap interactions
budgets = [1000, 10000, 30000, 50000, 60000, 70000]
path = "../Tranception/figures/samples_search/fsi_{}_{}.pickle"
all_results = {}
for budget in budgets:
    with open(path.format(property, budget), "rb") as f:
        all_results[budget] = pickle.load(f)
ax2 = fig.add_subplot(gs[2, 1])

plot_shap_interaction_convergence(ax2, all_results, budgets, color=color, ylabel=y_label_interaction, vline=10000)


fig.savefig('supp_shap_convergence.pdf')
