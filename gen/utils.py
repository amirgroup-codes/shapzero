import os
import re
import pickle
import zlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from scipy.stats import pearsonr, spearmanr

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(zlib.decompress(f.read()))
    return data

def save_data(data, filename):
    with open(filename, 'wb') as f:
        f.write(zlib.compress(pickle.dumps(data, protocol=4), 9))
        
def count_interactions(locations):
    nonzero_counts = {}
    for row in locations:
        nonzero_indices = np.nonzero(row)[0]
        num_nonzero_indices = len(nonzero_indices)
        nonzero_counts[num_nonzero_indices] = nonzero_counts.get(num_nonzero_indices, 0) + 1
    
    for num_nonzero_indices, count in nonzero_counts.items():
        print("There are {} {}-order interactions.".format(count, num_nonzero_indices))
    
    return nonzero_counts

def calculate_fourier_magnitudes(locations, gwht):
    nonzero_counts = count_interactions(locations)
    k_values = sorted(nonzero_counts.keys())
    j = 0 if 0 in k_values else 1
    F_k_values = np.zeros(max(np.max(k_values)+1, len(k_values)))

    for row in locations:
        nonzero_indices = np.nonzero(row)[0]
        num_nonzero_indices = len(nonzero_indices)
        F_k_values[num_nonzero_indices-j] += np.abs(gwht[row])
    
    F_k_values = np.square(F_k_values)
    return dict(zip(k_values, F_k_values))

def plot_interaction_magnitudes(sum_squares, q, n, b, output_folder, args):
    index_counts = list(sum_squares.keys())
    values = list(sum_squares.values())
    plt.figure()
    plt.bar(index_counts, values, align='center', color='limegreen')
    plt.xlabel('$r^{th}$ order interactions')
    plt.ylabel('Magnitude of Fourier coefficients')
    plt.xticks(index_counts)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('q{}_n{}_b{}'.format(q, n, b))

    if args.param:
        param_path = args.param.replace('/', '_')
        file_path = 'magnitude_of_interactions_{}.png'.format(param_path)
    else:
        file_path = 'magnitude_of_interactions.png'
    plt.savefig(output_folder / file_path)
    plt.close()

def write_results_to_file(results_file, q, n, b, noise_sd, n_used, r2_value, nmse, avg_hamming_weight, max_hamming_weight):
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, "w") as file:
        file.write("q = {}, n = {}, b = {}, noise_sd = {}\n".format(q, n, b, noise_sd))
        file.write("\nTotal samples = {}\n".format(n_used))
        file.write("Total sample ratio = {}\n".format(n_used / q ** n))
        file.write("R^2 = {}\n".format(r2_value))
        file.write("NMSE = {}\n".format(nmse))
        file.write("AVG Hamming Weight of Nonzero Locations = {}\n".format(avg_hamming_weight))
        file.write("Max Hamming Weight of Nonzero Locations = {}\n".format(max_hamming_weight))
        

def summarize_results(locations, gwht, q, n, b, noise_sd, n_used, r2_value, nmse, avg_hamming_weight, max_hamming_weight, folder, args):

    sum_squares = calculate_fourier_magnitudes(locations, gwht)
    plot_interaction_magnitudes(sum_squares, q, n, b, folder, args)

    if args.param:
        param_path = args.param.replace('/', '_')
        file_path = 'helper_results_{}.txt'.format(param_path)
    else:
        file_path = 'helper_results.txt'

    results_file = folder / file_path
    write_results_to_file(results_file, q, n, b, noise_sd, n_used, r2_value, nmse, avg_hamming_weight, max_hamming_weight)

def run_linear_model(sequences_train, y_train, sequences_test, y_test):
    """
    Runs a linear model, feeding in a list of sequences and their corresponding values
    Returns the pearson correlation coefficient for the test set
    """

    # One-hot encoding nucleotides
    nts = ['A', 'C', 'G', 'T']
    one_hot_encoding = {}
    for i, nt in enumerate(nts):
        encoding = [0] * len(nts)
        encoding[i] = 1
        one_hot_encoding[nt] = encoding

    # One-hot encode the training sequences
    one_hot_encoded_sequences_train = []
    for sequence in sequences_train:
        sequence = list(sequence)
        one_hot_encoded_sequence = []
        for nt in sequence:
            one_hot_encoded_sequence.extend(one_hot_encoding[nt])
        one_hot_encoded_sequences_train.append(one_hot_encoded_sequence)
    one_hot_encoded_sequences_train = np.array(one_hot_encoded_sequences_train)

    # One-hot encode the test sequences
    one_hot_encoded_sequences_test = []
    for sequence in sequences_test:
        sequence = list(sequence)
        one_hot_encoded_sequence = []
        for nt in sequence:
            one_hot_encoded_sequence.extend(one_hot_encoding[nt])
        one_hot_encoded_sequences_test.append(one_hot_encoded_sequence)
    one_hot_encoded_sequences_test = np.array(one_hot_encoded_sequences_test)

    # Run the model
    model = Ridge(alpha=0.5)
    model.fit(one_hot_encoded_sequences_train, y_train)
    y_pred = model.predict(one_hot_encoded_sequences_test)
    corr = pearsonr(y_test, y_pred)[0]

    return corr, y_pred

degree = 2 

def run_pairwise_model(sequences_train, y_train, sequences_test, y_test):
    """
    Runs a pairwise model, feeding in a list of sequences and their corresponding values
    Returns the pearson correlation coefficient for the test set
    """

    # One-hot encoding nucleotides
    nts = ['A', 'C', 'G', 'T']
    one_hot_encoding = {}
    for i, nt in enumerate(nts):
        encoding = [0] * len(nts)
        encoding[i] = 1
        one_hot_encoding[nt] = encoding

    # One-hot encode the training sequences
    one_hot_encoded_sequences_train = []
    for sequence in sequences_train:
        sequence = list(sequence)
        one_hot_encoded_sequence = []
        for nt in sequence:
            one_hot_encoded_sequence.extend(one_hot_encoding[nt])
        one_hot_encoded_sequences_train.append(one_hot_encoded_sequence)
    one_hot_encoded_sequences_train = np.array(one_hot_encoded_sequences_train)

    # One-hot encode the test sequences
    one_hot_encoded_sequences_test = []
    for sequence in sequences_test:
        sequence = list(sequence)
        one_hot_encoded_sequence = []
        for nt in sequence:
            one_hot_encoded_sequence.extend(one_hot_encoding[nt])
        one_hot_encoded_sequences_test.append(one_hot_encoded_sequence)
    one_hot_encoded_sequences_test = np.array(one_hot_encoded_sequences_test)

    # Run the model
    model = Pipeline([
        ('poly_features', PolynomialFeatures(degree=degree, interaction_only=True)),  
        ('linear_regression', Ridge(alpha=0.5)) 
    ])
    model.fit(one_hot_encoded_sequences_train, y_train)
    y_pred = model.predict(one_hot_encoded_sequences_test)
    corr = pearsonr(y_test, y_pred)[0]

    return corr, y_pred


def plot_scatter_with_best_fit(ax, y, y_hat_model, pearson_corr, x_label='Ground-truth score', y_label='Predicted score', linewidth=0.25, font_size = 5, x_limits=None, y_limits=None, color='#00a087', marker='o', markersize=0.25, legend_marker_size=3):
    """
    Scatter plot with best fit line when comparing against different models
    """
    ax.scatter(y, y_hat_model, marker=marker, color=color, label='Empirical samples', s=markersize)
    coefficients = np.polyfit(y, y_hat_model, 1)
    poly_fit = np.poly1d(coefficients)
    line_of_best_fit = poly_fit(y)
    ax.plot(y, line_of_best_fit, color='#9b9ca0', linewidth=linewidth, label='Line of Best Fit')

    ax.set_ylabel(y_label, fontsize=font_size)
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size, width=linewidth)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if x_limits is not None:
        ax.set_xlim(x_limits)
    if y_limits is not None:
        ax.set_ylim(y_limits)

    legend_label = 'Pearson $r$ = {}'.format(pearson_corr.round(2))
    legend = ax.legend([ax.collections[0]], [legend_label], loc='lower right', markerscale=legend_marker_size, fontsize=font_size)
    legend.get_frame().set_linewidth(linewidth)

def summary_stats(y, y_hat):
    """
    Generates pearson and spearman for two numpy arrays. Returns a dataframe with statistics.

    Parameters:
        y (numpy.ndarray): The true values.
        y_hat (numpy.ndarray): The predicted values.
    """
    pearson_corr, _ = pearsonr(y, y_hat)
    spearman_corr, _ = spearmanr(y, y_hat)
    return pd.DataFrame({
        'pearson': pearson_corr,
        'spearman': spearman_corr
    }, index=[0])


def plot_time_complexity(ax, overall_shap_time, overall_mobius_time, font_size=6, x_label='Number of samples', y_label='Total runtime (seconds)', first_shap_method='KernelSHAP', first_shap_color='#00a087', second_shap_method='SHAP zero', second_shap_color='#3c5488', y_limits=None, legend=True, linewidth=0.2, tot_samples=None, markersize=0.25, legend_loc='upper left', offset_intersection_text=100):
    """
    Plot SHAP compute time per sample
    Parameters:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        overall_shap_time (list): List of cumulative runtimes for first SHAP method.
        overall_mobius_time (list): List of cumulative runtimes for second SHAP method.
        font_size (int, optional): Font size for labels and text. Defaults to 6.
        x_label (str, optional): Label for x-axis. Defaults to 'Number of samples'.
        y_label (str, optional): Label for y-axis. Defaults to 'Total runtime (seconds)'.
        first_shap_method (str, optional): Name of first SHAP method. Defaults to 'KernelSHAP'.
        first_shap_color (str, optional): Color for first SHAP method line. Defaults to '#00a087'.
        second_shap_method (str, optional): Name of second SHAP method. Defaults to 'SHAP zero'.
        second_shap_color (str, optional): Color for second SHAP method line. Defaults to '#3c5488'.
        y_limits (tuple, optional): Limits for y-axis. Defaults to None.
        legend (bool, optional): Whether to show legend. Defaults to True.
        linewidth (float, optional): Width of plotted lines. Defaults to 0.2.
        tot_samples (int, optional): Number of samples to mark with x. Defaults to None.
        markersize (float, optional): Size of markers. Defaults to 0.25.
        legend_loc (str, optional): Location of legend. Defaults to 'upper left'.
        offset_intersection_text (int, optional): Offset for intersection annotation. Defaults to 100.
    """
    num_samples = range(1, len(overall_mobius_time) + 1)
    # Find where shap methods intersect
    intersection = np.where(np.array(overall_shap_time) > np.array(overall_mobius_time))[0]
    if len(intersection) > 0:
        intersection = intersection[0]
        if intersection + offset_intersection_text < num_samples[-1]:
            intersection_text_loc = intersection + offset_intersection_text
        else:
            intersection_text_loc = intersection - offset_intersection_text
        if intersection == 1:
            ax.annotate(f'{intersection} explanation',  
                xy=(intersection, overall_shap_time[intersection]), 
                xytext=(intersection_text_loc, overall_shap_time[intersection] * 1.5),
                arrowprops=dict(facecolor='black', arrowstyle='-|>', lw=linewidth),
                bbox=dict(pad=0, facecolor="none", edgecolor="none"),
                fontsize=font_size, color='black', zorder=10)
        else:
            ax.annotate(f'{intersection} explanations',  
                xy=(intersection, overall_shap_time[intersection]), 
                xytext=(intersection_text_loc, overall_shap_time[intersection] * 1.5),
                arrowprops=dict(facecolor='black', arrowstyle='-|>', lw=linewidth),
                bbox=dict(pad=0, facecolor="none", edgecolor="none"),
                fontsize=font_size, color='black', zorder=10)
        ax.axvline(x=intersection, color='black', linestyle='--', linewidth=linewidth)

    ax.plot(num_samples, overall_mobius_time, '-', color=second_shap_color, label=second_shap_method, linewidth=linewidth)
    ax.plot(num_samples, overall_shap_time, '-', color=first_shap_color, label=first_shap_method, linewidth=linewidth, dashes=(10, 5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=font_size, width=linewidth)
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.legend(loc='upper right').get_frame().set_linewidth(linewidth)

    if y_limits is not None:
        ax.set_ylim(y_limits)
    if legend:
        legend = ax.legend(loc=legend_loc, fontsize=font_size)
        legend.get_frame().set_linewidth(0.5)
    if tot_samples is not None:
        ax.plot(tot_samples, overall_shap_time[tot_samples], marker='x', color='black', markersize=markersize, zorder=10)


def plot_multiple_time_complexity(ax, overall_shap_time, overall_mobius_time, overall_third_shap_time, font_size=6, x_label='Number of samples', y_label='Total runtime (seconds)', first_shap_method='FastSHAP', first_shap_color='#00a087', second_shap_method='SHAP zero', second_shap_color='#3c5488', third_shap_method='DeepSHAP', third_shap_color='#f39b7f', y_limits=None, legend=True, linewidth=0.2, tot_samples=None, markersize=0.25, legend_loc='lower right', offset_intersection_text=100):
    """
    Plot SHAP compute time per sample
    """
    num_samples = range(1, len(overall_mobius_time) + 1)
    # Find where shap methods intersect
    for shap_time in [overall_shap_time, overall_third_shap_time]:
        intersection = np.where(np.array(shap_time) > np.array(overall_mobius_time))[0]
        if len(intersection) > 0:
            intersection = intersection[0]
            if intersection + offset_intersection_text < num_samples[-1]:
                intersection_text_loc = intersection + offset_intersection_text
            else:
                intersection_text_loc = intersection - offset_intersection_text
            if intersection == 1:
                ax.annotate(f'{intersection} explanation',  
                    xy=(intersection, shap_time[intersection]), 
                    xytext=(intersection_text_loc, shap_time[intersection] * 1.5),
                    arrowprops=dict(facecolor='black', arrowstyle='-|>', lw=linewidth),
                    bbox=dict(pad=0, facecolor="none", edgecolor="none"),
                    fontsize=font_size, color='black', zorder=10)
            else:
                ax.annotate(f'{intersection} explanations',  
                    xy=(intersection, shap_time[intersection]), 
                    xytext=(intersection_text_loc, shap_time[intersection] * 1.5),
                    arrowprops=dict(facecolor='black', arrowstyle='-|>', lw=linewidth),
                    bbox=dict(pad=0, facecolor="none", edgecolor="none"),
                    fontsize=font_size, color='black', zorder=10)
            ax.axvline(x=intersection, color='black', linestyle='--', linewidth=linewidth)

    ax.plot(num_samples, overall_mobius_time, '-', color=second_shap_color, label=second_shap_method, linewidth=linewidth)
    ax.plot(num_samples, overall_shap_time, '-', color=first_shap_color, label=first_shap_method, linewidth=linewidth, dashes=(10, 5))
    ax.plot(num_samples, overall_third_shap_time, '-', color=third_shap_color, label=third_shap_method, linewidth=linewidth, dashes=(10, 5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=font_size, width=linewidth)
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.legend(loc='upper right').get_frame().set_linewidth(linewidth)

    if y_limits is not None:
        ax.set_ylim(y_limits)
    if legend:
        legend = ax.legend(loc=legend_loc, fontsize=font_size)
        legend.get_frame().set_linewidth(0.5)
    if tot_samples is not None:
        ax.plot(tot_samples, overall_third_shap_time[tot_samples], marker='x', color='black', markersize=markersize, zorder=10)


def compute_fourier_output(seqs_qary, qsft_transform, q):
    """
    Given a list of q-ary encoded sequence and the qsft transform, compute the predicted Fourier output.
    Parameters:
        seqs_qary (np.ndarray): Array of q-ary encoded sequences
        qsft_transform (dict): Dictionary mapping frequency vectors to Fourier coefficients
        q (int): Size of alphabet (e.g. 4 for DNA/RNA sequences)
        
    Returns:
        y_hat (np.ndarray): Predicted outputs for each sequence using the Fourier transform
    """
    batch_size = 10000
    beta_keys = list(qsft_transform.keys())
    beta_values = list(qsft_transform.values())
    y_hat = []
    for i in range(0, len(seqs_qary), batch_size):
        seqs_qary_batch = seqs_qary[i:i + batch_size, :]
        freqs = np.array(seqs_qary_batch) @ np.array(beta_keys).T
        H = np.exp(2j * np.pi * freqs / q)
        y_hat.append(H @ np.array(beta_values))

    y_hat = np.concatenate(y_hat)

    return y_hat
