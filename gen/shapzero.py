import pickle
import cmath
from itertools import combinations, product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from scipy.stats import pearsonr

class shapzero:
    def __init__(self, transform, q, n):
        self.transform = transform
        self.q = q
        self.n = n

    def degree(self, index):
        count = 0
        coordinates = []
        coordinate_values = []
        for i in range(len(index)):
            if index[i] != 0:
                count += 1
                coordinates.append(i)
                coordinate_values.append(index[i])
        return count, coordinates, coordinate_values


    def get_permutations(self, coordinates):
        """
        Get all possible combinations of coordinates up to length of input coordinates.

        Parameters:
            coordinates (list): List of coordinate indices to generate combinations from.

        Returns:
            list: List of tuples containing all possible combinations of coordinates.
        """
        
        result = []
        for r in range(1, len(coordinates) + 1):
            perms = combinations(coordinates, r)
            result.extend(set(perms))
        return result

    def get_permutations_of_length_n(self, tuple_elements, q=None):
        """
        Get all possible q-ary permutations of a given tuple of elements.

        Parameters:
            tuple_elements (tuple): Tuple of coordinate indices to generate permutations for.
            q (int, optional): Base of q-ary system. If not provided, uses self.q.

        Returns:
            list: List of tuples containing all possible q-ary permutations of the input coordinates.
                 Each element in the permutations is in range [1, q-1].
        """
        if q is not None:
            perms = list(product(range(1, q), repeat=len(tuple_elements)))
        else:
            perms = list(product(range(1, self.q), repeat=len(tuple_elements)))
        return perms

    def get_permutations_of_lower_orders(self, coordinates, order):
        """
        Get all possible combinations of coordinates up to a specified order.

        Parameters:
            coordinates (list): List of coordinate indices to generate combinations from.
            order (int): Maximum order of combinations to generate.

        Returns:
            list: List of tuples containing all possible combinations of coordinates up to the specified order.
                 Each tuple represents a subset of the input coordinates.
        """
        result = []
        for r in range(1, order):
            perms = combinations(coordinates, r)
            result.extend(set(perms))
        return result


    def get_permutations_of_order(self, coordinates, order):
        """
        Get all possible combinations of coordinates of a specific order.

        Parameters:
            coordinates (list): List of coordinate indices to generate combinations from.
            order (int): The exact order of combinations to generate.

        Returns:
            list: List of tuples containing all possible combinations of coordinates of the specified order.
                 Each tuple represents a subset of the input coordinates of length equal to order.
        """
        result = []
        perms = combinations(coordinates, order)
        result.extend(set(perms))
        return result


    def run_mobius_transform(self):
        """
        Run the Mobius transform on the q-ary Fourier transform.
        
        The Mobius transform converts the Fourier coefficients into localized interaction effects.
        It iteratively builds up interaction terms by combining lower-order terms.
        
        The transform is stored in self.mobius_tf, which maps tuples of q-ary indices to complex values.
        The keys are tuples of length n (number of features) where non-zero elements indicate interactions.
        
        Returns:
            None
        
        Side Effects:
            Sets self.mobius_tf with the computed Mobius transform
        """
        omega = cmath.exp(2 * cmath.pi * 1j / self.q)
        data = self.transform

        # Sort transform by degree
        data = dict(sorted(data.items(), key=lambda x: sum(1 for i in x[0] if i != 0)))

        zeros = [0 for _ in range(self.n)]
        all_mobius_tfs = []

        for key, value in zip(data.keys(), data.values()):
            mobius_tf_perkey = dict()
            deg, coordinates, coordinate_vals = self.degree(key)

            # Update the 0th term
            mobius_tf_perkey[tuple(zeros)] = mobius_tf_perkey.get(tuple(zeros), 0) + value

            # Update all other interactions
            if deg > 0:
                all_interactions = self.get_permutations(coordinates)
                for interaction in all_interactions:
                    curr_key = zeros.copy()
                    mobius_nonzeros = self.get_permutations_of_length_n(interaction)
                    for mobius_nonzero in mobius_nonzeros:
                        for coord, iter_val in zip(interaction, mobius_nonzero):
                            curr_key[coord] = iter_val
                        k_i = key[coord]
                        iterative_key = curr_key.copy()
                        y_i = iterative_key[coord]
                        iterative_key[coord] = 0
                        mobius_tf_perkey[tuple(curr_key)] = mobius_tf_perkey.get(tuple(iterative_key), 0) * (omega ** (k_i * y_i) - 1)

            all_mobius_tfs.append(mobius_tf_perkey)

        # Aggregate all terms
        self.mobius_tf = {}
        for d in all_mobius_tfs:
            for key, value in d.items():
                self.mobius_tf[key] = self.mobius_tf.get(key, 0) + value


    def get_mobius_tf(self):
        return self.mobius_tf
    

    def localize_sample(self, sample):
        """
        Given a qsft transform, run the Mobius transform such that the transform is localized to the sample.
        Parameters:
            sample: numpy array
                The sample to localize the transform around. Each element should be in [0, q-1].
        
        Returns:
            localized_mobius_tf: dict
                The localized Mobius transform, where the all-zeros vector represents the sample without mutations. The other entries represent 
                arbitrary numbers that can be converted back to the original encodings.
                Keys are tuples representing positions, values are the transform coefficients.
            
            localized_mobius_tf_encoding: dict 
                The localized Mobius transform with keys expressed in terms of the original q-ary encodings.
                Keys are tuples representing positions, values are the transform coefficients.
        """
        omega = np.exp(2j * np.pi / self.q)
        w_d_k = omega ** (sample @ np.array(list(self.transform.keys())).T)    
        F_k = np.multiply(list(self.transform.values()), w_d_k)
        delayed_transform = dict(zip(list(self.transform.keys()), F_k))
        mobius_transform = shapzero(delayed_transform, q=self.q, n=self.n)
        mobius_transform.run_mobius_transform()
        localized_mobius_tf = mobius_transform.get_mobius_tf()

        # Convert the localized mobius_tf to represent the original encodings
        localized_mobius_tf_encoding = {}
        for key, value in localized_mobius_tf.items():
            delayed_key = tuple((key + np.array(sample)) % self.q)
            localized_mobius_tf_encoding[delayed_key] = value

        return localized_mobius_tf, localized_mobius_tf_encoding
    

    def explain(self, sample, explanation='shap_value'):
        """
        Explain a sample using the sparse mobius transform via shap values or shap interactions
        Returns a list of dictionaries containing the contributions per position
        """
        if explanation == 'shap_value':
            interactions = self.explain_shap_value(sample)
        elif explanation == 'faith_shap':
            interactions = self.explain_faith_shap_interaction(sample)
        else:
            raise ValueError(f"Explanation method {explanation} not supported")

        return interactions


    def compute_shap_value(self, localized_mobius_tf):
        """
        Given a localized mobius transform, compute the weighted average contribution of mobius coefficients for each position
        """
        interactions = {}
        for key, value in localized_mobius_tf.items():
            mobius_nonzeros = np.nonzero(key)[0]
            if mobius_nonzeros.size: # Check that the key is not all zeros
                interaction_order = mobius_nonzeros.size
                for nonzero in mobius_nonzeros:
                    interactions[tuple([nonzero])] = interactions.get(tuple([nonzero]), 0) - (1/ (interaction_order * self.q**(interaction_order)) ) * np.real(value)

        return interactions
    

    def compute_faith_shap_interactions(self, localized_mobius_tf):
        """
        Given a localized mobius transform, compute the shap interactions for each position
        """
        interactions = {}
        for key, value in localized_mobius_tf.items():
            mobius_nonzeros = np.nonzero(key)[0]
            if mobius_nonzeros.size: # Check that the key is not all zeros
                k = mobius_nonzeros.size
                interaction_permutations = self.get_permutations(mobius_nonzeros) # get all permutations of mobius interactions
                for interaction in interaction_permutations:
                    interaction_order = len(interaction)
                    interactions[tuple(interaction)] = interactions.get(tuple(interaction), 0) + ((-1) ** interaction_order) * (1/ ( self.q**k) ) * np.real(value)

        return interactions
    

    def explain_shap_value(self, sample):
        """
        Given a sample, compute SHAP values using the mobius transform
        """
        localized_mobius_tf, _ = self.localize_sample(sample)
        interactions = self.compute_shap_value(localized_mobius_tf)

        return interactions
    

    def explain_faith_shap_interaction(self, sample):
        """
        Given a sample, compute Faith-Shap interactions using the mobius transform
        """
        localized_mobius_tf, _ = self.localize_sample(sample)
        interactions = self.compute_faith_shap_interactions(localized_mobius_tf)

        return interactions
    
    
    def compute_scores(self, x, mean=0):
        """
        Compute the scores for samples using the mobius transform
        Input: Encoded q-ary sample. Mean value of the sample (if not zero)
        Output: Score given the mobius transform
        """
        xM_value = 0
        zeros = [0 for _ in range(self.n)]

        for key in self.mobius_tf.keys():
            mobius_nonzeros = np.nonzero(key)[0]

            # Check if mobius should be added
            if mobius_nonzeros.size:
                matches = True
                for coord in mobius_nonzeros:
                    if x[coord] != key[coord]:
                        matches = False
                        break
                if matches: 
                    xM_value += self.mobius_tf.get(tuple(key))

        xM_value += self.mobius_tf.get(tuple(zeros), 0) 
        xM_value += mean

        return xM_value
    

def plot_shap_values(ax, sequences, shap_values, colors=None, markers=None, x_label='Sequence position', y_label='SHAP value', y_limits=None, font_size=6, markersize=0.25, legend=True, legend_marker_size=3, linewidth=0.25):
    """
    Plots SHAP values for target sequences on the provided axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot on.
        sequences (list): List of sequences as strings.
        shap_values (np.ndarray): SHAP values for the sequences.
        colors (dict, optional): Dictionary mapping nucleotides as strings of letters to colors. If none specified, use default colors.
        font_size (int, optional): Font size for labels.
        markersize (int, optional): Marker size for scatter points.
    """

    if colors == None:
        colors = {'A': '#008000', 'C': '#0000ff', 'G': '#ffa600', 'T': '#ff0000'}
    if markers == None:
        markers = {
            'A': 'o',  
            'C': 's',  
            'G': 'd',  
            'T': '^',  
        }
    marker_size_adjustments = {'^': 1, 'd': 1}

    seq_length = np.shape(shap_values)[1]

    for seq, sample in zip(sequences, shap_values):

        seq = list(seq)
        color = []
        marker = []
        for nt in seq:
            color.append(colors.get(nt, 'k'))
            marker.append(markers.get(nt, 'o'))
        for i, (x, m, c) in enumerate(zip(range(1, seq_length + 1), marker, color)):
            ax.scatter(x, sample[i], color=c, marker=m, s=markersize, alpha=0.6)
    
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size, width=linewidth)
    ax.set_xlim(0.5, seq_length + 0.5)
    tick_positions = [i for i in range(1, seq_length + 1) if i % 5 == 0]
    tick_labels = [str(i) for i in tick_positions]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    if y_limits is not None:
        ax.set_ylim(y_limits)

    if legend:
        labels = list(colors.keys())
        handles = []
        for label in labels:
            handles.append(plt.Line2D([0], [0], marker=markers.get(label, 'o'), color='w', markerfacecolor=colors.get(label), markersize=legend_marker_size + marker_size_adjustments.get(markers.get(label, 'o'), 0)))
        legend_plt = ax.legend(handles, labels, loc='upper right', framealpha=1, ncol=len(labels), handletextpad=0.1, columnspacing=0.5, fontsize=font_size)
        legend_plt.get_frame().set_linewidth(linewidth)


def top_shap_values(sequences, shap_values, top_values=10, top_values_filename=None, all_values_filename=None):
    """
    Returns the top SHAP values.

    Parameters:
        sequences (list): List of sequences as strings.
        shap_values (np.ndarray): SHAP values for the sequences.
        colors (dict, optional): Dictionary mapping nucleotides as strings of letters to colors. If none specified, use default colors.
        top_values (int, optional): Number of top interactions to print.
        filename (str, optional): Name of the file to save the top interactions to as a csv.
    """
    sequences_list = sequences.copy()
    sequences = [list(seq) for seq in sequences]
    seq_length = len(sequences[0])
    all_interactions_positive = {}
    all_interactions_positive_count = {}
    all_interactions_negative = {}
    all_interactions_negative_count = {}
    for (shap, sequence) in zip(shap_values, sequences):
        for pos, (value, nuc) in enumerate(zip(shap, sequence)):
            nuc_pos = tuple([pos, nuc])
            if value > 0:
                all_interactions_positive[nuc_pos] = all_interactions_positive.get(nuc_pos, 0) + value
                all_interactions_positive_count[nuc_pos] = all_interactions_positive_count.get(nuc_pos, 0) + 1
            elif value < 0:
                all_interactions_negative[nuc_pos] = all_interactions_negative.get(nuc_pos, 0) + value
                all_interactions_negative_count[nuc_pos] = all_interactions_negative_count.get(nuc_pos, 0) + 1

    interaction_values_average_positive = {}
    interaction_values_average_negative = {}
    for key, value in all_interactions_positive.items():
        interaction_values_average_positive[key] = value / all_interactions_positive_count[key]
    for key, value in all_interactions_negative.items():
        interaction_values_average_negative[key] = value / all_interactions_negative_count[key]

    interaction_values_positive = dict(sorted(interaction_values_average_positive.items(), key=lambda item: np.abs(item[1]), reverse=True))
    interaction_values_negative = dict(sorted(interaction_values_average_negative.items(), key=lambda item: np.abs(item[1]), reverse=True))
    top_positive = dict(list(interaction_values_positive.items())[:top_values])
    top_negative = dict(list(interaction_values_negative.items())[:top_values])

    print(f"Top positive interactions:")
    for key, value in top_positive.items():
        print(f"{key}: {value}")
    print(f"Top negative interactions:")
    for key, value in top_negative.items():
        print(f"{key}: {value}")
    
    # Save to CSV:
    if top_values_filename is not None:
        data = []
        for key, value in top_positive.items():
            position, feature = key
            data.append(['Positive', position, feature, value])
        for key, value in top_negative.items():
            position, feature = key
            data.append(['Negative', position, feature, value])
        df = pd.DataFrame(data, columns=['Sign', 'Position', 'Feature', 'Average value'])
        df.to_csv(f'{top_values_filename}.csv', index=False)

    # Save all SHAP values to a CSV
    if all_values_filename is not None:
        df = pd.DataFrame(sequences_list, columns=['Sequence'])
        shap_columns = [f'Position {i+1}' for i in range(shap_values.shape[1])]
        shap_df = pd.DataFrame(shap_values, columns=shap_columns)
        final_df = pd.concat([df, shap_df], axis=1)
        final_df.to_csv(f'{all_values_filename}.csv', index=False)


def plot_interactions(ax, sequences, shap_interactions, top_values=None, colors=None, markers=None, x_label='Sequence position', y_label='SHAP value', y_limits=None, font_size=6, markersize=0.25, legend=True, legend_marker_size=3, linewidth=0.25):
    """
    Plots the top interactions on the provided axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot on.
        sequences (list): List of sequences as strings.
        shap_interactions (list of dict): SHAP interaction values for each sample. Keys are a tuple of positions, values are the associated SHAP interaction values.
        top_values (int, optional): Number of samples to plot.
        colors (dict, optional): Dictionary mapping nucleotides as strings of letters to colors. If none specified, use default colors.
        font_size (int, optional): Font size for labels.
        markersize (int, optional): Marker size for scatter points.
    """
    sequences = [list(seq) for seq in sequences]
    seq_length = len(sequences[0])

    if colors == None:
        colors = {'A': '#008000', 'C': '#0000ff', 'G': '#ffa600', 'T': '#ff0000'}
    if markers == None:
        markers = {
            'A': 'o',  
            'C': 's',  
            'G': 'd',  
            'T': '^',  
        }
    marker_size_adjustments = {'^': 1, 'd': 1}

    # Only plot interactions that will be visible on a plot, determined by epsilon
    interactions = {}
    seen_pairs = {}
    epsilon = 0.001
    for i, sample in enumerate(shap_interactions):
        for key, value in sample.items():
            pair_key = tuple(list(key))

            if pair_key in seen_pairs:
                all_seen_values = np.array(seen_pairs[pair_key])
                subtracted_values = np.abs(all_seen_values - value)
                smallest_value = np.min(subtracted_values)
                flag = smallest_value >= epsilon
            else:
                flag = True

            if flag:
                if pair_key not in seen_pairs:
                    seen_pairs[pair_key] = [value]
                else:
                    seen_pairs[pair_key].append(value)
                full_key = tuple(list(key) + [i])
                interactions[full_key] = interactions.get(full_key, 0) + np.real(value)
  
    # Sort and select top values
    if top_values is not None:
        interactions = dict(sorted(interactions.items(), key=lambda item: np.abs(item[1]), reverse=True))
        top_items = dict(list(interactions.items())[:top_values])
    else:
        top_items = interactions

    # Plot the top interactions
    for i, (key, value) in enumerate(top_items.items()):
        x_positions = [index + 1 for index in key[:-1]]  # Adjust +1 for plotting
        y_position = value

        if len(x_positions) == 1:
            # Plot first order interactions as scatter points
            nt = sequences[key[-1]][x_positions[0] - 1]
            color = colors.get(nt, 'k')
            marker = markers.get(nt, '.')
            ax.scatter(x_positions, y_position, c=color, marker=marker, s=markersize)
        else:
            # Plot higher order points by connecting them with lines
            ax.plot(x_positions, [y_position] * len(x_positions), linestyle='-', color='#9b9ca0', zorder=0, linewidth=linewidth)
            for x in x_positions:
                nt = sequences[key[-1]][x - 1]
                color = colors.get(nt, 'k')
                marker = markers.get(nt, '.')
                ax.scatter(x, y_position, c=color, marker=marker, s=markersize)

    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size, width=linewidth)
    ax.set_xlim(0.5, seq_length + 0.5)
    tick_positions = [i for i in range(1, seq_length + 1) if i % 5 == 0]
    tick_labels = [str(i) for i in tick_positions]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    if y_limits is not None:
        ax.set_ylim(y_limits)

    if legend:
        labels = list(colors.keys())
        handles = []
        for label in labels:
            handles.append(plt.Line2D([0], [0], marker=markers.get(label, 'o'), color='w', markerfacecolor=colors.get(label), markersize=legend_marker_size + marker_size_adjustments.get(markers.get(label, 'o'), 0)))
        legend_plt = ax.legend(handles, labels, loc='upper right', framealpha=1, ncol=len(labels), handletextpad=0.1, columnspacing=0.5, fontsize=font_size)
        legend_plt.get_frame().set_linewidth(linewidth)


def plot_interactions_summary(ax, sequences, shap_interactions, min_order = 2, colors=None, x_label='Sequence position', y_label='SHAP interaction', y_limits=None, font_size=6, legend=True, legend_marker_size=3, linewidth=0.2, barwidth=0.4, print_ratio=None):
    """
    Plots a bar graph of the average contribution of interactions on the provided axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot on.
        sequences (list): List of sequences as strings.
        shap_interactions (list of dict): SHAP interaction values for each sample. Keys are a tuple of positions, values are the associated SHAP interaction values.
        min_order (int, optional): Filters out interactions with orders less than the specified value.
        colors (dict, optional): Dictionary mapping nucleotides as strings of letters to colors. If none specified, use default colors.
        font_size (int, optional): Font size for labels.
        markersize (int, optional): Marker size for scatter points.
        barwidth (int, optional): Bar width for plotting.
        print_ratio (float, optional): Prints the percentage of interactions for different models.
    """
    sequences = [list(seq) for seq in sequences]
    seq_length = len(sequences[0])

    if colors == None:
        colors = {'A': '#008000', 'C': '#0000ff', 'G': '#ffa600', 'T': '#ff0000'}

    # Condense higher order interactions per position
    interaction_values_positive = {}
    interaction_values_negative = {}
    for (shap_interaction, sequence) in zip(shap_interactions, sequences):
        shap_interactions_min_order = {key: value for key, value in shap_interaction.items() if sum(1 for element in key if element != 0) >= min_order}
        sequence_i = list(sequence)
        
        for key, value in shap_interactions_min_order.items():
            # Distribute interaction values evenly across the affected positions, making sure to keep track of the nucleotide. Denote whether the interaction is positive or negative
            for pos in key:
                if value > 0:
                    interaction_values_positive[tuple([pos, sequence_i[pos]])] = interaction_values_positive.get(tuple([pos, sequence_i[pos]]), 0) + (value / len(key))
                if value < 0:
                    interaction_values_negative[tuple([pos, sequence_i[pos]])] = interaction_values_negative.get(tuple([pos, sequence_i[pos]]), 0) + (value / len(key))

    interaction_values_average = {} # Count how many times each nucleotide is present for averaging
    for seq in sequences:
        for pos in range(seq_length):
            interaction_values_average[tuple([pos, seq[pos]])] = interaction_values_average.get(tuple([pos, seq[pos]]), 0) + 1

    # Get average contribution among all sequences
    interaction_values_positive = {key: interaction_values_positive[key] / interaction_values_average[key] for key in interaction_values_positive if key in interaction_values_average}
    interaction_values_negative = {key: interaction_values_negative[key] / interaction_values_average[key] for key in interaction_values_negative if key in interaction_values_average}
    plotting_keys = list(reversed(list(colors.keys())))
    grouped_positive = group_by_position(interaction_values_positive, seq_length, plotting_keys)
    grouped_negative = group_by_position(interaction_values_negative, seq_length, plotting_keys)

    if print_ratio == 'tiger':
        # Return ratio of interactions in the seed region
        seed_sum = 0
        total_sum = 0

        # Iterate through both lists
        for position, values in grouped_positive.items():
            if 6-1 <= position <= 12-1:
                seed_sum += sum(abs(value) for value in values.values())
            total_sum += sum(abs(value) for value in values.values())

        for position, values in grouped_negative.items():
            if 6-1 <= position <= 12-1:
                seed_sum += sum(abs(value) for value in values.values())
            total_sum += sum(abs(value) for value in values.values())

        # Calculate the ratio
        ratio = seed_sum / total_sum if total_sum != 0 else 0
        print('Percentage of interactions in the seed region: ', ratio * 100)
    elif print_ratio == 'inDelphi':
        # Return ratio of interactions +/- 3 away from the cutsite
        seed_sum = 0
        total_sum = 0

        # Iterate through both lists
        for position, values in grouped_positive.items():
            if 18-1 <= position <= 23-1:
                seed_sum += sum(abs(value) for value in values.values())
            total_sum += sum(abs(value) for value in values.values())

        for position, values in grouped_negative.items():
            if 18-1 <= position <= 23-1:
                seed_sum += sum(abs(value) for value in values.values())
            total_sum += sum(abs(value) for value in values.values())

        # Calculate the ratio
        ratio = seed_sum / total_sum if total_sum != 0 else 0
        print('Percentage of interactions +/- 3 away from the cutsite: ', ratio * 100)

    # Plotting
    bottom_positive = np.zeros(seq_length)
    bottom_negative = np.zeros(seq_length)
    # Negative values 
    for nucleotide in plotting_keys:
        negative_values = [grouped_negative[pos].get(nucleotide, 0) for pos in range(seq_length)]
        ax.bar(range(1, seq_length + 1), negative_values, bottom=bottom_negative, color=colors[nucleotide], label=f"{nucleotide} (negative)" if nucleotide == 'T' else "", width=barwidth)
        bottom_negative += negative_values  # Update bottom for the next nucleotide stack
    # Positive values
    for nucleotide in plotting_keys:
        positive_values = [grouped_positive[pos].get(nucleotide, 0) for pos in range(seq_length)]
        ax.bar(range(1, seq_length + 1), positive_values, bottom=bottom_positive, color=colors[nucleotide], label=f"{nucleotide} (positive)", width=barwidth)
        bottom_positive += positive_values  # Update bottom for the next nucleotide stack

    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    tick_positions = [i for i in range(1, seq_length + 1) if i % 5 == 0]
    tick_labels = [str(i) for i in tick_positions]
    ax.tick_params(axis='both', labelsize=font_size, width=linewidth)
    ax.set_xlim(0.5, seq_length + 0.5)
    ax.axhline(0, color='black', linewidth=linewidth)  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    if y_limits is not None:
        ax.set_ylim(y_limits)
    else:
        # Find the bounds for the y-axis
        sums_positive = {key: sum(subdict.values()) for key, subdict in grouped_positive.items()}
        max_key = max(sums_positive, key=sums_positive.get)
        max_value_positive = sums_positive[max_key]
        sums_negative = {key: sum(subdict.values()) for key, subdict in grouped_negative.items()}
        max_key = min(sums_negative, key=sums_negative.get)
        max_value_negative = sums_negative[max_key]
        max_y = max(max_value_positive, np.abs(max_value_negative))
        ax.set_ylim((-max_y - 0.1*max_y, max_y + 0.1*max_y))

    if legend:
        labels = list(colors.keys())
        handles = []
        for label in labels:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors.get(label), markersize=legend_marker_size))
        legend_plt = ax.legend(handles, labels, loc='upper right', framealpha=1, ncol=len(labels), handletextpad=0.1, columnspacing=0.5, fontsize=font_size)
        legend_plt.get_frame().set_linewidth(linewidth)


def top_interactions(sequences, shap_interactions, top_values=10, top_interactions_filename=None, all_interactions_filename=None, min_order=2):
    """
    Returns and optionally saves the top SHAP interactions.

    Parameters:
        sequences (list): List of sequences as strings.
        shap_interactions (list of dict): SHAP interaction values for each sample.
        top_values (int, optional): Number of top interactions to print.
        top_interactions_filename (str, optional): Filename to save top interactions as CSV.
        all_interactions_filename (str, optional): Filename to save all interactions as CSV.
        min_order (int, optional): Minimum interaction order to consider.
    """
    sequences = [list(seq) for seq in sequences]
    
    # Track positive and negative interactions
    pos_interactions = {}
    pos_counts = {}
    neg_interactions = {}
    neg_counts = {}

    # Aggregate interactions across sequences
    for seq, interactions in zip(sequences, shap_interactions):
        filtered_interactions = {k: v for k, v in interactions.items() 
                              if sum(1 for x in k if x != 0) >= min_order}
        
        for positions, value in filtered_interactions.items():
            nucleotides = tuple(seq[pos] for pos in positions)
            key = (nucleotides, positions)
            
            if value > 0:
                pos_interactions[key] = pos_interactions.get(key, 0) + value
                pos_counts[key] = pos_counts.get(key, 0) + 1
            elif value < 0:
                neg_interactions[key] = neg_interactions.get(key, 0) + value 
                neg_counts[key] = neg_counts.get(key, 0) + 1

    # Calculate averages
    avg_pos = {k: v / pos_counts[k] for k, v in pos_interactions.items()}
    avg_neg = {k: v / neg_counts[k] for k, v in neg_interactions.items()}

    # Get top interactions
    top_pos = dict(sorted(avg_pos.items(), key=lambda x: abs(x[1]), reverse=True)[:top_values])
    top_neg = dict(sorted(avg_neg.items(), key=lambda x: abs(x[1]), reverse=True)[:top_values])

    # Print results
    print("Top positive interactions:")
    for key, value in top_pos.items():
        print(f"{key}: {value}")
    print("\nTop negative interactions:")
    for key, value in top_neg.items():
        print(f"{key}: {value}")

    # Save top interactions if filename provided
    if top_interactions_filename:
        data = []
        for sign, interactions in [("Positive", top_pos), ("Negative", top_neg)]:
            for (features, positions), value in interactions.items():
                data.append([
                    sign,
                    ', '.join(map(str, positions)),
                    ', '.join(features),
                    value
                ])
        pd.DataFrame(data, columns=['Sign', 'Position', 'Feature', 'Average value'])\
          .to_csv(f'{top_interactions_filename}.csv', index=False)

    # Save all interactions if filename provided
    if all_interactions_filename:
        _save_all_interactions(sequences, shap_interactions, all_interactions_filename)

def _save_all_interactions(sequences, shap_interactions, filename, chunk_size=500):
    """Helper function to save all interactions to CSV"""
    max_order = max(len(key) for key in shap_interactions[0].keys())
    
    def split_dict(d):
        keys = list(d.keys())
        return [{k: d[k] for k in keys[i:i + chunk_size]}
                for i in range(0, len(keys), chunk_size)]

    data = []
    for sequence, interactions in zip(sequences, shap_interactions):
        row = [sequence]
        for order in range(1, max_order + 1):
            order_interactions = {k: v for k, v in interactions.items() if len(k) == order}
            if len(order_interactions) > chunk_size:
                row.extend(split_dict(order_interactions))
            else:
                row.append(order_interactions)
        data.append(row)

    # Create column names
    columns = ['Sequence']
    for order in range(1, max_order + 1):
        max_chunks = max(len(split_dict({k: v for k, v in inter.items() if len(k) == order}))
                        for inter in shap_interactions)
        if max_chunks > 1:
            columns.extend(f'Order {order} part {i + 1}' for i in range(max_chunks))
        else:
            columns.append(f'Order {order}')

    pd.DataFrame(data, columns=columns).to_csv(f'{filename}.csv', index=False)


def group_by_position(data, seq_length, nucleotides):
    grouped = {pos: {nuc: 0 for nuc in nucleotides} for pos in range(seq_length)}
    for (position, nucleotide), value in data.items():
        grouped[position][nucleotide] = value
    return grouped


def correlation_shap_values(shapzero, shap):
    """
    Pearson correlation of SHAP values
    """
    shapzero = shapzero.flatten()
    shap = shap.flatten()
    pearson_corr, _ = pearsonr(shapzero, shap)
    print(f'SHAP pearson correlation: {pearson_corr:.2f}')


def correlation_interactions(shapzero, shapzero_sequences, interactions, interactions_sequences, encoding={0:'A', 1:'C', 2:'T', 3:'G'}):
    """
    Pearson correlation of interactions
    """
    reverse_encoding = {v: k for k, v in encoding.items()}
    interactions_sequences_encoding = [[reverse_encoding[num] for num in row] for row in interactions_sequences]
    shapzero_sequences_encoding = [[reverse_encoding[num] for num in row] for row in shapzero_sequences]

    all_interactions = 0
    all_shapzero_interactions = 0
    # Only correlate interactions that are shared
    interactions_corr = {}
    interactions_shapzero_corr = {}

    for i, (wt, sample) in enumerate(zip(interactions_sequences_encoding, interactions)):

        # Find which interactions in shap match up with shapzero
        wt = np.array(wt)
        row_index = np.where(np.all(shapzero_sequences_encoding == wt, axis=1))[0][0]
        sample_shapzero = shapzero[row_index]

        # Keep track how many interactions there are
        all_interactions += len(list(sample.values()))
        all_shapzero_interactions += len(list(sample_shapzero.values()))

        for key, _ in sample_shapzero.items():
            # Check to make sure the interaction is shared between mobius and shap, then add it to both dictionaries
            if tuple(list(key)) in sample.keys():
                interactions_corr[tuple(list(key) + [i])] = sample.get(tuple(list(key)))
                interactions_shapzero_corr[tuple(list(key) + [i])] = sample_shapzero.get(tuple(list(key)))

    shapzero = list(interactions_shapzero_corr.values())
    interactions = list(interactions_corr.values())
    pearson_corr, _ = pearsonr(shapzero, interactions)
    print(f'Interactions pearson correlation: {pearson_corr:.2f}')
    print('Fraction of interactions shared with respect to SHAP zero: ', len(shapzero) / all_shapzero_interactions)
    print('Fraction of interactions shared with respect to SHAP-IQ: ', len(shapzero) / all_interactions)



def plot_shap_values_protein(
    ax,
    sequences,
    shap_values,
    aa_dict=None,
    x_label='Position',
    y_label='Amino Acid',
    font_size=6,
    color_bounds=None,
    cmap='bwr',
    colorbar=True,
):
    """
    Plots a heatmap of average SHAP values per amino acid and sequence position on the provided axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot on.
        sequences (list of str): Amino acid sequences (list of strings, each of length = seq_length).
        shap_values (np.ndarray): SHAP values corresponding to sequences (shape: [num_samples, seq_length]).
        aa_dict (dict, optional): Dictionary mapping amino acid letters to integers. Default is standard 20 amino acids.
        x_label (str, optional): Label for x-axis.
        y_label (str, optional): Label for y-axis.
        font_size (int, optional): Font size for labels and ticks.
        color_bounds (tuple, optional): (vmin, vmax) bounds for color scale. If None, determined automatically.
        cmap (str, optional): Colormap to use.
        colorbar (bool, optional): Whether to add a colorbar.
    """

    if aa_dict is None:
        aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}

    index_to_aa = {v: k for k, v in aa_dict.items()}
    aa_order = [aa for aa, _ in sorted(aa_dict.items(), key=lambda x: x[1])]

    num_aas = len(aa_order)
    seq_len = shap_values.shape[1]

    # Build heatmap array
    heatmap = np.full((num_aas, seq_len), np.nan)

    for aa_idx in range(num_aas):
        for pos in range(seq_len):
            # Find sequences where the amino acid at this position matches
            mask = np.array([seq[pos] == index_to_aa[aa_idx] for seq in sequences])
            if np.any(mask):
                heatmap[aa_idx, pos] = np.mean(shap_values[mask, pos])

    # Determine vmin, vmax
    if color_bounds is not None:
        vmin, vmax = color_bounds
    else:
        vmin = np.nanmin(heatmap)
        vmax = np.nanmax(heatmap)

    # Plot heatmap
    im = ax.imshow(
        heatmap,
        aspect='equal',  # Square cells
        cmap=cmap,
        interpolation='nearest',
        vmin=vmin,
        vmax=vmax
    )

    # Labels and ticks
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.set_xticks(np.arange(seq_len))
    ax.set_xticklabels([str(i+1) for i in range(seq_len)], fontsize=font_size)
    ax.set_yticks(np.arange(num_aas))
    ax.set_yticklabels(aa_order, fontsize=font_size)
    ax.tick_params(axis='both', which='both', width=0.25)

    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Optional colorbar
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Average SHAP value', fontsize=font_size, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=font_size)

    return im, (vmin, vmax)



import numpy as np
import matplotlib.pyplot as plt

# def plot_shap_values_protein_top_mutations(
#     ax,
#     sequences,
#     shap_values,
#     aa_dict=None,
#     x_label='Position',
#     y_label='Amino Acid',
#     font_size=6,
#     percent=10,
#     colorbar=False,
# ):
#     """
#     Plots a binary heatmap showing only the top X% positive and negative SHAP values.
#     Positive values shown in red, negative values in blue, rest in white.

#     Parameters:
#         ax (matplotlib.axes.Axes): The axis to plot on.
#         sequences (list of str): Amino acid sequences (list of strings).
#         shap_values (np.ndarray): SHAP values for the sequences (shape: [num_samples, seq_length]).
#         aa_dict (dict, optional): Mapping from amino acid letters to indices. Defaults to standard.
#         x_label (str, optional): Label for x-axis.
#         y_label (str, optional): Label for y-axis.
#         font_size (int, optional): Font size for axis labels and ticks.
#         percent (float, optional): Percent of SHAP values (positive and negative) to highlight.
#         colorbar (bool, optional): Whether to include a colorbar (not needed here usually).
#     """

#     if aa_dict is None:
#         aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
#                    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}

#     index_to_aa = {v: k for k, v in aa_dict.items()}
#     aa_order = [aa for aa, _ in sorted(aa_dict.items(), key=lambda x: x[1])]

#     num_aas = len(aa_order)
#     seq_len = shap_values.shape[1]

#     # Compute full SHAP mean heatmap
#     heatmap = np.full((num_aas, seq_len), np.nan)
#     for aa_idx in range(num_aas):
#         for pos in range(seq_len):
#             mask = np.array([seq[pos] == index_to_aa[aa_idx] for seq in sequences])
#             if np.any(mask):
#                 heatmap[aa_idx, pos] = np.mean(shap_values[mask, pos])

#     # Flatten and get thresholds
#     flattened = heatmap[~np.isnan(heatmap)]
#     num_values = len(flattened)
#     num_to_keep = max(1, int((percent / 100) * num_values))

#     # Compute thresholds
#     sorted_vals = np.sort(flattened)
#     print(sum(sorted_vals > 0))
#     print(sum(sorted_vals < 0))
#     neg_thresh = sorted_vals[num_to_keep - 1]
#     pos_thresh = sorted_vals[-num_to_keep]
#     print(neg_thresh)
#     print(sorted_vals)

#     # Build binary mask: -1 (blue), 0 (white), +1 (red)
#     binary_mask = np.full_like(heatmap, 0)
#     binary_mask[heatmap <= neg_thresh] = -1
#     binary_mask[heatmap >= pos_thresh] = 1
#     binary_mask[np.isnan(heatmap)] = np.nan

#     # Define a colormap from bwr: white (0), blue (-1), red (+1)
#     from matplotlib.colors import ListedColormap, BoundaryNorm
#     cmap = ListedColormap(['blue', 'white', 'red'])
#     norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

#     # Plot
#     im = ax.imshow(
#         binary_mask,
#         cmap=cmap,
#         norm=norm,
#         aspect='equal',
#         interpolation='nearest'
#     )

#     # Axis setup
#     ax.set_xlabel(x_label, fontsize=font_size)
#     ax.set_ylabel(y_label, fontsize=font_size)
#     ax.set_xticks(np.arange(seq_len))
#     ax.set_xticklabels([str(i + 1) for i in range(seq_len)], fontsize=font_size)
#     ax.set_yticks(np.arange(num_aas))
#     ax.set_yticklabels(aa_order, fontsize=font_size)
#     ax.tick_params(axis='both', which='both', width=0.25)
#     ax.invert_yaxis()
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     # Optional colorbar (not really needed)
#     if colorbar:
#         cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1], fraction=0.046, pad=0.04)
#         cbar.ax.set_yticklabels(['Top Neg', 'None', 'Top Pos'])
#         cbar.set_label('Top SHAP', fontsize=font_size)
#         cbar.ax.tick_params(labelsize=font_size)

#     return im


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# def plot_shap_values_protein_top_mutations(
#     ax,
#     sequences,
#     shap_values,
#     aa_dict=None,
#     x_label='Position',
#     y_label='Amino Acid',
#     font_size=6,
#     percent=15,
#     colorbar=False,
#     min_number=10,
# ):
#     """
#     Plots a binary heatmap showing only the top X% positive and top X% negative SHAP values.
#     Positive values shown in red, negative values in blue, rest in white.

#     Parameters:
#         ax (matplotlib.axes.Axes): The axis to plot on.
#         sequences (list of str): Amino acid sequences (list of strings).
#         shap_values (np.ndarray): SHAP values for the sequences (shape: [num_samples, seq_length]).
#         aa_dict (dict, optional): Mapping from amino acid letters to indices. Defaults to standard.
#         x_label (str, optional): Label for x-axis.
#         y_label (str, optional): Label for y-axis.
#         font_size (int, optional): Font size for axis labels and ticks.
#         percent (float, optional): Percent of SHAP values (positive and negative) to highlight.
#         colorbar (bool, optional): Whether to include a colorbar (not needed here usually).
#         min_number (int, optional): Minimum number of positive/negative SHAP values to show.
#     """

#     if aa_dict is None:
#         aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
#                    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}

#     index_to_aa = {v: k for k, v in aa_dict.items()}
#     aa_order = [aa for aa, _ in sorted(aa_dict.items(), key=lambda x: x[1])]
#     num_aas = len(aa_order)
#     seq_len = shap_values.shape[1]

#     # Compute SHAP mean heatmap
#     heatmap = np.full((num_aas, seq_len), np.nan)
#     for aa_idx in range(num_aas):
#         for pos in range(seq_len):
#             mask = np.array([seq[pos] == index_to_aa[aa_idx] for seq in sequences])
#             if np.any(mask):
#                 heatmap[aa_idx, pos] = np.mean(shap_values[mask, pos])

#     # Compute thresholds separately for pos and neg
#     flat_vals = heatmap[~np.isnan(heatmap)]

#     pos_vals = flat_vals[flat_vals > 0]
#     neg_vals = flat_vals[flat_vals < 0]

#     num_pos = max(min_number, int((percent / 100) * len(pos_vals)))
#     num_neg = max(min_number, int((percent / 100) * len(neg_vals)))

#     pos_thresh = np.sort(pos_vals)[-num_pos] if len(pos_vals) > 0 else np.inf
#     neg_thresh = np.sort(neg_vals)[:num_neg][-1] if len(neg_vals) > 0 else -np.inf

#     # Binary mask: 1 = red (top pos), -1 = blue (top neg), 0 = white
#     binary_mask = np.full_like(heatmap, 0)
#     binary_mask[heatmap >= pos_thresh] = 1
#     binary_mask[heatmap <= neg_thresh] = -1
#     binary_mask[np.isnan(heatmap)] = np.nan

#     # Use bwr colors: [blue, white, red]
#     cmap = ListedColormap(['blue', 'white', 'red'])
#     norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

#     # Plot
#     im = ax.imshow(binary_mask, cmap=cmap, norm=norm, aspect='equal', interpolation='nearest')

#     ax.set_xlabel(x_label, fontsize=font_size)
#     ax.set_ylabel(y_label, fontsize=font_size)
#     ax.set_xticks(np.arange(seq_len))
#     ax.set_xticklabels([str(i + 1) for i in range(seq_len)], fontsize=font_size)
#     ax.set_yticks(np.arange(num_aas))
#     ax.set_yticklabels(aa_order, fontsize=font_size)
#     ax.tick_params(axis='both', which='both', width=0.25)
#     ax.invert_yaxis()
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     # Optional colorbar
#     if colorbar:
#         cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1], fraction=0.046, pad=0.04)
#         cbar.ax.set_yticklabels(['Top Neg', 'None', 'Top Pos'])
#         cbar.set_label('Top SHAP', fontsize=font_size)
#         cbar.ax.tick_params(labelsize=font_size)

#     return im


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def plot_shap_values_protein_top_mutations(
    ax,
    sequences,
    shap_values,
    aa_dict=None,
    x_label='Amino Acid',
    y_label='Position',
    font_size=6,
    percent=20,
    colorbar=True,
    colorbar_label='Average SHAP value'
):
    """
    Plots a gradient heatmap showing only the top X% positive and top X% negative SHAP values.
    Amino acids are on the x-axis, positions are on the y-axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot on.
        sequences (list of str): Amino acid sequences (list of strings).
        shap_values (np.ndarray): SHAP values for the sequences (shape: [num_samples, seq_length]).
        aa_dict (dict, optional): Mapping from amino acid letters to indices. Defaults to standard.
        x_label (str, optional): Label for x-axis.
        y_label (str, optional): Label for y-axis.
        font_size (int, optional): Font size for axis labels and ticks.
        percent (float, optional): Percent of SHAP values (positive and negative) to highlight.
        colorbar (bool, optional): Whether to include a colorbar.
    """

    if aa_dict is None:
        aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}

    index_to_aa = {v: k for k, v in aa_dict.items()}
    aa_order = [aa for aa, _ in sorted(aa_dict.items(), key=lambda x: x[1])]
    num_aas = len(aa_order)
    seq_len = shap_values.shape[1]

    # Compute SHAP mean heatmap
    heatmap = np.full((num_aas, seq_len), np.nan)
    for aa_idx in range(num_aas):
        for pos in range(seq_len):
            mask = np.array([seq[pos] == index_to_aa[aa_idx] for seq in sequences])
            if np.any(mask):
                heatmap[aa_idx, pos] = np.mean(shap_values[mask, pos])

    # Compute thresholds separately for pos and neg
    flat_vals = heatmap[~np.isnan(heatmap)]
    pos_vals = flat_vals[flat_vals > 0]
    neg_vals = flat_vals[flat_vals < 0]

    num_pos = max(10, int((percent / 100) * len(pos_vals)))
    num_neg = max(10, int((percent / 100) * len(neg_vals)))

    pos_thresh = np.sort(pos_vals)[-num_pos] if len(pos_vals) > 0 else np.inf
    neg_thresh = np.sort(neg_vals)[:num_neg][-1] if len(neg_vals) > 0 else -np.inf

    # Keep only top mutations
    masked_heatmap = np.full_like(heatmap, np.nan)
    keep_mask = (heatmap >= pos_thresh) | (heatmap <= neg_thresh)
    masked_heatmap[keep_mask] = heatmap[keep_mask]

    # Flip axes: transpose and adjust colormap
    flipped_heatmap = masked_heatmap.T  # Now shape: (positions, amino acids)

    # Color normalization
    max_pos = np.nanmax(flipped_heatmap)
    min_neg = np.nanmin(flipped_heatmap)
    norm = Normalize(vmin=min_neg, vmax=max_pos, clip=True)

    # Plot heatmap
    im = ax.imshow(
        flipped_heatmap,
        cmap='bwr',
        norm=norm,
        aspect='equal',
        interpolation='nearest'
    )

    # Axis formatting
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.set_xticks(np.arange(num_aas))
    ax.set_xticklabels(aa_order, fontsize=font_size - 1)
    ax.set_yticks(np.arange(seq_len))
    ax.set_yticklabels([str(i + 1) for i in range(seq_len)], fontsize=font_size - 1)
    ax.tick_params(axis='both', which='both', width=0.25)
    # ax.invert_yaxis()  # Optional: put position 1 at top
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Colorbar
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label, fontsize=font_size, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=font_size - 1)

    return im


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# import matplotlib.patches as patches

# def plot_top_shap_interactions_grid(
#     ax,
#     sequences,
#     shap_interactions,
#     top_k=10,
#     font_size=6,
#     cmap='bwr',
#     colorbar=True,
#     min_order=2
# ):
#     """
#     Plots top-k SHAP interactions on a grid. Each row is one interaction. Cells show amino acids and are colored by interaction value.

#     Parameters:
#         ax (matplotlib.axes.Axes): Axis to plot on.
#         sequences (list of str): Amino acid sequences.
#         shap_interactions (list of dict): SHAP interaction dicts for each sample.
#         top_k (int): Number of top interactions to show.
#         font_size (int): Base font size; will be adjusted to fit cells.
#         cmap (str): Colormap ('bwr' recommended).
#         colorbar (bool): Whether to show a colorbar.
#         min_order (int): Minimum interaction order to consider.
#     """
#     from collections import defaultdict

#     seq_length = len(sequences[0])
#     sequences = [list(seq) for seq in sequences]

#     # Aggregate interaction values
#     pos_interactions = defaultdict(float)
#     neg_interactions = defaultdict(float)
#     pos_counts = defaultdict(int)
#     neg_counts = defaultdict(int)

#     for seq, interaction_dict in zip(sequences, shap_interactions):
#         filtered = {k: v for k, v in interaction_dict.items() if len(k) >= min_order}
#         for positions, value in filtered.items():
#             aas = tuple(seq[pos] for pos in positions)
#             key = (aas, positions)
#             if value > 0:
#                 pos_interactions[key] += value
#                 pos_counts[key] += 1
#             elif value < 0:
#                 neg_interactions[key] += value
#                 neg_counts[key] += 1

#     # Average interactions
#     avg_pos = {k: v / pos_counts[k] for k, v in pos_interactions.items()}
#     avg_neg = {k: v / neg_counts[k] for k, v in neg_interactions.items()}
#     all_avg = {**avg_pos, **avg_neg}
#     top_interactions = dict(sorted(all_avg.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k])

#     # Build grid
#     grid_vals = np.full((top_k, seq_length), np.nan)
#     grid_labels = np.full((top_k, seq_length), '', dtype=object)

#     for i, ((aas, positions), value) in enumerate(top_interactions.items()):
#         for aa, pos in zip(aas, positions):
#             grid_vals[i, pos] = value
#             grid_labels[i, pos] = aa

#     # Color normalization
#     max_val = np.nanmax(np.abs(grid_vals))
#     norm = Normalize(vmin=-max_val, vmax=max_val)

#     # Plot heatmap
#     im = ax.imshow(grid_vals, cmap=cmap, norm=norm, aspect='equal', interpolation='nearest')

#     # Add text, adjust font to fit
#     cell_height = 1
#     cell_width = 1
#     adjusted_font_size = min(font_size, 0.4 * 72 / max(cell_height, cell_width))  # convert to pt

#     for i in range(top_k):
#         for j in range(seq_length):
#             label = grid_labels[i, j]
#             if label:
#                 ax.text(j, i, label, ha='center', va='center', fontsize=adjusted_font_size, color='black')

#     # Draw grid borders
#     for i in range(top_k + 1):
#         ax.axhline(i - 0.5, color='black', linewidth=0.5)
#     for j in range(seq_length + 1):
#         ax.axvline(j - 0.5, color='black', linewidth=0.5)

#     # Axis formatting
#     ax.set_xticks(np.arange(seq_length))
#     ax.set_xticklabels([str(i + 1) for i in range(seq_length)], fontsize=font_size, rotation=90)
#     ax.set_yticks(np.arange(top_k))
#     ax.set_yticklabels([f"{i+1}" for i in range(top_k)], fontsize=font_size)
#     ax.set_xlabel("Position", fontsize=font_size)
#     ax.set_ylabel("Top Interaction", fontsize=font_size)
#     ax.tick_params(axis='both', which='both', width=0.25, length=0)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)

#     if colorbar:
#         cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#         cbar.set_label('SHAP Interaction Value', fontsize=font_size)
#         cbar.ax.tick_params(labelsize=font_size)

#     return im


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize

# def plot_pairwise_interaction_heatmap(
#     ax,
#     sequences,
#     shap_interactions,
#     font_size=6,
#     cmap='bwr',
#     colorbar=True
# ):
#     """
#     Plots an L x L symmetric heatmap of average 2nd-order SHAP interactions.

#     Parameters:
#         ax (matplotlib.axes.Axes): Axis to draw on.
#         sequences (list of str): List of amino acid sequences.
#         shap_interactions (list of dict): SHAP interaction dicts per sample.
#         font_size (int): Font size for ticks and labels.
#         cmap (str): Colormap (e.g. 'bwr').
#         colorbar (bool): Whether to include a colorbar.
#     """
#     L = len(sequences[0])
#     interaction_matrix = np.zeros((L, L))
#     count_matrix = np.zeros((L, L))

#     for interaction_dict in shap_interactions:
#         for positions, value in interaction_dict.items():
#             if len(positions) == 2:
#                 i, j = positions
#                 if i == j:
#                     continue  # skip diagonal (self-interaction)
#                 interaction_matrix[i, j] += value
#                 interaction_matrix[j, i] += value  # symmetric entry
#                 count_matrix[i, j] += 1
#                 count_matrix[j, i] += 1  # symmetric entry

#     # Avoid division by zero
#     with np.errstate(invalid='ignore', divide='ignore'):
#         avg_matrix = np.where(count_matrix > 0, interaction_matrix / count_matrix, np.nan)

#     # Optional: clear diagonal
#     np.fill_diagonal(avg_matrix, np.nan)

#     # Normalize colors symmetrically
#     max_val = np.nanmax(np.abs(avg_matrix))
#     norm = Normalize(vmin=-max_val, vmax=max_val)

#     # Plot
#     im = ax.imshow(avg_matrix, cmap=cmap, norm=norm, aspect='equal', interpolation='nearest')

#     ax.set_xticks(np.arange(L))
#     ax.set_xticklabels([str(i+1) for i in range(L)], fontsize=font_size, rotation=90)
#     ax.set_yticks(np.arange(L))
#     ax.set_yticklabels([str(i+1) for i in range(L)], fontsize=font_size)
#     ax.set_xlabel("Position", fontsize=font_size)
#     ax.set_ylabel("Position", fontsize=font_size)
#     ax.tick_params(axis='both', which='both', width=0.25)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     if colorbar:
#         cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#         cbar.set_label("Avg SHAP Interaction", fontsize=font_size)
#         cbar.ax.tick_params(labelsize=font_size)

#     return im


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import Normalize

# def plot_pairwise_interaction_heatmap(
#     ax,
#     sequences,
#     shap_interactions,
#     aa_order=None,
#     font_size=6,
#     percent=15,
#     colorbar=False
# ):
#     """
#     Plots a 20x20 binary heatmap of average pairwise SHAP interactions between amino acids.

#     Parameters:
#         ax (matplotlib.axes.Axes): Axis to draw on.
#         sequences (list of str): List of amino acid sequences.
#         shap_interactions (list of dict): SHAP interaction dicts per sample.
#         aa_order (list of str): Order of amino acids (default: alphabetical 20 standard AAs).
#         font_size (int): Font size for labels.
#         percent (float): Top % of positive/negative interactions to include.
#         colorbar (bool): Whether to include a colorbar.
#     """
#     if aa_order is None:
#         aa_order = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
#                     'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
#     aa_index = {aa: i for i, aa in enumerate(aa_order)}
#     n_aa = len(aa_order)

#     interaction_matrix = np.zeros((n_aa, n_aa))
#     count_matrix = np.zeros((n_aa, n_aa))

#     for seq, interaction_dict in zip(sequences, shap_interactions):
#         for positions, value in interaction_dict.items():
#             if len(positions) == 2:
#                 i, j = positions
#                 if i == j:
#                     continue
#                 aa1, aa2 = seq[i], seq[j]
#                 if aa1 in aa_index and aa2 in aa_index:
#                     idx1, idx2 = aa_index[aa1], aa_index[aa2]
#                     # Symmetrize by contributing to both (i,j) and (j,i)
#                     interaction_matrix[idx1, idx2] += value
#                     interaction_matrix[idx2, idx1] += value
#                     count_matrix[idx1, idx2] += 1
#                     count_matrix[idx2, idx1] += 1

#     with np.errstate(divide='ignore', invalid='ignore'):
#         avg_matrix = np.where(count_matrix > 0, interaction_matrix / count_matrix, np.nan)

#     np.fill_diagonal(avg_matrix, np.nan)

#     # Get top X% pos and neg
#     flat = avg_matrix[~np.isnan(avg_matrix)]
#     pos_vals = flat[flat > 0]
#     neg_vals = flat[flat < 0]

#     num_pos = max(1, int(len(pos_vals) * percent / 100))
#     num_neg = max(1, int(len(neg_vals) * percent / 100))

#     pos_thresh = np.sort(pos_vals)[-num_pos] if len(pos_vals) else np.inf
#     neg_thresh = np.sort(neg_vals)[:num_neg][-1] if len(neg_vals) else -np.inf

#     # Build binary matrix: -1 = top neg (blue), 1 = top pos (red), 0 = everything else
#     binary_matrix = np.full_like(avg_matrix, 0)
#     binary_matrix[avg_matrix >= pos_thresh] = 1
#     binary_matrix[avg_matrix <= neg_thresh] = -1
#     binary_matrix[np.isnan(avg_matrix)] = np.nan

#     # Color map
#     cmap = ListedColormap(['blue', 'white', 'red'])
#     norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], ncolors=cmap.N)

#     im = ax.imshow(binary_matrix, cmap=cmap, norm=norm, aspect='equal', interpolation='nearest')

#     # Label axes
#     ax.set_xticks(np.arange(n_aa))
#     ax.set_xticklabels(aa_order, fontsize=font_size, rotation=90)
#     ax.set_yticks(np.arange(n_aa))
#     ax.set_yticklabels(aa_order, fontsize=font_size)
#     ax.set_xlabel("AA at position j", fontsize=font_size)
#     ax.set_ylabel("AA at position i", fontsize=font_size)
#     ax.tick_params(axis='both', which='both', width=0.25)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     if colorbar:
#         cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1], fraction=0.046, pad=0.04)
#         cbar.ax.set_yticklabels(['Top Neg', 'None', 'Top Pos'])
#         cbar.set_label("AA Interaction SHAP (top %)", fontsize=font_size)
#         cbar.ax.tick_params(labelsize=font_size)

#     return im



def plot_pairwise_interaction_heatmap(
    ax,
    sequences,
    shap_interactions,
    aa_order=None,
    font_size=6,
    cmap='bwr',
    colorbar=True,
    percent=20,
    colorbar_label='Average SHAP interaction'
):
    """
    Plots a 20x20 gradient heatmap of top X% average pairwise SHAP interactions between amino acids.

    Parameters:
        ax (matplotlib.axes.Axes): Axis to draw on.
        sequences (list of str): Amino acid sequences.
        shap_interactions (list of dict): SHAP interaction dicts per sample.
        aa_order (list of str): Order of amino acids (default = 20 standard AAs).
        font_size (int): Font size for axis labels.
        cmap (str): Colormap for SHAP values (e.g. 'bwr').
        colorbar (bool): Whether to include a colorbar.
        percent (float): Percent of top positive and negative values to display.
    """
    if aa_order is None:
        aa_order = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    aa_index = {aa: i for i, aa in enumerate(aa_order)}
    n_aa = len(aa_order)

    interaction_matrix = np.zeros((n_aa, n_aa))
    count_matrix = np.zeros((n_aa, n_aa))

    for seq, interaction_dict in zip(sequences, shap_interactions):
        for positions, value in interaction_dict.items():
            if len(positions) == 2:
                i, j = positions
                if i == j:
                    continue
                aa1, aa2 = seq[i], seq[j]
                if aa1 in aa_index and aa2 in aa_index:
                    idx1, idx2 = aa_index[aa1], aa_index[aa2]
                    interaction_matrix[idx1, idx2] += value
                    interaction_matrix[idx2, idx1] += value
                    count_matrix[idx1, idx2] += 1
                    count_matrix[idx2, idx1] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_matrix = np.where(count_matrix > 0, interaction_matrix / count_matrix, np.nan)

    np.fill_diagonal(avg_matrix, np.nan)

    # Mask everything except top X% of positive/negative values
    flat_vals = avg_matrix[~np.isnan(avg_matrix)]
    pos_vals = flat_vals[flat_vals > 0]
    neg_vals = flat_vals[flat_vals < 0]

    num_pos = max(1, int((percent / 100) * len(pos_vals)))
    num_neg = max(1, int((percent / 100) * len(neg_vals)))

    pos_thresh = np.sort(pos_vals)[-num_pos] if len(pos_vals) > 0 else np.inf
    neg_thresh = np.sort(neg_vals)[:num_neg][-1] if len(neg_vals) > 0 else -np.inf

    masked_matrix = np.full_like(avg_matrix, np.nan)
    keep_mask = (avg_matrix >= pos_thresh) | (avg_matrix <= neg_thresh)
    masked_matrix[keep_mask] = avg_matrix[keep_mask]

    # Color normalization (asymmetric)
    vmax = np.nanmax(masked_matrix)
    vmin = np.nanmin(masked_matrix)
    if vmax <= 0: vmax = 1e-8
    if vmin >= 0: vmin = -1e-8
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    # Plot heatmap
    im = ax.imshow(masked_matrix, cmap=cmap, norm=norm, aspect='equal', interpolation='nearest')

    # Axis labels
    ax.set_xticks(np.arange(n_aa))
    ax.set_xticklabels(aa_order, fontsize=font_size - 1)#, rotation=90)
    ax.set_yticks(np.arange(n_aa))
    ax.set_yticklabels(aa_order, fontsize=font_size - 1)
    ax.set_xlabel("Amino acid", fontsize=font_size)
    ax.set_ylabel("Amino acid", fontsize=font_size)
    ax.tick_params(axis='both', which='both', width=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label, fontsize=font_size, rotation=270, labelpad=8)
        cbar.ax.tick_params(labelsize=font_size - 1)

    return im