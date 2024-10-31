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

    pd.DataFrame(data, columns=columns).to_csv(filename, index=False)


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


def correlation_interactions(shapzero, shapzero_sequences, interactions, interactions_sequences):
    """
    Pearson correlation of interactions
    """
    encoding = {0:'A', 1:'C', 2:'T', 3:'G'}
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
