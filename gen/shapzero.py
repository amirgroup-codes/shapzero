import pickle
import cmath
from itertools import combinations, product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
        result = []
        for r in range(1, len(coordinates) + 1):
            perms = combinations(coordinates, r)
            result.extend(set(perms))
        return result
    # # Example 
    # coordinates = [1, 2, 3]
    # permutations_list = get_permutations(coordinates)
    # for perm in permutations_list:
    #     print(perm)
    # (1,)
    # (2,)
    # (3,)
    # (2, 3)
    # (1, 2)
    # (1, 3)
    # (1, 2, 3)


    def get_permutations_of_length_n(self, tuple_elements, q=None):
        if q is not None:
            perms = list(product(range(1, q), repeat=len(tuple_elements)))
        else:
            perms = list(product(range(1, self.q), repeat=len(tuple_elements)))
        return perms
    # # Example
    # tuple_elements = (1,)
    # q = 4
    # permutations_list = get_permutations_of_length_n(tuple_elements, q)
    # for perm in permutations_list:
    #     print(perm)
    # (1,)
    # (2,)
    # (3,)


    def get_permutations_of_lower_orders(self, coordinates, order):
        result = []
        for r in range(1, order):
            perms = combinations(coordinates, r)
            result.extend(set(perms))
        return result


    def get_permutations_of_order(self, coordinates, order):
        result = []
        perms = combinations(coordinates, order)
        result.extend(set(perms))
        return result


    def run_mobius_transform(self):
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
        Given a qsft transform, run the Mobius transform such that the transform is localized to the sample
        Outputs:
            localized_mobius_tf: localized mobius transform, where the all 0's vector represents the sample without mutations. The other entries represent arbitrary numbers that can be converted back to the original encodings
            localized_mobius_tf_encoding: localized mobius transform, the key indices are expressed in terms of the original q-ary encodings
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
    

def plot_shap_values(ax, sequences, shap_values, colors=None, markers=None, x_label='Sequence position', y_label='SHAP value', y_limits=None, font_size=5, markersize=0.25, legend=True, legend_marker_size = 4, linewidth=0.25):
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


def top_shap_values(sequences, shap_values, top_values=10):
    """
    Returns the top SHAP values.

    Parameters:
        sequences (list): List of sequences as strings.
        shap_values (np.ndarray): SHAP values for the sequences.
        colors (dict, optional): Dictionary mapping nucleotides as strings of letters to colors. If none specified, use default colors.
        top_values (int, optional): Number of top interactions to print.
    """
    sequences = [list(seq) for seq in sequences]
    seq_length = len(sequences[0])
#  for key, value in shap_interactions_min_order.items():
#             nucleotides = [sequence_i[pos] for pos in key]
#             nuc_pos = tuple((tuple(nucleotides), key))
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
                all_interactions_negative_count[nuc_pos] = all_interactions_positive_count.get(nuc_pos, 0) + 1

    interaction_values_average_positive = {}
    interaction_values_average_negative = {}
    for key, value in all_interactions_positive.items():
        interaction_values_average_positive[key] = value / all_interactions_positive_count[key]
    for key, value in all_interactions_negative.items():
        interaction_values_average_negative[key] = value / all_interactions_negative_count[key]
    # interaction_values_average = {} # Count how many times each nucleotide is present for averaging
    # for seq in sequences:
    #     for pos in range(seq_length):
    #         interaction_values_average[tuple([pos, seq[pos]])] = interaction_values_average.get(tuple([pos, seq[pos]]), 0) + 1
    # print(all_interactions_positive)
    # print(interaction_values_average)
    # Get average contribution among all sequences
    # interaction_values_positive = {key: all_interactions_positive[key] / interaction_values_average[key] for key in all_interactions_positive if key in interaction_values_average}
    # interaction_values_negative = {key: all_interactions_negative[key] / interaction_values_average[key] for key in all_interactions_negative if key in interaction_values_average}

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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_interactions(ax, sequences, shap_interactions, top_values=None, colors=None, markers=None, x_label='Sequence position', y_label='SHAP value', y_limits=None, font_size=5, markersize=0.25, legend=True, legend_marker_size=3, linewidth=0.25):
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
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors.get(label), markersize=legend_marker_size + marker_size_adjustments.get(markers.get(label, 'o'), 0)))
        legend_plt = ax.legend(handles, labels, loc='upper right', framealpha=1, ncol=len(labels), handletextpad=0.1, columnspacing=0.5, fontsize=font_size)
        legend_plt.get_frame().set_linewidth(linewidth)


def plot_interactions_summary(ax, sequences, shap_interactions, min_order = 2, colors=None, x_label='Sequence position', y_label='SHAP interaction', y_limits=None, font_size=5, markersize=0.25, legend=True, legend_marker_size=3, linewidth=0.2, barwidth=0.4, print_top_interactions=False, top_values=10):
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
        print_top_interactions (bool, optional): If true, prints the top 10 interactions.
        top_interactions (int, optional): Number of top interactions to print if print_top_interactions is true.
    """
    sequences = [list(seq) for seq in sequences]
    seq_length = len(sequences[0])

    if colors == None:
        colors = {'A': '#008000', 'C': '#0000ff', 'G': '#ffa600', 'T': '#ff0000'}

    # Condense higher order interactions per position
    interaction_values_positive = {}
    interaction_values_negative = {}
    # For printing
    all_interactions_positive = {}
    all_interactions_positive_count = {}
    all_interactions_negative = {}
    all_interactions_negative_count = {}
    for (shap_interaction, sequence) in zip(shap_interactions, sequences):
        shap_interactions_min_order = {key: value for key, value in shap_interaction.items() if sum(1 for element in key if element != 0) >= min_order}
        sequence_i = list(sequence)
        
        for key, value in shap_interactions_min_order.items():
            nucleotides = [sequence_i[pos] for pos in key]
            nuc_pos = tuple((tuple(nucleotides), key))
            if value > 0:
                all_interactions_positive[nuc_pos] = all_interactions_positive.get(nuc_pos, 0) + value
                all_interactions_positive_count[nuc_pos] = all_interactions_positive_count.get(nuc_pos, 0) + 1
            elif value < 0:
                all_interactions_negative[nuc_pos] = all_interactions_negative.get(nuc_pos, 0) + value
                all_interactions_negative_count[nuc_pos] = all_interactions_negative_count.get(nuc_pos, 0) + 1
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

    # For printing
    interaction_values_average_positive = {}
    interaction_values_average_negative = {}
    for key, value in all_interactions_positive.items():
        interaction_values_average_positive[key] = value / all_interactions_positive_count[key]
    for key, value in all_interactions_negative.items():
        interaction_values_average_negative[key] = value / all_interactions_negative_count[key]

    # Get average contribution among all sequences
    interaction_values_positive = {key: interaction_values_positive[key] / interaction_values_average[key] for key in interaction_values_positive if key in interaction_values_average}
    interaction_values_negative = {key: interaction_values_negative[key] / interaction_values_average[key] for key in interaction_values_negative if key in interaction_values_average}
    plotting_keys = list(reversed(list(colors.keys())))
    grouped_positive = group_by_position(interaction_values_positive, seq_length, plotting_keys)
    grouped_negative = group_by_position(interaction_values_negative, seq_length, plotting_keys)

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

    if print_top_interactions:
        interaction_values_average_positive = dict(sorted(interaction_values_average_positive.items(), key=lambda item: np.abs(item[1]), reverse=True))
        interaction_values_average_negative = dict(sorted(interaction_values_average_negative.items(), key=lambda item: np.abs(item[1]), reverse=True))
        top_positive = dict(list(interaction_values_average_positive.items())[:top_values])
        top_negative = dict(list(interaction_values_average_negative.items())[:top_values])

        print(f"Top positive interactions:")
        for key, value in top_positive.items():
            print(f"{key}: {value}")
        print(f"Top negative interactions:")
        for key, value in top_negative.items():
            print(f"{key}: {value}")


def group_by_position(data, seq_length, nucleotides):
    grouped = {pos: {nuc: 0 for nuc in nucleotides} for pos in range(seq_length)}
    for (position, nucleotide), value in data.items():
        grouped[position][nucleotide] = value
    return grouped
