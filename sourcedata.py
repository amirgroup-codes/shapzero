"""
Script to compile source data from the paper
"""
import pandas as pd
import numpy as np
import sys
from gen.shapzero import group_by_position
np.random.seed(42)
import random
random.seed(42)

def str2numpy(string):
    string = string.strip('[]')
    str_list = string.split(',')
    float_list = [float(num) for num in str_list]
    arr = np.array(float_list)
    return arr

def process_interactions_summary(sequences, shap_interactions, min_order = 2, method='SHAP zero'):
    """Code borrowed from plot_interactions_summary"""
    sequences = [list(seq) for seq in sequences]
    seq_length = len(sequences[0])

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
    plotting_keys = list((list(colors.keys())))
    grouped_positive = group_by_position(interaction_values_positive, seq_length, plotting_keys)
    grouped_negative = group_by_position(interaction_values_negative, seq_length, plotting_keys)


    data = []
    for position, values in grouped_positive.items():
        for nucleotide, value in values.items():
            data.append({'Position': position + 1, 'Nucleotide': nucleotide, 'Sign': 'Positive', f'{method} interaction': value})

    # Process negative values
    for position, values in grouped_negative.items():
        for nucleotide, value in values.items():
            data.append({'Position': position + 1, 'Nucleotide': nucleotide, 'Sign': 'Negative', f'{method} interaction': value})

    # Convert the data into a DataFrame
    plot_data_df = pd.DataFrame(data)

    return plot_data_df

def process_interactions(sequences, shap_interactions, epsilon = 0.001, top_values=80, method='SHAP zero'):
    sequences = [list(seq) for seq in sequences]

    # Initialize interactions and tracking structures
    interactions = {}
    seen_pairs = {}

    for i, sample in enumerate(shap_interactions):
        for key, value in sample.items():
            pair_key = tuple(list(key))

            # Check epsilon to filter similar interactions
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

    # Create DataFrame structure
    interaction_data = []
    for key, value in top_items.items():
        positions = tuple(key[:-1])  # Positions as tuple
        sample_idx = key[-1]  # Sample index
        nucleotides = tuple(sequences[sample_idx][pos - 1] for pos in positions)  # Nucleotides as tuple
        interaction_data.append((positions, nucleotides, value))

    # Convert to DataFrame
    df = pd.DataFrame(interaction_data, columns=['Positions', 'Nucleotides', f'{method} interaction'])
    return df


df_master = pd.DataFrame()

"""Figure 2"""
# d) and f)
property = 'on_target' 
top_values_shap = 50

df_heldout = pd.read_csv(f'TIGER/figures/data/{property}_heldout.csv')
num_samples = range(1, 1100)
shapzero_sample_time = np.load(f'TIGER/figures/shap_results/time_per_sample_{property}.npy') 
shapzero_sample_time = shapzero_sample_time * 3981312 # Number of samples needed for q-sft
shapzero_shap_time = np.load(f'TIGER/figures/shap_results/time_shapzero_fsi_{property}.npy')
overall_shapzero_time_interactions = [shapzero_sample_time + shapzero_shap_time * i for i in num_samples]
shap_time = np.load(f'TIGER/figures/shap_results/time_fsi_{property}.npy')
overall_shap_time_interactions = [shap_time * i for i in num_samples]

shap_time = np.load(f'TIGER/figures/shap_results/time_shap_values_{property}.npy')
overall_shap_time_values = [shap_time * i for i in num_samples]
shapzero_shap_time = np.load(f'TIGER/figures/shap_results/time_shapzero_values_{property}.npy')
overall_shapzero_time_values = [shapzero_sample_time + shapzero_shap_time * i for i in num_samples]

df_2d = pd.DataFrame({
    "Number of samples": num_samples,
    "KernelSHAP total runtime (seconds)": overall_shap_time_values,
    "SHAP zero values total runtime (seconds)": overall_shapzero_time_values
})
df_2f = pd.DataFrame({
    "Number of samples": num_samples,
    "SHAP-IQ total runtime (seconds)": overall_shap_time_interactions,
    "SHAP zero interactions total runtime (seconds)": overall_shapzero_time_interactions
})

# c)
shap = np.load(f'TIGER/figures/shap_results/shap_values_{property}.npy')
random_indices = random.sample(range(0, len(shap)), top_values_shap)

seq_length = 26
shap_values = np.zeros((len(shap), seq_length))
for i, shap_val in enumerate(shap):
    # Loop through all positions in the sequence and calculate the SHAP value for each position for each guide:target pairing
    for pos in range(seq_length): 
        shap_values[i, pos] = shap_val[pos]

sequences = df_heldout['Full Sequence'].values
sequences_trimmed = sequences[random_indices]
shap_values_trimmed_kernelshap = [shap_values[i,:] for i in random_indices] 
shap_zero = np.load(f'TIGER/figures/shap_results/shapzero_values_{property}.npy')
shap_zero_trimmed = [shap_zero[i,:] for i in random_indices]

df = pd.DataFrame(sequences_trimmed, columns=['Sequence'])
kernelshap_columns = [f'KernelSHAP value Position {i+1}' for i in range(len(shap_values_trimmed_kernelshap[0]))]
shapzero_columns = [f'SHAP zero value Position {i+1}' for i in range(len(shap_zero_trimmed[0]))]
shap_df = pd.DataFrame(shap_values_trimmed_kernelshap, columns=kernelshap_columns)
shapzero_df = pd.DataFrame(shap_zero_trimmed, columns=shapzero_columns)
df_2c = pd.concat([df, shap_df, shapzero_df], axis=1)

# e)
shapzero_interactions = np.load(f'TIGER/figures/shap_results/shapzero_fsi_{property}.npz', allow_pickle=True)
shapzero_interactions_sequences = shapzero_interactions['interactions_sequences']
shapzero_sequences = shapzero_interactions['sequences']
encoding = {0:'A', 1:'C', 2:'T', 3:'G'}
shapzero_sequences = [''.join(encoding[num] for num in row) for row in shapzero_sequences]
df_2e_shapzero = process_interactions_summary(sequences, shapzero_interactions_sequences, min_order=2, method='SHAP zero')

x_valid = np.load(f'TIGER/figures/shap_results/fsi_{property}_samples.npy')
sequences = [''.join(encoding[num] for num in row) for row in x_valid]
shap_interactions = np.load(f'TIGER/figures/shap_results/fsi_{property}.pickle', allow_pickle=True)
df_2e_shapiq = process_interactions_summary(sequences, shap_interactions, min_order=2, method='SHAP-IQ')
df_2e = pd.merge(
    df_2e_shapzero[['Position', 'Nucleotide', 'Sign', 'SHAP zero interaction']],
    df_2e_shapiq[['Position', 'Nucleotide', 'Sign', 'SHAP-IQ interaction']],
    on=['Position', 'Nucleotide', 'Sign'],
    how='inner'
)



"""Figure 3"""
# d) and f)
property = 'frameshift'
celltype = 'HEK293'
df_heldout = pd.read_csv(f'inDelphi/figures/data/{celltype}_heldout.csv')
df_extra = pd.read_csv(f'inDelphi/figures/data/extra_shap_samples_{celltype}.csv')
df_extra_2 = pd.read_csv(f'inDelphi/figures/data/extra_shap_samples_{celltype}_x2.csv')

num_samples = range(1, 500)
shapzero_sample_time = np.load(f'inDelphi/figures/shap_results/time_per_sample_{celltype}.npy') 
shapzero_sample_time = shapzero_sample_time * 6045696 # Number of samples needed for q-sft
shapzero_shap_time = (np.load(f'inDelphi/figures/shap_results/time_shapzero_fsi_{celltype}_{property}.npy') * len(df_heldout) + np.load(f'inDelphi/figures/shap_results/time_shapzero_fsi_extra_{celltype}_{property}.npy') * len(df_extra) + np.load(f'inDelphi/figures/shap_results/time_shapzero_fsi_extra_{celltype}_{property}_x2.npy') * len(df_extra_2)) / (len(df_heldout) + len(df_extra) + len(df_extra_2))
overall_shapzero_time_interactions = [shapzero_sample_time + shapzero_shap_time * i for i in num_samples]

shap_time = np.load(f'inDelphi/figures/shap_results/time_fsi_{celltype}_{property}.npy')
overall_shap_time_interactions = [shap_time * i for i in num_samples]

shap_time = (np.load(f'inDelphi/figures/shap_results/time_shap_values_{celltype}.npy') * len(df_heldout) + np.load(f'inDelphi/figures/shap_results/time_shap_values_extra_{celltype}.npy') * len(df_extra) + np.load(f'inDelphi/figures/shap_results/time_shap_values_extra_{celltype}_x2.npy') * len(df_extra_2)) / (len(df_heldout) + len(df_extra) + len(df_extra_2))
overall_shap_time_values = [shap_time * i for i in num_samples]
shapzero_shap_time = (np.load(f'inDelphi/figures/shap_results/time_shapzero_values_{celltype}_{property}.npy') * len(df_heldout) + np.load(f'inDelphi/figures/shap_results/time_shapzero_values_extra_{celltype}_{property}.npy') * len(df_extra) +  np.load(f'inDelphi/figures/shap_results/time_shapzero_values_extra_{celltype}_{property}_x2.npy') * len(df_extra_2)) / (len(df_heldout) + len(df_extra) + len(df_extra_2))
overall_shapzero_time_values = [shapzero_sample_time + shapzero_shap_time * i for i in num_samples]

df_3d = pd.DataFrame({
    "Number of samples": num_samples,
    "KernelSHAP total runtime (seconds)": overall_shap_time_values,
    "SHAP zero values total runtime (seconds)": overall_shapzero_time_values
})
df_3f = pd.DataFrame({
    "Number of samples": num_samples,
    "SHAP-IQ total runtime (seconds)": overall_shap_time_interactions,
    "SHAP zero interactions total runtime (seconds)": overall_shapzero_time_interactions
})

# c)
shap = np.load(f'inDelphi/figures/shap_results/shap_values_{celltype}.npy')[0, :, :]
extra_shap = np.load(f'inDelphi/figures/shap_results/shap_values_extra_{celltype}.npy')[0, :, :]
extra_shap_x2 = np.load(f'inDelphi/figures/shap_results/shap_values_extra_{celltype}_x2.npy')[0, :, :] 
random.seed(42)
shap = np.concatenate((shap, extra_shap, extra_shap_x2), axis=0)
random_indices = random.sample(range(0, len(shap)), top_values_shap)

seq_length = 40
shap_values = np.zeros((len(shap), seq_length))
for i, shap_val in enumerate(shap):
    # Loop through all positions in the sequence and calculate the SHAP value for each position for each guide:target pairing
    for pos in range(seq_length): 
        shap_values[i, pos] = shap_val[pos]

sequences = df_heldout['Left_seq'].values + df_heldout['Right_seq'].values
sequences_extra = df_extra['Left_seq'].values + df_extra['Right_seq'].values
sequences_extra_2 = df_extra_2['Left_seq'].values + df_extra_2['Right_seq'].values
sequences = np.concatenate([sequences, sequences_extra, sequences_extra_2])
sequences_trimmed = sequences[random_indices]
shap_values_trimmed_kernelshap = [shap_values[i,:] for i in random_indices] 
shap_zero = np.load(f'inDelphi/figures/shap_results/shapzero_values_{celltype}_{property}.npy')
shap_zero_extra = np.load(f'inDelphi/figures/shap_results/shapzero_values_extra_{celltype}_{property}.npy')
shap_zero_extra_2 = np.load(f'inDelphi/figures/shap_results/shapzero_values_extra_{celltype}_{property}_x2.npy')
shap_zero = np.concatenate([shap_zero, shap_zero_extra, shap_zero_extra_2])
shap_zero_trimmed = [shap_zero[i,:] for i in random_indices]

df = pd.DataFrame(sequences_trimmed, columns=['Sequence'])
kernelshap_columns = [f'KernelSHAP value Position {i+1}' for i in range(len(shap_values_trimmed_kernelshap[0]))]
shapzero_columns = [f'SHAP zero value Position {i+1}' for i in range(len(shap_values_trimmed_kernelshap[0]))]
shap_df = pd.DataFrame(shap_values_trimmed_kernelshap, columns=kernelshap_columns)
shapzero_df = pd.DataFrame(shap_zero_trimmed, columns=shapzero_columns)
df_3c = pd.concat([df, shap_df, shapzero_df], axis=1)

# e)
shapzero_interactions = np.load(f'inDelphi/figures/shap_results/shapzero_fsi_{celltype}_{property}.npz', allow_pickle=True)
extra_shapzero_interactions = np.load(f'inDelphi/figures/shap_results/shapzero_fsi_extra_{celltype}_{property}.npz', allow_pickle=True)
extra_shapzero_interactions_x2 = np.load(f'inDelphi/figures/shap_results/shapzero_fsi_extra_{celltype}_{property}_x2.npz', allow_pickle=True)
shapzero_interactions_sequences = shapzero_interactions['interactions_sequences']
shapzero_interactions_sequences_extra = extra_shapzero_interactions['interactions_sequences']
shapzero_interactions_sequences_extra_x2 = extra_shapzero_interactions_x2['interactions_sequences']
shapzero_interactions_sequences = np.concatenate([shapzero_interactions_sequences, shapzero_interactions_sequences_extra, shapzero_interactions_sequences_extra_x2])
shapzero_sequences = shapzero_interactions['sequences']
shapzero_equences_extra = extra_shapzero_interactions['sequences']
shapzero_equences_extra_x2 = extra_shapzero_interactions_x2['sequences']
shapzero_sequences = np.concatenate([shapzero_sequences, shapzero_equences_extra, shapzero_equences_extra_x2])
encoding = {0:'A', 1:'C', 2:'T', 3:'G'}
shapzero_sequences = [''.join(encoding[num] for num in row) for row in shapzero_sequences]
df_3e_shapzero = process_interactions_summary(sequences, shapzero_interactions_sequences, min_order=2, method='SHAP zero')

x_valid = np.load(f'inDelphi/figures/shap_results/fsi_{celltype}_{property}_samples.npy')
sequences = [''.join(encoding[num] for num in row) for row in x_valid]
shap_interactions = np.load(f'inDelphi/figures/shap_results/fsi_{celltype}_{property}.pickle', allow_pickle=True)
df_3e_shapiq = process_interactions_summary(sequences, shap_interactions, min_order=2, method='SHAP-IQ')
df_3e = pd.merge(
    df_3e_shapzero[['Position', 'Nucleotide', 'Sign', 'SHAP zero interaction']],
    df_3e_shapiq[['Position', 'Nucleotide', 'Sign', 'SHAP-IQ interaction']],
    on=['Position', 'Nucleotide', 'Sign'],
    how='inner'
)



"""Figure S1"""
# a)
property = 'on_target'
seq_length = 26
shapzero_interactions = np.load(f'TIGER/figures/shap_results/shapzero_fsi_{property}.npz', allow_pickle=True)
shapzero_interactions_sequences = shapzero_interactions['interactions_sequences']
shapzero_sequences = shapzero_interactions['sequences']
encoding = {0:'A', 1:'C', 2:'T', 3:'G'}
sequences = [''.join(encoding[num] for num in row) for row in shapzero_sequences]
df_S1a = process_interactions(sequences, shapzero_interactions_sequences, top_values=80, method='SHAP zero')

# b)
x_valid = np.load(f'TIGER/figures/shap_results/fsi_{property}_samples.npy')
sequences = [''.join(encoding[num] for num in row) for row in x_valid]
shap_interactions = np.load(f'TIGER/figures/shap_results/fsi_{property}.pickle', allow_pickle=True)
df_S1b = process_interactions(sequences, shap_interactions, top_values=80, method='SHAP-IQ')



"""Figure S3"""
# a)
random.seed(42)
df_heldout = pd.read_csv(f'TIGER/figures/data/{property}_heldout.csv')
shap = np.load(f'TIGER/figures/shap_results/fastshap_{property}.npy')
random_indices = random.sample(range(0, len(shap)), top_values_shap)
shap_values = np.zeros((len(shap), seq_length))
for i, shap_val in enumerate(shap):
    # Loop through all positions in the sequence and calculate the SHAP value for each position for each guide:target pairing
    for pos in range(seq_length): 
        shap_values[i, pos] = shap_val[pos]

sequences = df_heldout['Full Sequence'].values
sequences_trimmed = sequences[random_indices]
shap_values_trimmed_fastshap = [shap_values[i,:] for i in random_indices]

df_shap = pd.read_csv(f'TIGER/figures/shap_results/deepshap_{property}.csv')
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

shap_values_trimmed_deepshap = [shap_values[i,:] for i in random_indices]

df = pd.DataFrame(sequences_trimmed, columns=['Sequence'])
fastshap_columns = [f'FastSHAP value Position {i+1}' for i in range(len(shap_values_trimmed_fastshap[0]))]
deepshap_columns = [f'DeepSHAP value Position {i+1}' for i in range(len(shap_values_trimmed_deepshap[0]))]
fastshap_df = pd.DataFrame(shap_values_trimmed_fastshap, columns=fastshap_columns)
deepshap_df = pd.DataFrame(shap_values_trimmed_deepshap, columns=deepshap_columns)
df_S3a = pd.concat([df, fastshap_df, deepshap_df], axis=1)

# b)
num_samples = range(1, 1100)
shapzero_sample_time = np.load(f'TIGER/figures/shap_results/time_per_sample_{property}.npy') 
shapzero_sample_time = shapzero_sample_time * 3981312 # Number of samples needed for q-sft
shapzero_shap_time = np.load(f'TIGER/figures/shap_results/time_shapzero_values_{property}.npy')
overall_shapzero_time = [shapzero_sample_time + shapzero_shap_time * i for i in num_samples]

# FastSHAP
fastshap_surrogate_time = np.load(f'TIGER/figures/shap_results/time_fastshap_surrogate_{property}.npy')
fastshap_explainer_time = np.load(f'TIGER/figures/shap_results/time_fastshap_explainer_{property}.npy')
fastshap_inference_time = np.load(f'TIGER/figures/shap_results/time_fastshap_inference_{property}.npy')
overall_fastshap_time = [fastshap_surrogate_time + fastshap_explainer_time + fastshap_inference_time * i for i in num_samples]

# DeepSHAP
deepshap_time = np.load(f'TIGER/figures/shap_results/time_deepshap_{property}.npy')
overall_deepshap_time = [deepshap_time * i for i in num_samples]

df_S3b = pd.DataFrame({
    "Number of samples": num_samples,
    "FastSHAP total runtime (seconds)": overall_fastshap_time,
    "DeepSHAP total runtime (seconds)": overall_deepshap_time,
    "SHAP zero values total runtime (seconds)": overall_shapzero_time
})



"""Figure S2"""
property = 'frameshift'
celltype = 'HEK293'
df_heldout = pd.read_csv(f'inDelphi/figures/data/{celltype}_heldout.csv')
df_extra = pd.read_csv(f'inDelphi/figures/data/extra_shap_samples_{celltype}.csv')
df_extra_2 = pd.read_csv(f'inDelphi/figures/data/extra_shap_samples_{celltype}_x2.csv')

shapzero_interactions = np.load(f'inDelphi/figures/shap_results/shapzero_fsi_{celltype}_{property}.npz', allow_pickle=True)
extra_shapzero_interactions = np.load(f'inDelphi/figures/shap_results/shapzero_fsi_extra_{celltype}_{property}.npz', allow_pickle=True)
extra_shapzero_interactions_x2 = np.load(f'inDelphi/figures/shap_results/shapzero_fsi_extra_{celltype}_{property}_x2.npz', allow_pickle=True)
shapzero_interactions_sequences = shapzero_interactions['interactions_sequences']
shapzero_interactions_sequences_extra = extra_shapzero_interactions['interactions_sequences']
shapzero_interactions_sequences_extra_x2 = extra_shapzero_interactions_x2['interactions_sequences']
shapzero_interactions_sequences = np.concatenate([shapzero_interactions_sequences, shapzero_interactions_sequences_extra, shapzero_interactions_sequences_extra_x2])
shapzero_sequences = shapzero_interactions['sequences']
shapzero_equences_extra = extra_shapzero_interactions['sequences']
shapzero_equences_extra_x2 = extra_shapzero_interactions_x2['sequences']
shapzero_sequences = np.concatenate([shapzero_sequences, shapzero_equences_extra, shapzero_equences_extra_x2])
sequences = [''.join(encoding[num] for num in row) for row in shapzero_sequences]
df_S2a = process_interactions(sequences, shapzero_interactions_sequences, top_values=80, method='SHAP zero')

x_valid = np.load(f'inDelphi/figures/shap_results/fsi_{celltype}_{property}_samples.npy')
sequences = [''.join(encoding[num] for num in row) for row in x_valid]
shap_interactions = np.load(f'inDelphi/figures/shap_results/fsi_{celltype}_{property}.pickle', allow_pickle=True)
df_S2b = process_interactions(sequences, shap_interactions, top_values=80, method='SHAP-IQ')



"""Figure S4"""
# a)
seq_length = 40
random.seed(42)
shap = np.load(f'inDelphi/figures/shap_results/fastshap_{celltype}_{property}.npy')
extra_shap = np.load(f'inDelphi/figures/shap_results/fastshap_extra_{celltype}_{property}.npy')
extra_shap_x2 = np.load(f'inDelphi/figures/shap_results/fastshap_extra_{celltype}_{property}_x2.npy')
shap = np.concatenate((shap, extra_shap, extra_shap_x2), axis=0)
random_indices = random.sample(range(0, len(shap)), top_values_shap)
shap_values = np.zeros((len(shap), seq_length))
for i, shap_val in enumerate(shap):
    for pos in range(seq_length): 
        shap_values[i, pos] = shap_val[pos]

sequences = df_heldout['Left_seq'].values + df_heldout['Right_seq'].values
sequences_extra = df_extra['Left_seq'].values + df_extra['Right_seq'].values
sequences_extra_2 = df_extra_2['Left_seq'].values + df_extra_2['Right_seq'].values
sequences = np.concatenate([sequences, sequences_extra, sequences_extra_2])
sequences_trimmed = sequences[random_indices]
shap_values_trimmed_fastshap = [shap_values[i,:] for i in random_indices]

df = pd.DataFrame(sequences_trimmed, columns=['Sequence'])
fastshap_columns = [f'FastSHAP value Position {i+1}' for i in range(len(shap_values_trimmed_fastshap[0]))]
fastshap_df = pd.DataFrame(shap_values_trimmed_fastshap, columns=fastshap_columns)
df_S4a = pd.concat([df, fastshap_df], axis=1)

# b)
num_samples = range(1, 500)
shapzero_sample_time = np.load(f'inDelphi/figures/shap_results/time_per_sample_{celltype}.npy') 
shapzero_sample_time = shapzero_sample_time * 6045696 # Number of samples needed for q-sft
shapzero_shap_time = (np.load(f'inDelphi/figures/shap_results/time_shapzero_values_{celltype}_{property}.npy') * len(df_heldout) + np.load(f'inDelphi/figures/shap_results/time_shapzero_values_extra_{celltype}_{property}.npy') * len(df_extra) +  np.load(f'inDelphi/figures/shap_results/time_shapzero_values_extra_{celltype}_{property}_x2.npy') * len(df_extra_2)) / (len(df_heldout) + len(df_extra) + len(df_extra_2))
overall_shapzero_time = [shapzero_sample_time + shapzero_shap_time * i for i in num_samples]

# FastSHAP
fastshap_surrogate_time = np.load(f'inDelphi/figures/shap_results/time_fastshap_surrogate_{celltype}_{property}.npy')
fastshap_explainer_time = np.load(f'inDelphi/figures/shap_results/time_fastshap_explainer_{celltype}_{property}.npy')
fastshap_inference_time = (np.load(f'inDelphi/figures/shap_results/time_fastshap_inference_{celltype}_{property}.npy') * len(df_heldout) + np.load(f'inDelphi/figures/shap_results/time_fastshap_inference_extra_{celltype}_{property}.npy') * len(df_extra) + np.load(f'inDelphi/figures/shap_results/time_fastshap_inference_extra_{celltype}_{property}_x2.npy') * len(df_extra_2)) / (len(df_heldout) + len(df_extra) + len(df_extra_2))
overall_fastshap_time = [fastshap_surrogate_time + fastshap_explainer_time + (fastshap_inference_time / len(df_heldout)) * i for i in num_samples]

df_S4b = pd.DataFrame({
    "Number of samples": num_samples,
    "FastSHAP total runtime (seconds)": overall_fastshap_time,
    "SHAP zero values total runtime (seconds)": overall_shapzero_time
})

with pd.ExcelWriter("SourceData.xlsx") as writer:
    df_2c.to_excel(writer, sheet_name="Fig. 2c", index=False)
    df_2d.to_excel(writer, sheet_name="Fig. 2d", index=False)
    df_2e.to_excel(writer, sheet_name="Fig. 2e", index=False)
    df_2f.to_excel(writer, sheet_name="Fig. 2f", index=False)
    df_3c.to_excel(writer, sheet_name="Fig. 3c", index=False)
    df_3d.to_excel(writer, sheet_name="Fig. 3d", index=False)
    df_3e.to_excel(writer, sheet_name="Fig. 3e", index=False)
    df_3f.to_excel(writer, sheet_name="Fig. 3f", index=False)
    df_S1a.to_excel(writer, sheet_name="Fig. S1a", index=False)
    df_S1b.to_excel(writer, sheet_name="Fig. S1b", index=False)
    df_S2a.to_excel(writer, sheet_name="Fig. S2a", index=False)
    df_S2b.to_excel(writer, sheet_name="Fig. S2b", index=False)
    df_S3a.to_excel(writer, sheet_name="Fig. S3a", index=False)
    df_S3b.to_excel(writer, sheet_name="Fig. S3b", index=False)
    df_S4a.to_excel(writer, sheet_name="Fig. S4a", index=False)
    df_S4b.to_excel(writer, sheet_name="Fig. S4b", index=False)



