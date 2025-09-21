import pandas as pd
import numpy as np
import re
import os
import sys
sys.path.append('../..')
import random
random.seed(0)

def extract_positions(mutant_string):
    """Extracts mutation positions from a mutant string."""
    matches = re.findall(r'(?<=\D)(\d+)(?=\D)', mutant_string)  
    return set(map(int, matches))  

def filter_by_positions(df, positions):
    """Filters df to only keep rows where at least one mutation position is in the given positions list."""
    positions_set = set(positions)  
    
    # Extract positions for each row and check for intersection
    filtered_rows = []
    for _, row in df.iterrows():
        mutant_positions = extract_positions(row["mutant"])
        if mutant_positions.issubset(positions_set):  
            filtered_rows.append(row)
        # break

    return pd.DataFrame(filtered_rows)  # Reconstruct DataFrame with filtered rows

def generate_full_sequence_array(initial, base_sequence, final_total_sequences=200, num_amino_acids=20, num_min_mutations=1, num_max_mutations=5):
    _, num_cols = initial.shape
    target_set = set(range(num_amino_acids))
    col_value_sets = [set(initial[:, i]) for i in range(num_cols)]
    arr_list = initial.tolist()

    # Extend the initial array to the desired length, being sure that every amino acid is represented at each position
    while len(arr_list) < final_total_sequences:
        new_row = base_sequence.copy()
        num_changes = np.random.randint(num_min_mutations, num_max_mutations + 1)  # Make mutations to the wildtype
        available_cols = list(range(num_cols))
        random.shuffle(available_cols)
        changes_made = 0

        for col in available_cols:
            if col_value_sets[col] != target_set:
                missing_vals = list(target_set - col_value_sets[col])
                if missing_vals:
                    new_val = random.choice(missing_vals)
                    new_row[col] = new_val
                    col_value_sets[col].add(new_val)
                    changes_made += 1

            if changes_made >= num_changes:
                break

        if all(col_value_sets[c] == target_set for c in range(num_cols)):
            new_row = base_sequence.copy()
            num_changes = np.random.randint(1, 6)
            change_cols = random.sample(range(num_cols), num_changes)
            for col in change_cols:
                new_row[col] = random.randint(0, num_amino_acids - 1)

        arr_list.append(new_row)

    result = np.array(arr_list)

    for i in range(num_cols):
        assert set(result[:, i]) == target_set, f"Column {i} missing values"

    return result

# Load data plus positions
q = 20
n = 10
b = 4
M = 3
D = 3
DMS = pd.read_csv('../data/GFP_AEQVI_Sarkisyan_2016.csv')
epistatic_positions = [83, 87, 82, 84, 43, 164, 47, 162, 140, 165, 221, 64, 66, 65, 95, 61] 
for i in range(len(epistatic_positions)):
    epistatic_positions[i] += 1 # 1-indexed
positions = epistatic_positions[0:n]

# Apply filtering
filtered_DMS = filter_by_positions(DMS, positions)
y = filtered_DMS['DMS_score'].values
np.save("data/GFP_Tranception_train_scores.npy", y)

# # For the training set, use the DMS data
# aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
# sequences = [''.join(row['mutated_sequence'][pos-1] for pos in positions) for _, row in filtered_DMS.iterrows()]
# sequences = np.array([[aa_dict[char] for char in seq] for seq in sequences])
# print(np.shape(sequences))
# np.save("data/GFP_Tranception_train.npy", sequences)

# # For the heldout set, randomly mutate the wildtype sequence up to 5 mutations away 
# DMS='GFP_AEQVI_Sarkisyan_2016' 
# DMS_list = pd.read_csv('../data/ProteinGym_reference_file_substitutions.csv')
# DMS_list = DMS_list[DMS_list['DMS_id'] == DMS]
# sequence = DMS_list["target_seq"].values[0].upper()
# base_sequence = np.array([aa_dict[c] for c in sequence])
# base_sequence = base_sequence[positions]
# sequences = generate_full_sequence_array(base_sequence.reshape(1, -1), base_sequence, final_total_sequences=200)
# np.save('data/GFP_Tranception_heldout.npy', sequences)