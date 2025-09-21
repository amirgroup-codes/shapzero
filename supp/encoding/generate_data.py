# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Special imports
# import mavenn
import os
import urllib.request

# Specify dataset
dataset = 'mpsa' # use to make 'mpsa' dataset
# dataset = 'mpsa_replicate' # uncomment to make 'mpsa_replicate' dataset instead
print(f'Creating dataset "{dataset}"...')

# Specify online directory
url_dir = 'https://github.com/jbkinney/15_splicing/raw/master/for_mavenn/'

# Specify raw data file and output file names
if dataset=='mpsa':
    raw_data_file = 'results.brca2_9nt_lib1_rep1.txt.gz'
elif dataset=='mpsa_replicate':
    raw_data_file = 'results.brca2_9nt_lib2_rep1.txt.gz'
else:
    assert False

# Download raw dataset
print(f'Retrieving data from {url_dir+raw_data_file}...')
urllib.request.urlretrieve(url_dir+raw_data_file, raw_data_file)

# Load raw datset
raw_df = pd.read_csv(raw_data_file, sep='\t', index_col=0, compression='gzip')

# Delete raw dataset
os.remove(raw_data_file)

# Preview raw dataset
print('Done!')
raw_df

# Remove unwanted columns
data_df = raw_df[['tot_ct','ex_ct','ss','bc']].copy()

# Marginalize by splice site
data_df = data_df.groupby('ss').sum().reset_index()

# Rename columns
data_df.rename(columns={'ss':'x'}, inplace=True)

# Make sequences RNA
data_df['x'] = [ss.replace('T','U') for ss in data_df['x']]

# Remove ss with minimum tot_ct
min_ct = 10
ix = data_df['tot_ct'] >= min_ct
data_df = data_df[ix].copy()
print(f'{(~ix).sum()} ss removed for having tot_ct < {min_ct}')

# Remove ss with invalid sequences
ix = np.array([((x[3]=='G') and (x[4] in {'C','U'})) for x in data_df['x']])
data_df = data_df[ix]
print(f'{(~ix).sum()} ss with invalid sequences removed')

# Get consensus i_cons and o_cons
pseudoct = 1.0
cons_seq = 'CAGGUAAGU'
tmp_df = data_df.set_index('x')
i_cons = tmp_df.loc[cons_seq,'tot_ct']
o_cons = tmp_df.loc[cons_seq,'ex_ct']
relative_psi_cons = (o_cons+pseudoct)/(i_cons+pseudoct)

# Compute y
i_n = data_df['tot_ct']
o_n = data_df['ex_ct']
relative_psi_n = (o_n+pseudoct)/(i_n+pseudoct)
psi_n = 100*relative_psi_n/relative_psi_cons
y_n = np.log10(psi_n)
data_df['y'] = y_n

# Assign data to training, validation, or test sets
np.random.seed(0)
data_df['set'] = np.random.choice(a=['training', 'validation', 'test'],
                                   p=[.6,.2,.2],
                                   size=len(data_df))

# Shuffle data for extra safety
data_df = data_df.sample(frac=1).reset_index(drop=True)

# Order columns
final_df = data_df[['set', 'tot_ct', 'ex_ct', 'y', 'x']].copy()

# Save to file (uncomment to execute)
out_file = f'{dataset}_data.csv.gz'
final_df.to_csv(out_file, index=False, compression='gzip')
print(f'Dataset "{dataset}" saved to {out_file}')

# Preview final dataframe
print('Done!')
final_df

final_df.to_csv('mpsa.csv')