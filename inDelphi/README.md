
# Reproducing inDelphi experiments

SHAP zero is divided into two steps:

1. Recover the Fourier coefficients from the model. 
2. Convert the Fourier coefficients to SHAP values and Shapley interactions.

The contents of this README and the steps are similar to the README in TIGER. As inDelphi runs on an older version of Python, we will be juggling between the inDelphi (for model-specific commands) and the shapzero_expt environment. First, if you have not installed the shapzero_expts environment, run the following command to create the environment:
```console
conda env create --name shapzero_expts -f ../TIGER/shapzero_expts.yml
conda activate shapzero_expts
```
Then, to run all scripts inDelphi-specific, create the inDelphi environment (no need to load this yet):
```console
conda env create --name inDelphi -f inDelphi.yml
```

## Step 1: Recover the Fourier coefficients from the model

Frequent error with *q*-SFT: if you get the following message
> ImportError: cannot import name 'decode_jit' from 'galois._codes._reed_solomon' (/usr/scratch/dtsui/.conda/pkgs/tiger/lib/python3.10/site-packages/galois/_codes/_reed_solomon.py)

Be sure to install this specific version of galois:
```console
pip install galois==0.1.1
```

All scripts needed to recover the Fourier coefficients are in located in the the **scripts/** folder. Inside the folder are the following scripts:

1. **0_get_qary_indices.sh:** Lets *q*-SFT know which sequences and how many sequences to evaluate. The parameters chosen in the paper were `q=4` (for 4 nucleotides), `n=40` (for a 40-length sequence), `b=7`, `num_subsample=3`, and `num_repeat=3`. `b`, `num_subsample`, and `num_repeat` are parameters that control how many samples to take. To run your own SHAP zero experiments, we recommend keeping `num_subsample` and `num_repeat` both fixed at 3, and changing `b`. For our experiments, these parameters will tell our model to take 6,045,696 samples (you can see this number if you go into `/results/q4_n40_b7/helper_results_HEK293_frameshift.txt`). 

> If not present, running the following command will create the folder `/results/q{q}_n{n}_b{b}/` and generate all the necessary files for *q*-SFT.
```console
sh 0_get_qary_indices.sh
```

> To check whether or not things are working properly, you should see the following files:
> - `/results/q{q}_n{n}_b{b}/train/M{0-2}_D{0-2}_qaryindices.pickle` + `/results/q{q}_n{n}_b{b}/train/M{0-2}_D{0-2}_queryindices.pickle`
> - `/results/q{q}_n{n}_b{b}/test/signal_t_query_qaryindices.pickle` + `/results/q{q}_n{n}_b{b}/test/signal_t_queryindices.pickle`

Before running the next script, switch the environment to inDelphi: 
```console
conda activate inDelphi
```
2. **1_compute_samples.sh:** Using the files generated in the previous step, this script will compute the samples needed for *q*-SFT. All parameters from the previous script are needed here, plus the parameter `celltype`, which loads the inDelphi model specifically designed for a given cell type.  To reproduce the paper's results, `param="HEK293"`.
- Running this script from scratch will create 6 folders in `/results/q{q}_n{n}_b{b}/train/samples/{celltype}`: `1_bp_ins`, `frameshift`, `indel_length`, `insertion`, `MH_del_freq`, and `precision`. All folders represent different summary statistics inDelphi produces. Since it doesn't cost anymore compute power to compute a single statistic versus all 6, we decided to compute them all. We will only need the `frameshift` parameter for our analysis, but feel free to run the script and extract the other 5 statistics for your own analysis.  

> If not present, running the following command will create the folders `/results/q{q}_n{n}_b{b}/train/samples/{param}`, `/results/q{q}_n{n}_b{b}/train/samples_mean/{param}`, `/results/q{q}_n{n}_b{b}/train/transforms/{param}`, and `/results/q{q}_n{n}_b{b}/test/{param}`. These folders are not particularly important as long as they're created.
```console
sh 1_compute_samples.sh
```
> To check whether or not things are working properly, you need the following files to run the next script:
> - `/results/q{q}_n{n}_b{b}/train/samples/{celltype}/frameshift/M{0-2}_D{0-2}.pickle` + `/results/q{q}_n{n}_b{b}/train/samples/{celltype}/frameshift/train_mean.npy`
> - `/results/q{q}_n{n}_b{b}/test/{celltype}/frameshift/signal_t.pickle`
> - For more information about how samples are computed, see the `../compute_samples.py` file. This file is specific to every model, so if you want to run your own SHAP zero experiment, you'll need to modify this file. 
 
Before running the next script, switch the environment back to shapzero_expts: 
```console
conda activate shapzero_expts
```
3. **2_run_helper.sh:** This script will run *q*-SFT to compute the Fourier coefficients. All parameters from the previous script are needed here, plus three new parameters: `noise_sd`, `hyperparam`, and `hyperparam_range`. `noise_sd` controls the peeling threshold for Fourier coefficients, and heavily determines how well the Fourier coefficients fit to the model. `hyperparam` controls whether or not you want to perform an iterate search through the hyperparameter range set by `hyperparam_range`, which is a list of numbers in the format `[min, max, step]`. To reproduce the paper's results, set `noise_sd=15` and `hyperparam="False"`
> If not present, running the following command will create the files `/results/q{q}_n{n}_b{b}/helper_results_{celltype}_frameshift.txt`, `/results/q{q}_n{n}_b{b}/magnitude_of_interactions_{celltype}_frameshift.png`, `/results/q{q}_n{n}_b{b}/qsft_transform_{celltype}_frameshift.pickle`, and `/results/q{q}_n{n}_b{b}/test_samples_{celltype}_frameshift.png`.
```console
sh 2_run_helper.sh
```

While the details of *q*-SFT are not important, it's very important to check the quality of the Fourier coefficients. Going into `helper_results_{celltype}_frameshift.txt` will show you the NMSE and R^2 of the Fourier coefficients when predicting 10,000 random samples. In our experiments, we found that we could approximate the samples using an R^2 of 0.82, but your results may vary depending on how many samples you take and the complexity of your model. The worse the R^2 is, the less accurate the Shapley estimations are. You can see the quality of the fit on these samples in `test_samples_{celltype}_frameshift.png`, and you can see the hamming weight of the Fourier coefficients in `magnitude_of_interactions_{celltype}_frameshift.png` (higher = more higher order Shapley interactions). Lastly, `qsft_transform_{celltype}_frameshift.pickle` contains a dictionary of Fourier coefficients. To continue onto step 2, we need `qsft_transform_{celltype}_frameshift.pickle` specifically.


## Step 2: Convert the Fourier coefficients to SHAP values and Shapley interactions 

Now that we have `qsft_transform_{celltype}_frameshift.pickle`, we can use it to compute SHAP values and Shapley interactions. To do this, we will go into the `figures/` folder to reproduce the results from the paper. Below is the structure of the `figures/` folder:

- **correlation_results/:** folder containing all information about experimentally correlating with real-life sequences.
- **data/:** folder containing all experimental data.
- **shap_results/:** folder containing all information about SHAP values and Shapley interactions (our results will be located here!).

As with Step 1, we will walk through all the Python scripts in numerical order.

1. **0_generate_data.py:** This script will generate the experimental data needed to reproduce the results from the paper. If run properly, this script will create CSVs in the `data/` folder. 

Before running the next script, switch the environment to inDelphi: 
```console
conda activate inDelphi
```

2. **1_scores.py:** Computes the various metrics to assess how well the Fourier coefficients approximate real-life data. Using the data from the `data/` folder, this script will predict the frameshift frequency (`Frameshift frequency`) using the Fourier coefficients, inDelphi, a linear model, and a pairwise model. All results will be saved to the `correlation_results/` folder.
3. **2_explain_shap.py:** This script will run KernelSHAP and SHAP-IQ from the paper. SHAP-IQ is run using the `shapiq/` directory in this folder, which is compatible with the inDelphi environment. All results are deposited into the `shap_results/` folder.

Before running the next script, switch the environment back to shapzero_expts. We will not need the inDelphi environment anymore: 
```console
conda activate shapzero_expts
```

4. **2_fastshap.py:** This script will run FastSHAP using the FastSHAP directory in TIGER. Since we directly use the training data made from `0_generate_data.py`, we will not need to use the inDelphi environment.
5. **2_shapzero.py:** This script will run SHAP zero by calling the `shapzero.py` file in `gen`. All results are deposited into the `shap_results/` folder.
6. **3_maintext_figure.py:** This script will reproduce the main text figure from the paper by combining all plots/data into a PDF. We'll get both time complexity and SHAP values/interactions plots in this script. We'll also get our supplementary tables signifiying the top interactions and top SHAP values in the `shap_results/` folder. For SHAP zero, this will be under `shapzero_results/{celltype}_frameshift_shapzero_values.csv` for SHAP values and `shapzero_results/{celltype}_frameshift_shapzero_fsi.csv` for Shapley interactions.
6. **4_supp_figures.py:** This script will reproduce the supplementary figures from the paper, which includes DeepSHAP and FastSHAP.

## Description of folders

**scripts/:** folder containing all scripts needed to run Fourier transform.

**figures/:** folder containing all figures and results.

**results/:** folder containing Fourier coefficients.

**shapiq/:** codebase for SHAP-IQ which is compatible with the inDelphi environment.

**inDelphi/:** codebase for inDelphi.

**src/:** inDelphi-specific functions.