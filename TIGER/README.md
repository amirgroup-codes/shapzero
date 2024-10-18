
# Reproducing TIGER experiments

SHAP zero is divided into two steps:

1. Recover the Fourier coefficients from the model. 
2. Convert the Fourier coefficients to SHAP values and Shapley interactions.

For all scripts in this directory, we provide the `shapzero_expts.yml` environment file. Run the following command to create the environment:
```console
conda env create --name shapzero_expts -f shapzero_expts.yml
conda activate shapzero_expts
```

## Step 1: Recover the Fourier coefficients from the model

Frequent error with *q*-SFT: if you get the following message
> ImportError: cannot import name 'decode_jit' from 'galois._codes._reed_solomon' (/usr/scratch/dtsui/.conda/pkgs/tiger/lib/python3.10/site-packages/galois/_codes/_reed_solomon.py)

Be sure to install this specific version of galois:
```console
pip install galois==0.1.1
```

All scripts needed to recover the Fourier coefficients are in located in the the **scripts/** folder. Inside the folder are the following scripts:

1. **0_get_qary_indices.sh:** Lets *q*-SFT know which sequences and how many sequences to evaluate. The parameters chosen in the paper were `q=4` (for 4 nucleotides), `n=26` (for a 26-length sequence), `b=7`, `num_subsample=3`, and `num_repeat=3`. `b`, `num_subsample`, and `num_repeat` are parameters that control how many samples to take. To run your own SHAP zero experiments, we recommend keeping `num_subsample` and `num_repeat` both fixed at 3, and changing `b`. For our experiments, these parameters will tell our model to take 3,981,312 samples (you can see this number if you go into `/results/q4_n26_b7/helper_results_on_target.txt`). 

> If not present, running the following command will create the folder `/results/q{q}_n{n}_b{b}/` and generate all the necessary files for *q*-SFT.
```console
sh 0_get_qary_indices.sh
```

> To check whether or not things are working properly, you should see the following files:
> - `/results/q{q}_n{n}_b{b}/train/M{0-2}_D{0-2}_qaryindices.pickle` + `/results/q{q}_n{n}_b{b}/train/M{0-2}_D{0-2}_queryindices.pickle`
> - `/results/q{q}_n{n}_b{b}/test/signal_t_query_qaryindices.pickle` + `/results/q{q}_n{n}_b{b}/test/signal_t_queryindices.pickle`

2. **1_compute_samples.sh:** Using the files generated in the previous step, this script will compute the samples needed for *q*-SFT. All parameters from the previous script are needed here, plus two new parameters: `param` and `gpu`. `param` controls a parameter in `../compute_samples.py` that determines whether TIGER should compute samples for on-target (**perfect match**) or the average titration score given a target sequence and all the possible 1-bp guide sequence mismatches. `gpu` controls whether or not a GPU should be used to compute the samples. To reproduce the paper's results, `param="on_target"` and `gpu="True"`.

> If not present, running the following command will create the folders `/results/q{q}_n{n}_b{b}/train/samples/{param}`, `/results/q{q}_n{n}_b{b}/train/samples_mean/{param}`, `/results/q{q}_n{n}_b{b}/train/transforms/{param}`, and `/results/q{q}_n{n}_b{b}/test/{param}`. These folders are not particularly important as long as they're created.
```console
sh 1_compute_samples.sh
```
> To check whether or not things are working properly, you need the following files to run the next script:
> - `/results/q{q}_n{n}_b{b}/train/samples/{param}/M{0-2}_D{0-2}.pickle` + `/results/q{q}_n{n}_b{b}/train/samples/{param}/train_mean.npy`
> - `/results/q{q}_n{n}_b{b}/test/{param}/signal_t.pickle`
> - For more information about how samples are computed, see the `../compute_samples.py` file. This file is specific to every model, so if you want to run your own SHAP zero experiment, you'll need to modify this file. 
 
3. **2_run_helper.sh:** This script will run *q*-SFT to compute the Fourier coefficients. All parameters from the previous script are needed here, plus three new parameters: `noise_sd`, `hyperparam`, and `hyperparam_range`. `noise_sd` controls the peeling threshold for Fourier coefficients, and heavily determines how well the Fourier coefficients fit to the model. `hyperparam` controls whether or not you want to perform an iterate search through the hyperparameter range set by `hyperparam_range`, which is a list of numbers in the format `[min, max, step]`. To reproduce the paper's results, set `noise_sd=0.725` and `hyperparam="False"`
> If not present, running the following command will create the files `/results/q{q}_n{n}_b{b}/helper_results_{param}.txt`, `/results/q{q}_n{n}_b{b}/magnitude_of_interactions_{param}.png`, `/results/q{q}_n{n}_b{b}/qsft_transform_{param}.pickle`, and `/results/q{q}_n{n}_b{b}/test_samples_{param}.png`.
```console
sh 2_run_helper.sh
```

While the details of *q*-SFT are not important, it's very important to check the quality of the Fourier coefficients. Going into `helper_results_{param}.txt` will show you the NMSE and R^2 of the Fourier coefficients when predicting 10,000 random samples. In our experiments, we found that we could approximate the samples using an R^2 of 0.55, but your results may vary depending on how many samples you take and the complexity of your model. The worse the R^2 is, the less accurate the Shapley estimations are. You can see the quality of the fit on these samples in `test_samples_{param}.png`, and you can see the hamming weight of the Fourier coefficients in `magnitude_of_interactions_{param}.png` (higher = more higher order Shapley interactions). Lastly, `qsft_transform_{param}.pickle` contains a dictionary of Fourier coefficients. To continue onto step 2, we need `qsft_transform_{param}.pickle` specifically.


## Step 2: Convert the Fourier coefficients to SHAP values and Shapley interactions 

Now that we have `qsft_transform_{param}.pickle`, we can use it to compute SHAP values and Shapley interactions. To do this, we will go into the `figures/` folder to reproduce the results from the paper. Below is the structure of the `figures/` folder:

- **correlation_results/:** folder containing all information about experimentally correlating with real-life sequences.
- **data/:** folder containing all experimental data.
- **shap_results/:** folder containing all information about SHAP values and Shapley interactions (our results will be located here!).

As with Step 1, we will walk through all the Python scripts in numerical order.

1. **0_generate_data.py:** This script will generate the experimental data needed to reproduce the results from the paper. If run properly, this script will create CSVs in the `data/` folder. 
2. **1_scores.py:** Computes the various metrics to assess how well the Fourier coefficients approximate real-life data. Using the data from the `data/` folder, this script will predict the guide score (`observed_lfc`) using the Fourier coefficients, TIGER, a linear model, and a pairwise model. All results will be saved to the `correlation_results/` folder.
3. **2_explain_shap.py:** This script will run all the baselines (KernelSHAP, SHAP-IQ, FastSHAP, and DeepSHAP) from the paper. All results are deposited into the `shap_results/` folder.
4. **2_shapzero.py:** This script will run SHAP zero by calling the `shapzero.py` file in `gen`. All results are deposited into the `shap_results/` folder.
5. **3_maintext_figure.py:** This script will reproduce the main text figure from the paper by combining all plots/data into a PDF. We'll get both time complexity and SHAP values/interactions plots in this script. We'll also get our supplementary tables signifiying the top interactions and top SHAP values in the `shap_results/` folder. For SHAP zero, this will be under `shapzero_results/{param}_shapzero_values.csv` for SHAP values and `shapzero_results/{param}_shapzero_fsi.csv` for Shapley interactions.
6. **4_supp_figures.py:** This script will reproduce the supplementary figures from the paper, which includes DeepSHAP and FastSHAP.


## Description of folders

**scripts/:** folder containing all scripts needed to run Fourier transform.

**figures/:** folder containing all figures and results.

**results/:** folder containing Fourier coefficients.

**fastshap-main/:** codebase for FastSHAP.

**tiger/:** codebase for TIGER.

**src/:** TIGER-specific functions.