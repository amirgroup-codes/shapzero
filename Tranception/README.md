
# Reproducing Tranception experiments

SHAP zero is divided into two steps:

1. Recover the Fourier coefficients from the model. 
2. Convert the Fourier coefficients to SHAP values and Shapley interactions.

The contents of this README and the steps are similar to the README in TIGER. Run run all Tranception-specific scripts, such as querying the model, we provide the 'tranception.yml' environment file. First, if you have not installed the shapzero_expts environment, run the following command to create the environment:
```console
conda env create --name shapzero_expts -f ../TIGER/shapzero_expts.yml
conda activate shapzero_expts
```
Then, to run all scripts Tranception-specific, create the Tranception environment (no need to load this yet):
```console
conda env create --name tranception -f tranception.yml
```

## Step 0: Download Tranception_Large

Please go to [https://github.com/OATML-Markslab/Tranception](https://github.com/OATML-Markslab/Tranception) and follow instructions on how to download Tranception. These tests were performed on the large model. Place the model at `model/Tranception_Large/`.

## Step 1: Recover the Fourier coefficients from the model

Frequent error with *q*-SFT: if you get the following message
> ImportError: cannot import name 'decode_jit' from 'galois._codes._reed_solomon' (/.conda/pkgs/tiger/lib/python3.10/site-packages/galois/_codes/_reed_solomon.py)

Be sure to install this specific version of galois:
```console
pip install galois==0.1.1
```

All scripts needed to recover the Fourier coefficients are in located in the the **scripts/** folder. Inside the folder are the following scripts:

1. **0_get_qary_indices.sh:** Lets *q*-SFT know which sequences and how many sequences to evaluate. The parameters chosen in the paper were `q=20` (for 20 amino acids), `n=10` (for a 10-length sequence), `b=3`, `num_subsample=3`, and `num_repeat=3`.\\. For our experiments, these parameters will tell our model to take 792,000 samples (you can see this number if you go into `/results/q20_n10_b3/helper_results_GFP_Tranception.txt`). 

> If not present, running the following command will create the folder `/results/q{q}_n{n}_b{b}/` and generate all the necessary files for *q*-SFT.
```console
sh 0_get_qary_indices.sh
```

> To check whether or not things are working properly, you should see the following files:
> - `/results/q{q}_n{n}_b{b}/train/M{0-2}_D{0-2}_qaryindices.pickle` + `/results/q{q}_n{n}_b{b}/train/M{0-2}_D{0-2}_queryindices.pickle`
> - `/results/q{q}_n{n}_b{b}/test/signal_t_query_qaryindices.pickle` + `/results/q{q}_n{n}_b{b}/test/signal_t_queryindices.pickle`

Before running the next script, switch the environment to inDelphi: 
```console
conda activate tranception
```
2. **1_compute_samples.sh:** Using the files generated in the previous step, this script will compute the samples needed for *q*-SFT. All parameters from the previous script are needed here, plus `param="GFP"` and `model="Tranception"`.
- Running this script from scratch will create `/results/q{q}_n{n}_b{b}/train/samples/GFP_Tranception`.

```console
sh 1_compute_samples.sh
```
> To check whether or not things are working properly, you need the following files to run the next script:
> - `/results/q{q}_n{n}_b{b}/train/samples/GFP_Tranception/M{0-2}_D{0-2}.pickle` + `/results/q{q}_n{n}_b{b}/train/samples/GFP_Tranception/train_mean.npy`
> - `/results/q{q}_n{n}_b{b}/test/GFP_Tranception/signal_t.pickle`
> - For more information about how samples are computed, see the `../compute_samples.py` file. This file is specific to every model, so if you want to run your own SHAP zero experiment, you'll need to modify this file. 
 
Before running the next script, switch the environment back to shapzero_expts: 
```console
conda activate shapzero_expts
```
3. **2_run_helper.sh:** This script will run *q*-SFT to compute the Fourier coefficients. All parameters from the previous script are needed here, plus three new parameters: `noise_sd`, `hyperparam`, and `hyperparam_range`. `noise_sd` controls the peeling threshold for Fourier coefficients, and heavily determines how well the Fourier coefficients fit to the model. `hyperparam` controls whether or not you want to perform an iterate search through the hyperparameter range set by `hyperparam_range`, which is a list of numbers in the format `[min, max, step]`. To reproduce the paper's results, set `noise_sd=0.03` and `hyperparam="False"`
> If not present, running the following command will create the files `/results/q{q}_n{n}_b{b}/helper_results_GFP_Tranception.txt`, `/results/q{q}_n{n}_b{b}/magnitude_of_interactions_GFP_Tranception.png`, `/results/q{q}_n{n}_b{b}/qsft_transform_GFP_Tranception.pickle`, and `/results/q{q}_n{n}_b{b}/test_samples_GFP_Tranception.png`.
```console
sh 2_run_helper.sh
```

While the details of *q*-SFT are not important, it's very important to check the quality of the Fourier coefficients. Going into `helper_results_GFP_Tranception.txt` will show you the NMSE and R^2 of the Fourier coefficients when predicting 10,000 random samples. In our experiments, we found that we could approximate the samples using an R^2 of 0.97, but your results may vary depending on how many samples you take and the complexity of your model. The worse the R^2 is, the less accurate the Shapley estimations are. You can see the quality of the fit on these samples in `test_samples_GFP_Tranception.png`, and you can see the hamming weight of the Fourier coefficients in `magnitude_of_interactions_GFP_Tranception.png` (higher = more higher order Shapley interactions). Lastly, `qsft_transform_GFP_Tranception.pickle` contains a dictionary of Fourier coefficients. To continue onto step 2, we need `qsft_transform_GFP_Tranception.pickle` specifically.


## Step 2: Convert the Fourier coefficients to SHAP values and Shapley interactions 

Now that we have `qsft_transform_GFP_Tranception.pickle`, we can use it to compute SHAP values and Shapley interactions. To do this, we will go into the `figures/` folder to reproduce the results from the paper. Below is the structure of the `figures/` folder:

- **data/:** folder containing all experimental data.
- **shap_results/:** folder containing all information about SHAP values and Shapley interactions (our results will be located here!).
- **samples_search/:** folder containing all results for the Shapley convergence plots.

As with Step 1, we will walk through all the Python scripts in numerical order.

1. **0_generate_data.py:** This script will generate the experimental data needed to reproduce the results from the paper. If run properly, this script will create npy files in the `data/` folder. 

Before running the next script, switch the environment to Tranception: 
```console
conda activate tranception
```

2. **1_time.py:** Used to time the speed of computing Tranception.

3. **2_explain_shap.py:** This script will run KernelSHAP and SHAP-IQ from the paper, as well as FastSHAP. SHAP-IQ is run using the `shapiq/` directory in TIGER. All results are deposited into the `shap_results/` folder.

Before running the next script, switch the environment back to shapzero_expts. We will not need the Tranception environment anymore: 
```console
conda activate shapzero_expts
```

4. **2_shapzero.py:** This script will run SHAP zero by calling the `shapzero.py` file in `gen`. All results are deposited into the `shap_results/` folder.
5. **3_maintext_figure.py:** This script will reproduce the main text figure from the paper by combining all plots/data into a PDF. We'll get both time complexity and SHAP values/interactions plots in this script. We'll also get our supplementary tables signifiying the top interactions and top SHAP values in the `shap_results/` folder. For SHAP zero, this will be under `shapzero_results/GFP_Tranception_shapzero_values.csv` for SHAP values and `shapzero_results/GFP_Tranception_shapzero_fsi.csv` for Shapley interactions.
6. **4_supp_figures.py:** This script will reproduce the supplementary figures from the paper, which includes FastSHAP.
7. **4_shap_convergence.py:** This script will reproduce the Shapley convergence experiments from the paper.

## Description of folders

**scripts/:** folder containing all scripts needed to run Fourier transform.

**figures/:** folder containing all figures and results.

**results/:** folder containing Fourier coefficients.

**tranception/:** codebase for Tranception.

**src/:** Tranception-specific functions.