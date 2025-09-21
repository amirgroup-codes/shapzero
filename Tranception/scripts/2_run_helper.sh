HELPER_PATH="../run_helper.py"

python "$HELPER_PATH" \
    --q 20 \
    --n 10 \
    --b 3 \
    --noise_sd 0.03 \
    --num_subsample 3\
    --num_repeat 3 \
    --hyperparam "False" \
    --hyperparam_range "[0, 1, 0.01]" \
    --param "GFP_Tranception"