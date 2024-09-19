HELPER_PATH="../run_helper.py"

python "$HELPER_PATH" \
    --q 4 \
    --n 26 \
    --b 7 \
    --noise_sd 0.725 \
    --num_subsample 3\
    --num_repeat 3 \
    --hyperparam "False" \
    --hyperparam_range "[0.7, 0.8, 0.025]" \
    --param "on_target"