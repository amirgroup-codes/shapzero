HELPER_PATH="../run_helper.py"

python "$HELPER_PATH" \
    --q 4 \
    --n 40 \
    --b 7 \
    --noise_sd 15.0 \
    --num_subsample 3\
    --num_repeat 3 \
    --hyperparam "False" \
    --hyperparam_range "[14, 16, 0.5]" \
    --param "HEK293/frameshift"