HELPER_PATH="../compute_samples.py"

python "$HELPER_PATH" \
    --q 2 \
    --n 26 \
    --b 2 \
    --num_subsample 2 \
    --num_repeat 2 \
    --param "all" \
    --gpu "False" 
