HELPER_PATH="../compute_samples.py"

python "$HELPER_PATH" \
    --q 2 \
    --n 5 \
    --b 2 \
    --num_subsample 3 \
    --num_repeat 3 \
    --celltype "HEK293"
