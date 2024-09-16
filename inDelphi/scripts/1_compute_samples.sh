HELPER_PATH="../compute_samples.py"

python "$HELPER_PATH" \
    --q 4 \
    --n 40 \
    --b 7 \
    --num_subsample 3 \
    --num_repeat 3 \
    --celltype "HEK293"
