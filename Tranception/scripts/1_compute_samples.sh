HELPER_PATH="../compute_samples.py"

python "$HELPER_PATH" \
    --q 20 \
    --n 10 \
    --b 3 \
    --num_subsample 3 \
    --num_repeat 3 \
    --param "GFP" \
    --model "Tranception"
