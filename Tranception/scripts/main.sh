STEP_ZERO="../get_qary_indices.py"
STEP_ONE="../compute_samples.py"
STEP_TWO="../run_helper.py"

q=20
n=10
b=3
num_subsample=3
num_repeat=3
protein="GFP"
model="Tranception"
param="${protein}_${model}"

python "$STEP_ZERO" \
    --q ${q} \
    --n ${n} \
    --b ${b} \
    --num_subsample ${num_subsample} \
    --num_repeat ${num_repeat} \

python "$STEP_ONE" \
    --q ${q} \
    --n ${n} \
    --b ${b} \
    --num_subsample ${num_subsample} \
    --num_repeat ${num_repeat} \
    --protein ${protein} \
    --model ${model} \

python "$STEP_TWO" \
    --q ${q} \
    --n ${n} \
    --b ${b} \
    --noise_sd 0.725 \
    --num_subsample ${num_subsample} \
    --num_repeat ${num_repeat} \
    --hyperparam "True" \
    --hyperparam_range "[0, 0.1, 0.01]" \
    --param ${param}
