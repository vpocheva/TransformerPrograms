#!/bin/bash

V=${V:-8}
N=${N:-8}
L=${L:-3}
H=${H:-4}
M=${M:-2}

VOCAB_SIZE=${V}
MAX_LENGTH=${N}
N_LAYERS=${L}
N_HEADS_CAT=${H}
N_HEADS_NUM=${H}
N_CAT_MLPS=${M}
N_NUM_MLPS=${M}
DATASET_SIZE=${DATASET_SIZE:-20000}
N_EPOCHS=${N_EPOCHS:-250}
SEED=0

DEVICE=${DEVICE:-"cuda"}

OUTPUT_DIR="output/rasp/${DATASET}/vocab${VOCAB_SIZE}maxlen${MAX_LENGTH}/transformer_program/headsc${N_HEADS_CAT}headsn${N_HEADS_NUM}nlayers${N_LAYERS}cmlps${N_MLPS}nmlps${N_NUM_CAT_MLPS}/s${SEED}/${DATASET_SIZE}";

# create dirs if they don't exist
mkdir -p ${OUTPUT_DIR};
touch ${OUTPUT_DIR}/params.json;
# Echo the json that reflect the parameters used
echo "{\"V\": ${V}, \"N\": ${N}, \"L\": ${L}, \"H\": ${H}, \"M\": ${M}, \"dataset_size\": ${DATASET_SIZE}, \"seed\": ${SEED}}" > ${OUTPUT_DIR}/params.json;

python src/run.py \
     --dataset "${DATASET}" \
     --vocab_size "${VOCAB_SIZE}" \
     --dataset_size ${DATASET_SIZE} \
     --min_length 1 \
     --max_length "${MAX_LENGTH}" \
     --n_epochs ${N_EPOCHS} \
     --batch_size 512 \
     --lr "5e-2" \
     --gumbel_samples 1 \
     --sample_fn "gumbel_soft" \
     --tau_init 3.0 \
     --tau_end 0.01 \
     --tau_schedule "geomspace" \
     --n_vars_cat 1 \
     --d_var "${MAX_LENGTH}" \
     --n_vars_num 1 \
     --n_layers "${N_LAYERS}" \
     --n_heads_cat "${N_HEADS_CAT}" \
     --n_heads_num "${N_HEADS_NUM}" \
     --n_cat_mlps "${N_CAT_MLPS}" \
     --n_num_mlps "${N_NUM_MLPS}" \
     --attention_type "cat" \
     --rel_pos_bias "fixed" \
     --one_hot_embed \
     --dropout 0.0 \
     --mlp_vars_in 2 \
     --d_mlp 64 \
     --count_only \
     --selector_width 0 \
     --seed "${SEED}" \
     --unique 1 \
     --device "${DEVICE}" \
     --save \
     --save_code \
     --output_dir ${OUTPUT_DIR};
