DATA_DIR=data/ACE_distar
ROLE_DESC_DIR=source/data_processing/role_description
BERT_PATH=/data/transformers
OUTPUT_DIR=output

SPLIT_SETTING_LIST=(a b)
SPLIT_SETTING=${SPLIT_SETTING_LIST[0]}
KGE_SCORER_NAME=(TransE DistMult ComplEx RotatE)
TRIPLET_COMBINATION=(ar_t at_r tr_a)

python -u train.py \
    --split_setting "${SPLIT_SETTING}" \
    --type_split_path ${DATA_DIR}/split_"${SPLIT_SETTING}"/type_split.json \
    --train_data_path ${DATA_DIR}/split_"${SPLIT_SETTING}"/train.distar.json \
    --dev_data_path ${DATA_DIR}/split_"${SPLIT_SETTING}"/dev.distar.json \
    --test_data_path ${DATA_DIR}/split_"${SPLIT_SETTING}"/test.distar.json \
    --aug_data_path ${DATA_DIR}/split_"${SPLIT_SETTING}"/augment.distar.json \
    --num_aug_data 500 \
    --role_desc_path ${ROLE_DESC_DIR}/surface_name_type_constraint.txt \
    --trigger_left_token "[TRI]" \
    --trigger_right_token "[TRI]" \
    --bert_model_name ${BERT_PATH}/roberta-large \
    --num_labels 3 \
    --need_lstm \
    --lstm_dim 128 \
    --num_lstm_layer 1 \
    --num_role_encoder_layers 2 \
    --kge_scorer_name "${KGE_SCORER_NAME[0]}" \
    --triplet_comb "${TRIPLET_COMBINATION[0]}" \
    --seed 42 \
    --device cuda:0 \
    --num_neg_role 5 \
    --max_input_length 128 \
    --max_role_length 32 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --num_train_epochs 15 \
    --learning_rate 3e-5 \
    --lstm_learning_rate 2e-4 \
    --crf_learning_rate 2e-3 \
    --weight_decay 1e-4 \
    --adam_epsilon 1e-8 \
    --warmup_steps 100 \
    --grad_clip 10.0 \
    --eval_step 100 \
    --output_dir ${OUTPUT_DIR} \
    --do_train \
    --do_predict \
    --trigger_type gold
