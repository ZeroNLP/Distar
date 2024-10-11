SPLIT_SETTING_LIST=(a b)
SPLIT_SETTING=${SPLIT_SETTING_LIST[0]}
MODEL_PATH=/data/transformers

python -u process_data.py \
    --seed 42 \
    --input_data data/ACE_converted/english.event.json \
    --output_dir data/ACE_distar/split_"${SPLIT_SETTING}" \
    --type_split_path data/ACE_distar/split_"${SPLIT_SETTING}"/type_split.json \
    --dev_ratio 0.1 \
    --do_augment \
    --aug_model_name ${MODEL_PATH}/t5-base \
    --device cuda:0 \
    --num_beam 200 \
    --max_argument_length 10 \
    --num_return 15
