# Distar: Zero-shot Event Argument Extraction by Disentangling Trigger from Argument and Role

Source code for paper __Zero-shot Event Argument Extraction by Disentangling Trigger from Argument and Role__

## Installation

```shell
cd Distar
conda env create -f environment.yml
conda activate distar_env
```

## Data

- ACE-2005-E+: Access from LDC [ACE 2005 Multilingual Training Corpus](https://catalog.ldc.upenn.edu/LDC2006T06)

After downloading the corpus, please put the data in the directory `data/ACE_raw`. For example, put English corpus of
ACE-2005-E+ in `data/ACE_raw/English`


## Data Preprocessing

Our preprocessing scripts `source/data_processing/ace_preprocess.py` and `process_raw_ace.py` are adapted from
the [Zero-shot Event Extraction](https://github.com/veronica320/Zeroshot-Event-Extraction)

To preprocess ACE-2005 into the format used for event extraction, run

```shell
bash scripts/run_preprocess.sh
```

or

```shell
python process_raw_ace.py \
    --input data/ACE_raw \
    --output data/ACE_converted \
    --split data/ACE_converted/doc_split \
    --bert roberta-large \
    --lang english \
    --time_and_val
```

Arguments for `process_raw_ace.py`

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `i`, `input` | str | `data/ACE_raw` | Path to the input folder |
| `o`, `output` | str | `data/ACE_converted` | Path to the output folder |
| `s`, `split` | str | `data/ACE_converted/doc_split` | Path to the split folder |
| `b`, `bert` | str | `roberta-large` | BERT model name |
| `l`, `lang` | str | `english` | Document language |
| `time_and_val` | bool | False | Extracts times and values |

## Data Conversion And Augmentation

To convert the preprocessed data to the format used for our method and augment the converted data, run

```shell
bash scripts/run_process_data.sh
```

or

```shell
SPLIT_SETTING_LIST=(a b)
SPLIT_SETTING=${SPLIT_SETTING_LIST[1]}
MODEL_PATH=/data/transformers

python process_data.py \
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

```

Arguments for `process_data.py`

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `seed` | int | `42` | Random number seed |
| `input_data` | str | `english.event.json` | Path to the input converted data |
| `output_dir` | str | `split_a` | Path to the output directory |
| `type_split_path` | str | `type_split.json` | Path to the type_split.json file |
| `is_char_offset` | bool | `False` | Whether the offset of input data is character level |
| `dev_ratio` | float | `0.1` | Ratio of development set |
| `do_augment` | bool | `False` | Whether generate augmented data |
| `aug_model_name` | str | `t5-base` | Model name or path of the model for data augmentation |
| `device` | str | `cuda:0` | CPU or GPU device used in augmenting |
| `num_beam` | int | `200` | Number of beams for beam search |
| `max_argument_length` | int | `10` | The maximum length the generated tokens can have |
| `num_return` | int | `15` | The number of independently computed returned sequences |

## Start Training, Prediction And Evaluation

To Train the Distar model, use the best checkpoint to predict on data that only contains unseen event types, and
evaluate the prediction results, run

```shell
bash scripts/run_train.sh
```

or

```shell
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
    --num_aug_data 1000 \
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
    --num_neg_role 3 \
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
```

Changing the arguments of `split_setting`, `bert_model_name`, `num_role_encoder_layers`, `kge_scorer_name`, `triplet_comb`, `num_neg_role` and `trigger_type` can reproduce other experimental results in the paper.

Arguments for `train.py`

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `split_setting` | str | `a` | Split setting. Choose from `["a", "b"]` |
| `type_split_path` | str | `type_split.json` | Path to the type_split.json file |
| `train_data_path` | str | `train.distar.json` | Path to the training data |
| `dev_data_path` | str | `dev.distar.json` | Path to the development data |
| `test_data_path` | str | `test.distar.json` | Path to test data |
| `aug_data_path` | str | `augment.distar.json` | Path to augmented data |
| `num_aug_data` | int | `1000` | Number of augmented data |
| `role_desc_path` | str | `surface_name_type_constraint.txt` | Path to role description file |
| `trigger_left_token` | str | `[TRI]` | Special token indicating the trigger left position |
| `trigger_right_token` | str | `[TRI]` | Special token indicating the trigger right position |
| `bert_model_name` | str | `roberta-large` | Bert or Distar model name or path |
| `num_labels` | int | `3` | Number of BIO labels |
| `need_lstm` | bool | `False` | Whether need BiLSTM layer |
| `lstm_dim` | int | `128` | Hidden size of BiLSTM layer |
| `num_lstm_layer` | int | `1` | Number of BiLSTM layers |
| `num_role_encoder_layers` | int | `2` | Number of the transformer layers in role encoder |
| `kge_scorer_name` | str | `TransE` | Name of KGE score method. Choose from `["TransE", "DistMult", "ComplEx", "RotatE"]` |
| `triplet_comb` | str | `ar_t` | Triplet combination in KGE. Choose from `["ar_t", "at_r", "tr_a"]` |
| `seed` | int | `42` | Random number seed |
| `device` | str | `cuda:0` | CPU or GPU device used in training |
| `num_neg_role` | int | `1` | Number of negative role samples |
| `max_input_length` | int | `128` | Max input sequence length |
| `max_role_length` | int | `32` | Max role description sequence length |
| `train_batch_size` | int | `32` | Batch size for training |
| `eval_batch_size` | int | `32` | Batch size for evaluation |
| `num_train_epochs` | int | `15` | Number of epochs for training |
| `learning_rate` | float | `3e-5` | Learning rate for BERT layer |
| `lstm_learning_rate` | float | `2e-4` | Learning rate for BiLSTM layer |
| `crf_learning_rate` | float | `2e-3` | Learning rate for CRF and linear layer |
| `weight_decay` | float | `1e-4` | Weight decay for optimizer |
| `adam_epsilon` | float | `1e-8` | Epsilon for Adam/AdamW optimizer | 
| `warmup_steps` | int | `100` | Number of warm-up steps for scheduler |
| `grad_clip` | float | `10.0` | Max norm of the gradients |
| `eval_step` | int | `100` | Number of evaluation steps |
| `output_dir` | str | `output` | Path to output folder |
| `do_train` | bool | `False` | Whether to train |
| `do_predict` | bool | `False` | Whether to predict |
| `trigger_type` | str | `gold` | Input gold trigger or predicted trigger in prediction. Choose from `["gold", "pred]` |










