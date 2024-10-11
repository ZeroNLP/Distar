python -u process_raw_ace.py \
    -i data/ACE_raw \
    -o data/ACE_converted \
    -s data/ACE_converted/doc_split \
    -b /data/transformers/roberta-large \
    -l english \
    --time_and_val