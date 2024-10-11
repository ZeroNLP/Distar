"""
Adapted from https://github.com/veronica320/Zeroshot-Event-Extraction
"""

import os
import argparse

from transformers import (BertTokenizerFast,
                          RobertaTokenizerFast,
                          XLMRobertaTokenizer)

from source.data_processing.ace_preprocess import (convert_batch,
                                                   convert_to_event_only,
                                                   split_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', default="data/ACE_raw", help='Path to the input folder')
    parser.add_argument('-o', '--output', default="data/ACE_converted", help='Path to the output folder')
    parser.add_argument('-s', '--split', default="data/ACE_converted/doc_split",
                        help='Path to the split folder')
    parser.add_argument('-b',
                        '--bert',
                        help='BERT model name',
                        default='roberta-large')
    parser.add_argument('-c',
                        '--bert_cache_dir', default=None,
                        help='Path to the BERT cache directory')
    parser.add_argument('-l', '--lang', default='english',
                        help='Document language')
    parser.add_argument('--time_and_val', action='store_true',
                        help='Extracts times and values')

    args = parser.parse_args()
    if args.lang not in ['chinese', 'english']:
        raise ValueError('Unsupported language: {}'.format(args.lang))
    input_dir = os.path.join(args.input, args.lang.title())

    # Create a tokenizer based on the model name
    model_name = args.bert
    cache_dir = args.bert_cache_dir
    if model_name.startswith('bert-'):
        tokenizer = BertTokenizerFast.from_pretrained(model_name,
                                                      cache_dir=cache_dir)
    elif model_name.startswith('roberta-') or 'roberta' in model_name:
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)  # ,
        # cache_dir=cache_dir)
    elif model_name.startswith('xlm-roberta-'):
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name,
                                                        cache_dir=cache_dir)
    else:
        raise ValueError('Unknown model name: {}'.format(model_name))

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # Convert to doc-level JSON format
    json_path = os.path.join(args.output, '{}.json'.format(args.lang))
    convert_batch(input_dir, json_path, time_and_val=args.time_and_val,
                  language=args.lang)

    # Convert to event-only format
    event_path = os.path.join(args.output, '{}.event.json'.format(args.lang))
    convert_to_event_only(json_path, event_path)

    # Split the data
    if args.split:
        split_data(event_path, args.output, args.split)
