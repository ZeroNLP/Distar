import os
import argparse

from sklearn.model_selection import train_test_split

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

from source.utils.logger import logger
from source.utils.utils import load_split_types, load_json_data, save_json_data, set_seed
from source.data_processing.augmenter import ArgumentAugmenter
from source.data_processing.conversion import (split_by_event_type,
                                               merge_unseen_event,
                                               convert_seen_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42,
                        help="Random number seed")
    parser.add_argument("--input_data", type=str,
                        default="data/ACE_converted/english.event.json",
                        help="Path to the input converted data")
    parser.add_argument("--output_dir", type=str,
                        default="data/ACE_distar/split_a",
                        help="Path to the output directory")
    parser.add_argument("--type_split_path", type=str,
                        default="data/ACE_distar/split_a/type_split.json",
                        help="Path to the type_split.json file")
    parser.add_argument("--is_char_offset", action='store_true',
                        help="Whether the offset of input data is character level")
    parser.add_argument("--dev_ratio", type=float, default=0.1,
                        help="Ratio of development set")

    parser.add_argument("--do_augment", action="store_true", default=False,
                        help="Whether generate augmented data")
    parser.add_argument("--aug_model_name", type=str, default="t5-base",
                        help="Model name or path of the model for data augmentation")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CPU or GPU device used in augmenting")
    parser.add_argument("--num_beam", type=int, default=200,
                        help="Number of beams for beam search")
    parser.add_argument("--max_argument_length", type=int, default=10,
                        help="The maximum length the generated tokens can have")
    parser.add_argument("--num_return", type=int, default=15,
                        help="The number of independently computed returned sequences")

    args = parser.parse_args()

    set_seed(args.seed)

    # load preprocessed ACE data
    data_converted = load_json_data(args.input_data)
    # load seen and unseen event types
    type_split = load_split_types(args.type_split_path)
    # split preprocessed ACE data by seen and unseen event types
    data_dict = split_by_event_type(data_converted, type_split, args.is_char_offset)

    # save seen and unseen data
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for split, data in data_dict.items():
        save_json_data(data, os.path.join(args.output_dir, f"{split}_data.json"))

    # split seen data into training set and development set
    train_seen_split, dev_seen_split = train_test_split(data_dict["seen"],
                                                        test_size=args.dev_ratio,
                                                        random_state=args.seed,
                                                        shuffle=True)
    # convert seen data to the data with triplet
    train_seen_data = convert_seen_data(train_seen_split)
    dev_seen_data = convert_seen_data(dev_seen_split)

    # test_unseen_data = convert_seen_data(data_dict["unseen"])
    # save_json_data(test_unseen_data, os.path.join(args.output_dir, "test_unseen.distar.json"))

    # merge unseen data with same sentence
    # train_seen_data_merge = merge_unseen_event(train_seen_split)
    # dev_seen_data_merge = merge_unseen_event(dev_seen_split)
    # save_json_data(train_seen_data_merge, os.path.join(args.output_dir, "train.oneie.json"))
    # save_json_data(dev_seen_data_merge, os.path.join(args.output_dir, "dev.oneie.json"))

    test_unseen_data = merge_unseen_event(data_dict["unseen"])

    # save data
    save_json_data(train_seen_data, os.path.join(args.output_dir, "train.distar.json"))
    save_json_data(dev_seen_data, os.path.join(args.output_dir, "dev.distar.json"))
    save_json_data(test_unseen_data, os.path.join(args.output_dir, "test.distar.json"))

    if args.do_augment:
        logger.info(f"load {args.aug_model_name} model...")

        tokenizer = T5Tokenizer.from_pretrained(args.aug_model_name)
        config = T5Config.from_pretrained(args.aug_model_name)
        model = T5ForConditionalGeneration.from_pretrained(args.aug_model_name, config=config)
        model = model.to(args.device)

        augmenter = ArgumentAugmenter(
            tokenizer=tokenizer,
            config=config,
            model=model,
            num_beam=args.num_beam,
            num_return=args.num_return,
            max_argument_length=args.max_argument_length
        )

        logger.info("Start Generating Augmented Data")
        aug_seen_data = augmenter.generate_augmented_data(train_seen_split)
        aug_seen_data = convert_seen_data(aug_seen_data)
        save_json_data(aug_seen_data, os.path.join(args.output_dir, "augment.distar.json"))
