import json
import os
import gc
import math
from argparse import ArgumentParser

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import evaluate

from source.utils.utils import load_json_data, load_role_description, set_seed, load_split_types
from source.utils.logger import logger
from source.model.model import DistarModel
from source.trainer import DistarTrainer
from source.predictor import DistarPredictor
from source.metric.metric import F1MetricForEAE


def get_argparse() -> ArgumentParser:
    parser = ArgumentParser()

    # data arguments
    parser.add_argument("--split_setting", type=str,
                        default="a", choices=["a", "b"],
                        help="Split setting")
    parser.add_argument("--type_split_path", type=str,
                        default="data/ACE_distar/split_a/type_split.json",
                        help="Path to the type_split.json file")
    parser.add_argument("--train_data_path", type=str,
                        default="data/ACE_distar/split_a/train.distar.json",
                        help="Path to the training data")
    parser.add_argument("--dev_data_path", type=str,
                        default="data/ACE_distar/split_a/dev.distar.json",
                        help="Path to the development data")
    parser.add_argument("--test_data_path", type=str,
                        default="data/ACE_distar/split_a/test.distar.json",
                        help="Path to test data")
    parser.add_argument("--aug_data_path", type=str,
                        default="data/ACE_distar/split_a/augment.distar.json",
                        help="Path to augmented data")
    parser.add_argument("--num_aug_data", type=int, default=1000,
                        help="Number of augmented data")

    parser.add_argument("--role_desc_path", type=str,
                        default="source/data_processing/role_description/surface_name_type_constraint.txt",
                        help="Path to role description file")

    parser.add_argument("--trigger_left_token", type=str, default="[TRI]",
                        help="Special token indicating the trigger left position")
    parser.add_argument("--trigger_right_token", type=str, default="[TRI]",
                        help="Special token indicating the trigger right position")

    # model arguments
    parser.add_argument("--bert_model_name", type=str, default="roberta-large",
                        help="Bert or Distar model name or path")
    parser.add_argument("--num_labels", type=int, default=3,
                        help="Number of BIO labels")
    parser.add_argument("--need_lstm", action="store_true",
                        help="Whether need BiLSTM layer")
    parser.add_argument("--lstm_dim", type=int, default=128,
                        help="Hidden size of BiLSTM layer")
    parser.add_argument("--num_lstm_layer", type=int, default=1,
                        help="Number of BiLSTM layers")

    parser.add_argument("--num_role_encoder_layers", type=int, default=2,
                        help="Number of the transformer layers in role encoder")
    parser.add_argument("--kge_scorer_name", type=str,
                        default="TransE", choices=["TransE", "DistMult", "ComplEx", "RotatE"],
                        help="Name of KGE score method")
    parser.add_argument("--triplet_comb", type=str,
                        default="ar_t", choices=["ar_t", "at_r", "tr_a"],
                        help="Triplet combination in KGE")

    # training argument
    parser.add_argument("--seed", type=int, default=42,
                        help="Random number seed")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CPU or GPU device used in training")

    parser.add_argument("--num_neg_role", type=int, default=1,
                        help="Number of negative role samples")

    parser.add_argument("--max_input_length", type=int, default=128,
                        help="Max input sequence length")
    parser.add_argument("--max_role_length", type=int, default=32,
                        help="Max role description sequence length")

    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Batch size for evaluation")

    parser.add_argument("--num_train_epochs", type=int, default=15,
                        help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Learning rate for BERT layer")
    parser.add_argument("--lstm_learning_rate", type=float, default=2e-4,
                        help="Learning rate for BiLSTM layer")
    parser.add_argument("--crf_learning_rate", type=float, default=2e-3,
                        help="Learning rate for CRF and linear layer")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for optimizer")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Epsilon for Adam/AdamW optimizer")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warm-up steps for scheduler")
    parser.add_argument("--grad_clip", type=float, default=10.0,
                        help="Max norm of the gradients")

    parser.add_argument("--eval_step", type=int, default=100,
                        help="Number of evaluation steps")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Path to output folder")

    parser.add_argument("--do_train", action="store_true", default=False,
                        help="Whether to train")
    parser.add_argument("--do_predict", action="store_true", default=False,
                        help="Whether to predict")
    parser.add_argument("--trigger_type", type=str, default="gold", choices=["gold", "pred"],
                        help="Input gold trigger or predicted trigger in prediction")

    return parser


def main():
    parser = get_argparse()

    logger.info("parsing args...")
    args = parser.parse_args()
    args.index2tag = {0: "O", 1: "B", 2: "I"}
    args.tag2index = {"O": 0, "B": 1, "I": 2}

    # load data
    set_seed(args.seed)

    logger.info("loading data...")
    train_data = load_json_data(args.train_data_path)
    dev_data = load_json_data(args.dev_data_path)

    args.add_aug_data = False
    if (args.aug_data_path is not None) and os.path.exists(args.aug_data_path):
        aug_data = load_json_data(args.aug_data_path)

        # arg_aug_data = load_json_data("data/ACE_distar/split_a/augment.distar.json")
        # train_data = train_data + arg_aug_data[:500]
        train_data = train_data + aug_data[:min(len(aug_data), args.num_aug_data)]

        # aug_data_sample = np.random.choice(aug_data, min(len(aug_data), args.num_aug_data), replace=False)
        # train_data = train_data + aug_data_sample.tolist()
        args.add_aug_data = True

    role_description = load_role_description(args.role_desc_path)
    split_types = load_split_types(args.type_split_path)

    logger.info(f"seen_types   = {split_types['seen_types']}")
    logger.info(f"unseen_types = {split_types['unseen_types']}\n")

    # setup for training
    logger.info("loading distar model...")
    tokenizer, config, model = DistarModel.load_model(
        model_name_or_path=args.bert_model_name,
        num_labels=len(args.tag2index),
        need_lstm=args.need_lstm,
        lstm_dim=args.lstm_dim,
        num_lstm_layer=args.num_lstm_layer,
        num_role_encoder_layers=args.num_role_encoder_layers,
        kge_scorer_name=args.kge_scorer_name,
        triplet_comb=args.triplet_comb,
        trigger_left_token=args.trigger_left_token,
        trigger_right_token=args.trigger_right_token
    )
    model = model.to(args.device)

    if args.do_train:
        logger.info("initialize KGE encoder's parameters from BERT...")
        model.init_kge_encoder_from_bert()

        logger.info("preparing optimizer and scheduler...")
        if "roberta-" in args.bert_model_name:
            bert_param_optimizer = list(model.roberta.named_parameters())
        else:
            bert_param_optimizer = list(model.bert.named_parameters())

        if args.need_lstm:
            lstm_param_optimizer = list(model.lstm.named_parameters())
        crf_param_optimizer = list(model.crf.named_parameters())
        linear_param_optimizer = list(model.classifier.named_parameters()) \
                                 + list(model.kge_scorer.named_parameters())
        kge_encoder_param_optimizer = list(model.kge_model.encoder.named_parameters()) \
                                      + list(model.kge_model.pooler.named_parameters())

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.learning_rate},
            {'params': [p for n, p in bert_param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.learning_rate},

            {'params': [p for n, p in kge_encoder_param_optimizer
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.learning_rate},
            {'params': [p for n, p in kge_encoder_param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.learning_rate},
            

            {'params': [p for n, p in crf_param_optimizer
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
            {'params': [p for n, p in crf_param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.crf_learning_rate},

            {'params': [p for n, p in linear_param_optimizer
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
            {'params': [p for n, p in linear_param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.crf_learning_rate},
        ]
        
        if args.need_lstm:
            optimizer_grouped_parameters.extend(
                [
                    {'params': [p for n, p in lstm_param_optimizer
                                if not any(nd in n for nd in no_decay)],
                    'weight_decay': args.weight_decay, 'lr': args.lstm_learning_rate},
                    {'params': [p for n, p in lstm_param_optimizer
                                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                    'lr': args.lstm_learning_rate},
                ]
            )

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        num_training_steps = args.num_train_epochs * math.ceil(len(train_data) / args.train_batch_size)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps
        )

        logger.info("loading sequence labeling evaluation metric...")
        metric = evaluate.load("source/metric/seqeval.py", zero_division=0)

        torch.cuda.empty_cache()
        gc.collect()

        # preparing trainer
        logger.info("preparing trainer...")
        trainer = DistarTrainer(
            args=args,
            role_desc=role_description,
            tokenizer=tokenizer,
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            train_data=train_data,
            eval_data=dev_data,
            split_types=split_types
        )

        # start training
        trainer.train()

        tokenizer = trainer.tokenizer
        config = trainer.config
        model = trainer.model

    if args.do_predict:
        test_data = load_json_data(args.test_data_path)

        output_dir = args.best_checkpoint_path if args.do_train else args.bert_model_name
        output_dir = os.path.join(*list(os.path.split(output_dir))[:-1])

        predictor = DistarPredictor(
            args=args,
            role_desc=role_description,
            tokenizer=tokenizer,
            config=config,
            model=model,
            output_dir=output_dir
        )

        pred_test_data = predictor.predict_instances(test_data, trigger_type=args.trigger_type)

        # pred_path = os.path.join(args.bert_model_name.replace("checkpoint", "prediction"),
        #                          "pred_test.gold_trigger.json")
        # pred_test_data = load_json_data(pred_path)

        # f1_metric = F1MetricForEAE()
        # metric_result = f1_metric.compute(
        #     predictions=pred_test_data,
        #     references=test_data,
        #     ignore_types=split_types["unseen_types"]
        # )
        # with open(os.path.join(output_dir, "metric_result_seen.json"), "w") as file:
        #     json.dump(metric_result, file, indent=2)

        # logger.info("[Metric Result] (Seen)")
        # logger.info(f"AI Task     : {metric_result['AI']}")
        # logger.info(f"AI + AC Task: {metric_result['AI+AC']}\n")

        f1_metric = F1MetricForEAE()
        metric_result = f1_metric.compute(
            predictions=pred_test_data,
            references=test_data,
            ignore_types=split_types["seen_types"]
        )
        with open(os.path.join(output_dir, "metric_result_unseen.json"), "w") as file:
            json.dump(metric_result, file, indent=2)

        logger.info("[Metric Result] (Unseen)")
        logger.info(f"AI Task     : {metric_result['AI']}")
        logger.info(f"AI + AC Task: {metric_result['AI+AC']}\n")

        # f1_metric = F1MetricForEAE()
        # metric_result = f1_metric.compute(
        #     predictions=pred_test_data,
        #     references=test_data,
        #     # ignore_types=split_types["seen_types"]
        # )
        # with open(os.path.join(output_dir, "metric_result.json"), "w") as file:
        #     json.dump(metric_result, file, indent=2)

        # logger.info("[Metric Result] (Overall)")
        # logger.info(f"AI Task     : {metric_result['AI']}")
        # logger.info(f"AI + AC Task: {metric_result['AI+AC']}\n")

        # pred_path = os.path.join(args.bert_model_name.replace("checkpoint", "prediction"),
        #                          "pred_test1.gold_trigger.json")
        # pred_test_data = load_json_data(pred_path)

        # f1_metric = F1MetricForEAE()
        # metric_result = f1_metric.compute(
        #     predictions=pred_test_data,
        #     references=test_data,
        #     ignore_types=split_types["unseen_types"]
        # )
        # with open(os.path.join(output_dir, "metric_result1_seen.json"), "w") as file:
        #     json.dump(metric_result, file, indent=2)
        #
        # logger.info("[Metric Result]")
        # logger.info(f"AI Task     : {metric_result['AI']}")
        # logger.info(f"AI + AC Task: {metric_result['AI+AC']}\n")
        #
        # f1_metric = F1MetricForEAE()
        # metric_result = f1_metric.compute(
        #     predictions=pred_test_data,
        #     references=test_data,
        #     ignore_types=split_types["seen_types"]
        # )
        # with open(os.path.join(output_dir, "metric_result1_unseen.json"), "w") as file:
        #     json.dump(metric_result, file, indent=2)
        #
        # logger.info("[Metric Result]")
        # logger.info(f"AI Task     : {metric_result['AI']}")
        # logger.info(f"AI + AC Task: {metric_result['AI+AC']}\n")
        #
        # f1_metric = F1MetricForEAE()
        # metric_result = f1_metric.compute(
        #     predictions=pred_test_data,
        #     references=test_data,
        #     # ignore_types=split_types["seen_types"]
        # )
        # with open(os.path.join(output_dir, "metric_result1.json"), "w") as file:
        #     json.dump(metric_result, file, indent=2)
        #
        # logger.info("[Metric Result]")
        # logger.info(f"AI Task     : {metric_result['AI']}")
        # logger.info(f"AI + AC Task: {metric_result['AI+AC']}\n")


if __name__ == "__main__":
    main()
