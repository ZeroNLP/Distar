import os
import gc
import json
from datetime import datetime
from argparse import Namespace
from typing import Optional, Union, List, Dict, Any, AnyStr

from pprint import pformat
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
from transformers import (BertConfig,
                          RobertaConfig,
                          BertTokenizerFast,
                          RobertaTokenizerFast)
from evaluate import EvaluationModule

from .utils.logger import logger
from .model.model import BertCrfKge, RobertaCrfKge, DistarModel
from .dataset.dataset import DistarDataset


class DistarTrainer(object):
    def __init__(self,
                 args: Namespace,
                 role_desc: Dict[AnyStr, Dict[AnyStr, AnyStr]],
                 tokenizer: Union[BertTokenizerFast, RobertaTokenizerFast],
                 config: Union[BertConfig, RobertaConfig],
                 model: Union[BertCrfKge, RobertaCrfKge],
                 optimizer: Optimizer,
                 scheduler: Optional[LambdaLR] = None,
                 metric: Optional[EvaluationModule] = None,
                 train_data: List[Dict[AnyStr, Any]] = None,
                 eval_data: List[Dict[AnyStr, Any]] = None,
                 split_types: Dict[AnyStr, Any] = None,
                 ) -> None:
        self.args = args
        self.role_desc = role_desc
        self.device = args.device
        self.tokenizer = tokenizer
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric = metric
        self.train_data = train_data
        self.eval_data = eval_data
        self.split_types = split_types
        self.curr_step = 0

        self.train_record = {"loss": []}
        self.eval_record = {"loss": [],
                            "crf_f1": [],
                            "crf_precision": [],
                            "crf_recall": [],
                            "crf_accuracy": [],
                            "kge_accuracy": []}

        logger.info(f"preparing evaluation dataset...")
        self.eval_dataset = DistarDataset(
            data=self.eval_data,
            tokenizer=self.tokenizer,
            role_desc=self.role_desc,
            tag2index=self.args.tag2index,
            index2tag=self.args.index2tag,
            trigger_left_token=self.args.trigger_left_token,
            trigger_right_token=self.args.trigger_right_token,
            max_input_length=self.args.max_input_length,
            max_role_length=self.args.max_role_length,
            num_neg_role=self.args.num_neg_role,
            split_types=self.split_types
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            drop_last=False,
            shuffle=False
        )

        time = datetime.now()
        time_str = time.strftime("%y%m%d-%H%M%S")
        bert_model_name = os.path.split(self.args.bert_model_name)[-1]
        self.output_name = f"{self.args.split_setting}_neg{self.args.num_neg_role}_" \
                           f"seed{self.args.seed}_{bert_model_name}-" \
                           f"{self.args.kge_scorer_name}-layer{self.args.num_role_encoder_layers}-" \
                           f"{self.args.triplet_comb}_" \
                           f"aug{self.args.num_aug_data if self.args.add_aug_data else 0}" \
                           f"_{time_str}"
        self.output_path = os.path.join(args.output_dir, self.output_name)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.checkpoint_path = os.path.join(self.output_path, "checkpoint")

        self.writer = SummaryWriter(log_dir=os.path.join(self.output_path, "log"))

        print()
        logger.info("Trainer for Distar")
        logger.info(f"{os.path.abspath(self.output_name)}")
        logger.info(f"args = \n{pformat(vars(self.args))}\n")

    def train(self) -> None:
        logger.info("----------- Start Training -----------")

        self.curr_step = 0
        self.log_weight_to_tensorboard()

        for curr_epoch in range(1, self.args.num_train_epochs + 1):
            self.model.train(True)

            logger.info(f"Start Epoch {curr_epoch}")

            train_dataset = DistarDataset(
                data=self.train_data,
                tokenizer=self.tokenizer,
                role_desc=self.role_desc,
                tag2index=self.args.tag2index,
                index2tag=self.args.index2tag,
                trigger_left_token=self.args.trigger_left_token,
                trigger_right_token=self.args.trigger_right_token,
                max_input_length=self.args.max_input_length,
                max_role_length=self.args.max_role_length,
                num_neg_role=self.args.num_neg_role,
                split_types=self.split_types
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                drop_last=False,
                shuffle=True
            )

            batch_num = len(train_dataloader)
            with tqdm(total=batch_num, desc=f"Epoch {curr_epoch}") as tq:
                for batch in train_dataloader:
                    self.optimizer.zero_grad()

                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)

                    loss = outputs.loss
                    kge_loss = outputs.kge_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.curr_step += 1

                    self.train_record["loss"].append(loss.item())

                    tag_scalar_dict = {
                        "train_loss": loss.item(),
                        "train_crf_loss": loss.item() - kge_loss.item(),
                        "train_kge_loss": kge_loss.item()
                    }
                    self.writer.add_scalars(main_tag="loss",
                                            tag_scalar_dict=tag_scalar_dict,
                                            global_step=self.curr_step)

                    if outputs.contra_loss is None:
                        tq.set_postfix(loss=loss.item(), kge_loss=kge_loss.item())
                    else:
                        tq.set_postfix(loss=loss.item(),
                                       kge_loss=kge_loss.item(),
                                       contra_loss=outputs.contra_loss.item())
                    tq.update()

                    if (self.curr_step % self.args.eval_step) == 0:
                        self.evaluate()
                        self.model.train(True)

                        self.log_weight_to_tensorboard()

                        if self.eval_record["crf_f1"][-1] == max(self.eval_record["crf_f1"]):
                            self.tokenizer.save_pretrained(self.checkpoint_path)
                            self.model.save_pretrained(self.checkpoint_path, safe_serialization=False)
                            self.args.best_checkpoint_path = self.checkpoint_path
                            with open(os.path.join(self.output_path, "args.json"), "w") as file:
                                json.dump(vars(self.args), file, indent=2)

                            logger.info(f"[Checkpoint] | save model checkpoint "
                                        f"in {os.path.abspath(self.checkpoint_path)}")

            del train_dataset, train_dataloader
            torch.cuda.empty_cache()
            gc.collect()

            logger.info(f"[Training] epoch {curr_epoch} | "
                        f"loss: {np.mean(self.train_record['loss'][-batch_num:])}\n")

        self.tokenizer, self.config, self.model = DistarModel.load_model(
            model_name_or_path=self.checkpoint_path,
            num_labels=len(self.args.tag2index),
            need_lstm=self.args.need_lstm,
            lstm_dim=self.args.lstm_dim,
            num_lstm_layer=self.args.num_lstm_layer,
            num_role_encoder_layers=self.args.num_role_encoder_layers,
            kge_scorer_name=self.args.kge_scorer_name,
            triplet_comb=self.args.triplet_comb,
            trigger_left_token=self.args.trigger_left_token,
            trigger_right_token=self.args.trigger_right_token
        )
        self.model = self.model.to(self.device)

        logger.info(f"training completed")
        logger.info(f"load best model from {self.checkpoint_path}")
        torch.cuda.empty_cache()
        gc.collect()

    def evaluate(self) -> None:
        self.model.eval()

        total_loss = 0.0
        crf_predictions = []
        kge_predictions = []
        crf_labels = []
        kge_labels = []

        with torch.no_grad():
            for batch in self.eval_dataloader:
                crf_label = batch["labels"].cpu().numpy().tolist()
                kge_label = batch["triplet_label"].reshape((-1,)).cpu().numpy().tolist()

                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)

                loss = outputs.loss

                kge_pred = (outputs.kge_probs.cpu() >= 0.5).long().numpy().tolist()
                crf_pred = self.model.predict_crf(
                    input_ids=batch["input_ids"],
                    token_type_ids=batch["token_type_ids"] if "token_type_ids" in batch.keys() else None,
                    attention_mask=batch["attention_mask"]
                )

                total_loss += loss.item() * self.eval_dataloader.batch_size
                crf_predictions.extend(crf_pred)
                kge_predictions.extend(kge_pred)
                crf_labels.extend(crf_label)
                kge_labels.extend(kge_label)

        kge_accuracy = accuracy_score(y_pred=kge_predictions, y_true=kge_labels) * 100
        crf_results = self.compute_metrics(crf_predictions, crf_labels)
        crf_precision = crf_results["precision"]
        crf_recall = crf_results["recall"]
        crf_f1 = crf_results["f1"]
        crf_accuracy = crf_results["accuracy"]

        eval_loss = total_loss / len(self.eval_dataset)
        self.eval_record["loss"].append(eval_loss)
        self.eval_record["crf_f1"].append(crf_f1)
        self.eval_record["crf_precision"].append(crf_precision)
        self.eval_record["crf_recall"].append(crf_recall)
        self.eval_record["crf_accuracy"].append(crf_accuracy)
        self.eval_record["kge_accuracy"].append(kge_accuracy)

        self.writer.add_scalars(main_tag="loss", tag_scalar_dict={"eval_loss": eval_loss}, global_step=self.curr_step)
        self.writer.add_scalars(
            main_tag="eval_metric",
            tag_scalar_dict={
                "crf_f1": crf_f1,
                "crf_precision": crf_precision,
                "crf_recall": crf_recall,
                "crf_accuracy": crf_accuracy,
                "kge_accuracy": kge_accuracy
            },
            global_step=self.curr_step
        )

        logger.info(f"[Evaluation] | loss: {loss: .{8}}, "
                    f"crf_f1: {crf_f1: .{8}}, crf_prec: {crf_precision: .{8}}, crf_rec: {crf_recall: .{8}}, "
                    f"kge_acc: {kge_accuracy: .{8}}")

    def compute_metrics(self,
                        predictions: List[List[int]],
                        labels: List[List[int]]
                        ) -> Dict[AnyStr, int]:
        tag_predictions = [
            [self.args.index2tag[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]
        tag_labels = [
            [self.args.index2tag[lab] for (pred, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=tag_predictions, references=tag_labels, zero_division=0)

        return {
            "precision": results["overall_precision"] * 100,
            "recall": results["overall_recall"] * 100,
            "f1": results["overall_f1"] * 100,
            "accuracy": results["overall_accuracy"] * 100,
        }

    def log_weight_to_tensorboard(self) -> None:
        self.writer.add_histogram(
            tag="word_emb",
            values=self.model.kge_model.embeddings.word_embeddings.weight,
            global_step=self.curr_step
        )
        self.writer.add_histogram(
            tag="pos_emb",
            values=self.model.kge_model.embeddings.position_embeddings.weight,
            global_step=self.curr_step
        )
        self.writer.add_histogram(
            tag="tok_emb",
            values=self.model.kge_model.embeddings.word_embeddings.weight,
            global_step=self.curr_step
        )
        self.writer.add_histogram(
            tag="crf_transition",
            values=self.model.crf.transitions,
            global_step=self.curr_step
        )
