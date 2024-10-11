from collections import namedtuple
from typing import Dict, Any, AnyStr, List, Optional, Union, Tuple

argument_fields = ["sentence", "event_type", "trigger", "text", "start", "end", "role"]
Argument = namedtuple('Argument', field_names=argument_fields)


class F1MetricForEAE(object):
    def __init__(self):
        self.gold_argument_num = 0
        self.pred_argument_num = 0
        self.match_argument_num = 0
        self.match_role_num = 0

    def compute(self,
                predictions: List[Dict[AnyStr, Any]] = None,
                references: List[Dict[AnyStr, Any]] = None,
                ignore_types: Optional[List[AnyStr]] = None
                ) -> Dict[AnyStr, Any]:
        self.zero_record()

        samples = []

        for pred_inst, gold_inst in zip(predictions, references):
            pred_events = pred_inst["pred_event_mentions"]
            gold_events = gold_inst["event_mentions"]

            pred_arguments = []
            gold_arguments = []

            for event in pred_events:
                if (ignore_types is not None) and (event["event_type"] in ignore_types):
                    continue
                for argument_inst in event["arguments"]:
                    argument = Argument(
                        sentence=pred_inst["sentence"],
                        event_type=event["event_type"],
                        trigger=event["trigger"],
                        text=argument_inst["text"],
                        start=argument_inst["start"],
                        end=argument_inst["end"],
                        role=argument_inst["role"]
                    )
                    pred_arguments.append(argument)

            for event in gold_events:
                if (ignore_types is not None) and (event["event_type"] in ignore_types):
                    continue
                for argument_inst in event["arguments"]:
                    argument = Argument(
                        sentence=gold_inst["sentence"],
                        event_type=event["event_type"],
                        trigger=event["trigger"],
                        text=argument_inst["text"],
                        start=argument_inst["start"],
                        end=argument_inst["end"],
                        role=argument_inst["role"]
                    )
                    gold_arguments.append(argument)

            self.pred_argument_num += len(pred_arguments)
            self.gold_argument_num += len(gold_arguments)

            for gold_argument in gold_arguments:
                if self.exist_identification_match(gold_argument, pred_arguments):
                    self.match_argument_num += 1
                    if self.exist_classification_match(gold_argument, pred_arguments):
                        self.match_role_num += 1
                    else:
                        samples.append({"pred": pred_arguments, "gold": gold_argument})

        precision_ai, recall_ai, f1_ai = self.compute_f1(
            predicted=self.pred_argument_num,
            gold=self.gold_argument_num,
            matched=self.match_argument_num
        )
        precision_ac, recall_ac, f1_ac = self.compute_f1(
            predicted=self.pred_argument_num,
            gold=self.gold_argument_num,
            matched=self.match_role_num
        )

        result = {
            "AI": {"precision": precision_ai,
                   "recall": recall_ai,
                   "f1": f1_ai},
            "AI+AC": {"precision": precision_ac,
                      "recall": recall_ac,
                      "f1": f1_ac}
        }

        return result

    def zero_record(self):
        self.gold_argument_num = 0
        self.pred_argument_num = 0
        self.match_argument_num = 0
        self.match_role_num = 0

    @staticmethod
    def match_argument_identification(pred_argument: Argument,
                                      gold_argument: Argument
                                      ) -> bool:
        return (pred_argument.event_type == gold_argument.event_type
                and pred_argument.trigger == gold_argument.trigger
                and pred_argument.start == gold_argument.start
                and pred_argument.end == gold_argument.end
                and pred_argument.text == pred_argument.text)

    @staticmethod
    def match_argument_classification(pred_argument: Argument,
                                      gold_argument: Argument
                                      ) -> bool:
        return (pred_argument.event_type == gold_argument.event_type
                and pred_argument.trigger == gold_argument.trigger
                and pred_argument.start == gold_argument.start
                and pred_argument.end == gold_argument.end
                and pred_argument.text == pred_argument.text
                and pred_argument.role == gold_argument.role)

    def exist_identification_match(self,
                                   gold_argument: Argument,
                                   pred_arguments: List[Argument]
                                   ) -> bool:
        for pred_argument in pred_arguments:
            if self.match_argument_identification(pred_argument, gold_argument):
                return True

        return False

    def exist_classification_match(self,
                                   gold_argument: Argument,
                                   pred_arguments: List[Argument]
                                   ) -> bool:
        for pred_argument in pred_arguments:
            if self.match_argument_classification(pred_argument, gold_argument):
                return True

        return False

    @staticmethod
    def safe_division(numerator: Union[int, float],
                      denominator: Union[int, float]
                      ) -> float:
        if numerator > 0:
            return min(numerator / denominator, 1.0)
        else:
            return 0.0

    def compute_f1(self,
                   predicted: Union[int, float],
                   gold: Union[int, float],
                   matched: Union[int, float]
                   ) -> Tuple[Union[int, float],
                              Union[int, float],
                              Union[int, float]]:
        precision = self.safe_division(matched, predicted)
        recall = self.safe_division(matched, gold)
        f1_score = self.safe_division(2 * precision * recall, precision + recall)

        return precision, recall, f1_score
