import torch
import torch.nn as nn


class TransE(nn.Module):
    def __init__(self,
                 gamma: float
                 ) -> None:
        super(TransE, self).__init__()
        self.model_name = "TransE"
        self.gamma = gamma

    def forward(self,
                head: torch.Tensor,
                tail: torch.Tensor,
                relation: torch.Tensor
                ) -> torch.Tensor:
        score = head + relation - tail
        score = self.gamma - torch.norm(score, p=1, dim=1)

        return score


class DistMult(nn.Module):
    def __init__(self) -> None:
        super(DistMult, self).__init__()
        self.model_name = "DistMult"

    def forward(self,
                head: torch.Tensor,
                tail: torch.Tensor,
                relation: torch.Tensor
                ) -> torch.Tensor:
        score = head * tail * relation
        score = score.sum(dim=-1)

        return score


class ComplEx(nn.Module):
    def __init__(self) -> None:
        super(ComplEx, self).__init__()
        self.model_name = "ComplEx"

    def forward(self,
                head: torch.Tensor,
                tail: torch.Tensor,
                relation: torch.Tensor
                ) -> torch.Tensor:
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_relation, im_relation = torch.chunk(relation, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = re_score * re_tail + im_score * im_tail
        score = score.sum(dim=-1)

        return score


class RotatE(nn.Module):
    def __init__(self,
                 gamma: float,
                 hidden_size: int
                 ) -> None:
        super(RotatE, self).__init__()
        self.model_name = "RotatE"
        self.gamma = gamma
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=hidden_size // 2)

    def forward(self,
                head: torch.Tensor,
                tail: torch.Tensor,
                relation: torch.Tensor
                ) -> torch.Tensor:
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        proj_relation = self.linear(relation)
        re_relation = torch.cos(proj_relation)
        im_relation = torch.sin(proj_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        score = self.gamma - score.sum(dim=-1)

        return score
