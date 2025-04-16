r"""Implement a single gate network.
Given two hidden state tensor (from two different teacher layer),
Return aggregate hidden state tensor.
Note
---------
B: batch size
S: sequence length
D: dimension
"""
# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 3rd party modules

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


class HighwayGate(nn.Module):
    r"""Gate network implementation.
    Given two hidden state tensor (from two different teacher layer),
    Return aggregate hidden state tensor.
    Our implementation refer to `Highway Network`[1]

    Notes
    ----------
    [1]Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015).
    Highway networks. arXiv preprint arXiv:1505.00387.

    Parameters
    ----------
    dimension : int
        hidden state dimension
    seq_length : int
        sequence length, we need this info to set LayerNorm module
    """

    def __init__(self, dimension: int, seq_length: int):
        super().__init__()
        self.linear = nn.Linear(in_features=dimension, out_features=dimension)
        self.activation = nn.Sigmoid()
        self.layernorm = nn.LayerNorm(normalized_shape=[seq_length, dimension])

        # Xavier norm init.
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor):
        """Calculate aggregate hidden state tensor.

        Parameters
        ----------
        input1 : torch.Tensor
            Hidden state of i-th transformer layer
        input2 : torch.Tensor
            Hidden state of (i+1)-th transformer layer
        """
        if input1.shape != input2.shape:
            raise ValueError("Shape of two input dosen't match")

        if torch.all(torch.eq(input1, 0)):
            return input2
        transform_gate = self.activation(self.linear(input1))

        output = input2 * transform_gate + input1 * (1-transform_gate)

        return self.layernorm(output)


class DynamicGate(nn.Module):
    def __init__(self, dimension: int, seq_length: int, k: int = 3):
        super().__init__()
        self.k = k
        self.align_proj = nn.Linear(dimension, dimension)
        self.activation = nn.Sigmoid()
        self.layernorm = nn.LayerNorm([seq_length, dimension])
        nn.init.xavier_uniform_(self.align_proj.weight)

    def forward(self, student_h: torch.Tensor, teacher_hs: List[torch.Tensor]):
        """Dynamic top-k teacher layer selection"""
        # Compute similarity scores
        similarities = [
            F.cosine_similarity(
                self.align_proj(student_h),
                t_h,
                dim=-1
            ).mean()  # Average across sequence
            for t_h in teacher_hs
        ]

        # Select top-k layers
        topk_values, topk_indices = torch.topk(
            torch.stack(similarities),
            k=self.k
        )

        # Aggregate with softmax weights
        weights = F.softmax(topk_values, dim=0)
        aggregated = sum(
            weights[i] * teacher_hs[idx]
            for i, idx in enumerate(topk_indices)
        )

        return self.layernorm(aggregated)