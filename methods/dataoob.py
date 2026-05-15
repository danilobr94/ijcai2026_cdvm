"""DataOob variant that accepts custom training kwargs (e.g. epochs) passed to model.fit().

Adapted from the OpenDataVal library (https://github.com/opendataval/opendataval),
originally introduced in Kwon & Zou, "Data-OOB: Out-of-bag Experiment for Data Valuation",
ICML 2023.
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Subset
from opendataval.dataval.api import DataEvaluator, ModelMixin


class DataOobALT(DataEvaluator, ModelMixin):
    """Data Out-of-Bag data valuation (Kwon & Zou, 2023).

    Extends the original DataOob implementation with support for custom training
    kwargs (e.g. a fixed epoch count) passed directly to model.fit().

    References
    ----------
    .. [1] Y. Kwon and J. Zou,
        Data-OOB: Out-of-bag Estimate as a Simple and Efficient Data Value,
        arXiv:2304.07718, 2023.

    Parameters
    ----------
    num_models : int
        Number of bagged models, by default 1000.
    proportion : float
        Fraction of training points sampled per bag, by default 1.0.
    random_state : RandomState, optional
        Random seed, by default None.
    alt_train_kwargs : dict, optional
        If given, passed as keyword arguments to model.fit() instead of the
        caller-supplied args/kwargs. Useful for fixing epochs independently of
        the outer training loop.
    """

    def __init__(
        self,
        num_models: int = 1000,
        proportion: float = 1.0,
        random_state: Optional[RandomState] = None,
        alt_train_kwargs: Optional[dict] = None,
    ):
        self.num_models = num_models
        self.proportion = proportion
        self.random_state = check_random_state(random_state)
        self.alt_train_kwargs = alt_train_kwargs

    def __repr__(self) -> str:
        if self.alt_train_kwargs is not None and "epochs" in self.alt_train_kwargs:
            return f"DataOob({self.num_models}, epochs={self.alt_train_kwargs['epochs']})"
        return f"DataOob({self.num_models})"

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store training data; validation split is not used by DataOob."""
        self.x_train = x_train
        self.y_train = y_train
        _ = x_valid, y_valid  # Unused parameters

        self.num_points = len(x_train)
        [*self.label_dim] = (1,) if self.y_train.ndim == 1 else self.y_train[0].shape
        self.max_samples = round(self.proportion * self.num_points)

        self.oob_pred = torch.zeros((0, *self.label_dim), requires_grad=False)
        self.oob_indices = GroupingIndex()
        return self

    def train_data_values(self, *args, **kwargs):
        """Bag ``num_models`` models and collect out-of-bag predictions for each point."""
        sample_dim = (self.num_models, self.max_samples)
        subsets = self.random_state.randint(0, self.num_points, size=sample_dim)

        for i in tqdm.tqdm(range(self.num_models)):
            in_bag = subsets[i]

            # out_bag is the indices where the bincount is zero.
            out_bag = (np.bincount(in_bag, minlength=self.num_points) == 0).nonzero()[0]
            if not out_bag.any():
                continue

            curr_model = self.pred_model.clone()

            if self.alt_train_kwargs is not None:
                curr_model.fit(
                    Subset(self.x_train, indices=in_bag),
                    Subset(self.y_train, indices=in_bag),
                    **self.alt_train_kwargs
                )

            else:
                curr_model.fit(
                    Subset(self.x_train, indices=in_bag),
                    Subset(self.y_train, indices=in_bag),
                    *args,
                    **kwargs,
                )

            y_hat = curr_model.predict(Subset(self.x_train, indices=out_bag))
            self.oob_pred = torch.cat((self.oob_pred, y_hat.detach().cpu()), dim=0)
            self.oob_indices.add_indices(out_bag)

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return per-point data values as the mean OOB prediction score."""
        self.data_values = np.zeros(self.num_points)

        for i, indices in self.oob_indices.items():
            oob_labels = self.y_train[i].expand((len(indices), *self.label_dim))
            self.data_values[i] = self.evaluate(oob_labels, self.oob_pred[indices])

        return self.data_values


class GroupingIndex(defaultdict[int, list[int]]):
    """Maps each data index to a list of positions in the global OOB prediction stack."""

    def __init__(self, start: int = 0):
        super().__init__(list)
        self.position = start

    def add_indices(self, values: list[int]):
        """Append the current stack position for each index in ``values``."""
        for i in values:
            self.__getitem__(i).append(self.position)
            self.position += 1