import os
from copy import deepcopy
from opendataval.model import ModelFactory

import torch
from typing import Union, Optional, Literal
from sklearn.utils import check_random_state
from numpy.random import RandomState
import numpy as np
from opendataval.experiment import ExperimentMediator
from opendataval.dataloader import mix_labels

from tqdm import tqdm
from opendataval.dataval.api import DataEvaluator, ModelMixin


class DataAttributionRunner(DataEvaluator, ModelMixin):

    def __init__(
        self,
        file_path,
        num_models: Union[int, str] = 1000,
        prob: Union[float, Literal["random"]] = "random",
        random_state: Optional[RandomState] = None,
    ):
        super().__init__()

        self.file_path = file_path
        self.num_models = num_models
        self.prob = prob
        self.random_state = check_random_state(random_state)
        self.tran_new_models = False

    def __repr__(self) -> str:
        return f"DataAttrRunner({self.num_models})"

    def input_data(self, x_train: torch.Tensor, y_train: torch.Tensor,
                   x_valid: torch.Tensor, y_valid: torch.Tensor,
    ):
        """Store and transform input data."""
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.num_train_points = len(x_train)
        self.num_test_points = len(x_valid)

        return self

    def get_accuracy_and_true_predictions(self, clf):
        """Get predictions from model."""
        pred = clf.predict(self.x_valid)
        pred_np = np.argmax(pred.cpu().numpy(), axis=1)
        pred_true = pred_np == np.argmax(self.y_valid.cpu().numpy(), axis=1)
        acc = self.evaluate(pred, self.y_valid)

        return acc, pred_true

    def fit_clf(self, clf, x, y, *args, **kwargs):
        """Fit classifier."""
        clf.fit(x, y, *args, **kwargs)
        return clf

    def get_subsets(self):
        """Get subsets for training."""
        num_points = len(self.x_train)

        if self.prob == "random":
            dim = num_points
            subsets = np.array([self.random_state.binomial(1, self.random_state.uniform(0.05), dim) for _ in range(self.num_models)])
        else:
            sample_dim = (self.num_models, num_points)
            subsets = self.random_state.binomial(1, self.prob, size=sample_dim)

        return subsets

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values."""

        max_acc = 0
        num_points = len(self.x_train)
        subsets = self.get_subsets()  # shape: (num_models, num_train_points)

        sample_utility_per_train_test_idx = np.zeros((num_points, 2, len(self.x_valid)), dtype=np.ushort)
        sample_counts_per_train_test_idx = np.zeros((num_points, 2, len(self.x_valid)), dtype=np.ushort)

        for i in range(self.num_models):
            s = subsets[i]
            clf = self.pred_model.clone()
            clf = self.fit_clf(clf, self.x_train[s.nonzero()[0]], self.y_train[s.nonzero()[0]],
                               *args, **kwargs)

            acc, pred_true = self.get_accuracy_and_true_predictions(clf)  # pred_true has shape: (num_test_points,)
            max_acc = max(max_acc, acc)

            sample_utility_per_train_test_idx[range(num_points), s] += pred_true
            sample_counts_per_train_test_idx[range(num_points), s] += np.ones_like(pred_true)

        msr_tt = np.divide(sample_utility_per_train_test_idx, sample_counts_per_train_test_idx)

        self.attr_matrix = msr_tt[:, 1] - msr_tt[:, 0]  # shape (num_train_points, num_test_points)
        np.save(self.file_path, self.attr_matrix)

        print(f"Max accuracy during training: {max_acc}")
        return self

    def evaluate_data_values(self) -> np.ndarray:
        raise NotImplementedError("This method should be implemented in subclasses.")


def train_and_store_attr_matrix(exper_med, num_models, prob, pth, pbar=None):
    """"""
    dves = [DataAttributionRunner(
        file_path=pth,
        num_models=num_models,
        prob=prob,
    )]

    if pbar is not None:
        pbar.set_description(f"Saved attr matrix for {pth}")

    exper_med.data_evaluators = []
    exper_med = exper_med.compute_data_values(data_evaluators=dves)

    return pth

def maybe_train_and_store_attr_matrix(exper_med, n_models, prob, pbar=None, postfix=""):
    epchs = exper_med.train_kwargs["epochs"]
    pth_mat = f"data/attr_matr/{exper_med.fetcher.dataset.dataset_name}-{n_models}_{prob}_{epchs}{postfix}.npy"
    pth_weights = f"data/weights/{exper_med.fetcher.dataset.dataset_name}{postfix}.pth"

    if os.path.exists(pth_weights):
        exper_med.pred_model.load_state_dict(torch.load(pth_weights))
        print("Loaded initial weights")

    if not os.path.exists(pth_mat):
        train_and_store_attr_matrix(exper_med, n_models, prob, pth_mat, pbar)

        if not os.path.exists(pth_weights):
            torch.save(exper_med.pred_model.state_dict(), pth_weights)

    return pth_mat, pth_weights

def run_dataset_once(dataset_name, num_epochs, n_models, prob, pbar=None, postfix=""):
    train_count, valid_count, test_count = 1000, 500, 500
    noise_rate = 0.1
    noise_kwargs = {'noise_rate': noise_rate}
    model_name = "LogisticRegression"
    metric_name = "accuracy"
    train_kwargs = {"epochs": num_epochs, "batch_size": 250, "lr": 0.01}

    device = torch.device("cuda")

    exper_med = ExperimentMediator.model_factory_setup(
        dataset_name=dataset_name,
        cache_dir="./data_files/",
        force_download=False,
        train_count=train_count,
        valid_count=valid_count,
        test_count=test_count,
        add_noise=mix_labels,
        noise_kwargs=noise_kwargs,
        train_kwargs=train_kwargs,
        model_name=model_name,
        metric_name=metric_name,
        random_state=42,
        device=device,
    )

    return maybe_train_and_store_attr_matrix(exper_med, n_models, prob, pbar=pbar, postfix=postfix)


if __name__ == "__main__":
    # Example usage
    probs = ["random", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    probs = [0.1]
    num_models = [2]
    dataset_names = ["cifar10-embeddings", "imdb-embeddings", "bbc-embeddings", "adult", "pol", "nomao"]
    EPOCHS = 15

    import os

    # supress warnings
    import warnings
    warnings.filterwarnings("ignore")

    # redirect stdout to devnull to suppress tqdm output
    # with open(os.devnull, "w") as outer_space, redirect_stdout(outer_space):
    for d in (pbar:= tqdm(dataset_names)):
        for num_model in num_models:
            for prob in probs:
                run_dataset_once(d, EPOCHS, num_model, prob, pbar=None)



