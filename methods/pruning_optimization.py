"""Data pruning via influence-based LP optimization. (Yang et al., "Dataset Pruning:
Reducing Training Data by Examining Generalization Influence", ICLR 2023).
Code adapted from https://shuoyang-1998.github.io/assets/code/code_datasetptuning.zip
"""
from typing import Optional, Union, Literal
import os

import numpy as np
import torch
from tqdm import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.autograd import grad
from torch.nn.utils import parameters_to_vector
import cvxpy as cp

from opendataval.model import GradientModel
from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.model.mlp import ClassifierMLP
from opendataval.model.logistic_regression import LogisticRegression


def value_guided_opt(S, size):
    """Select exactly ``size`` points that minimise the influence residual norm."""
    n, m = S.shape
    W = cp.Variable(n - 1)

    obj = cp.Minimize(cp.norm(W @ S[:-1], 2))
    constraints = [cp.sum(W) == size, W >= 0, W <= 1]

    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=True, solver=cp.GUROBI)
    W_optim = W.value

    return W_optim


def size_guided_opt(S, epsilon):
    """Maximise the number of selected points while keeping the influence residual below ``epsilon``."""
    n, m = S.shape
    W = cp.Variable(n)

    obj = cp.Maximize(cp.sum(W))
    constraints = [cp.norm(W@S, 2) <= epsilon, W >= 0, W <= 1]

    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=True, solver=cp.GUROBI)
    W_optim = W.value

    return W_optim


class PruningOptimization(DataEvaluator, ModelMixin):
    """Influence-based data pruning via LP optimization.

    Computes per-sample influence scores using implicit Hessian-vector products,
    then solves a linear program to find the optimal subset under either a size
    constraint (``optimization_type="size"``) or an influence-residual constraint
    (``optimization_type="value"``). Requires a Gurobi license.

    Parameters
    ----------
    num_models : ignored
        Accepted for API compatibility; has no effect.
    optimization_type : {"value", "size"}
        Whether to minimize retained set size (``"size"``) or maximize data value
        (``"value"``), by default ``"size"``.
    random_state : RandomState, optional
        Random seed for influence estimation, by default None.
    rerun : bool
        If True, recompute the influence score matrix even when a cached file
        exists, by default False.
    only_size : int or False
        If an integer, run a single value-guided LP for that exact subset size
        instead of the default multi-size ensemble, by default False.
    dataset_name : str
        Appended to the influence score cache filename, by default ``""``.
    """

    def __init__(
            self,
            num_models=None,
            optimization_type: Literal["value", "size"] = "size",
            random_state: Optional[RandomState] = None,
            rerun: bool = False,
            only_size=False,
            dataset_name: str = "",
    ):
        super().__init__()
        self.num_models = num_models
        self.optimization_type = optimization_type
        self.random_state = check_random_state(random_state)
        self.rerun = rerun
        self.only_size = only_size
        self.dataset_name = dataset_name

    def __repr__(self) -> str:
        return f"PruningOptimization(optimization_type={self.optimization_type}, only_size={self.only_size})"

    def input_data(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            x_valid: torch.Tensor,
            y_valid: torch.Tensor,
    ):
        """Store training and validation data."""
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.num_train_points = len(x_train)
        self.num_test_points = len(x_valid)

        return self

    def input_model(self, pred_model: GradientModel):
        """Store ``pred_model``; asserts it exposes a ``grad`` method."""
        assert (  # In case model doesn't inherit but still wants the grad function
                isinstance(pred_model, GradientModel)
                or callable(getattr(pred_model, "grad"))
        ), "Model with gradient required."

        self.pred_model = pred_model.clone()
        return self

    def _get_influence_score_list(self, r=3, *args, **kwargs):
        """Return influence score matrix of shape (num_train_points, num_params).

        Each row is the mean of ``r`` stochastic s_train estimates for that point.
        """

        features, classifier = self._maybe_extract_features(*args, **kwargs)
        kwargs = {"damp": 0.01, "scale": 25, "recursion_depth": 5000}

        influence_score_list = []
        for i in tqdm(range(self.num_train_points)):
            s_train_vec_list = [s_train(classifier, features, self.y_train, i,  **kwargs) for _ in range(r)]

            # Take average over 'r' repetitions
            influence_score_list.append(np.array(s_train_vec_list).mean(0))
            print(i)

        return np.array(influence_score_list)

    def _maybe_extract_features(self, *args, **kwargs):
        """Extract features from input data if model is an mlp."""
        if isinstance(self.pred_model, ClassifierMLP):
            clf = self.pred_model.clone()
            clf.fit(self.x_train, self.y_train, *args, **kwargs)

            features_train = clf.mlp[:-2](self.x_train).detach()
            features_valid = clf.mlp[:-2](self.x_valid).detach()

            feature_clf = LogisticRegression(features_train.shape[1], self.y_train.shape[1]).to(self.pred_model.device).clone()
            feature_clf.fit(features_train, self.y_train, *args, **kwargs)
            print("Extracted feature classifier with performance: ", self.evaluate(self.y_valid, feature_clf(features_valid)))

            return features_train, feature_clf

        clf = self.pred_model.clone()
        clf.fit(self.x_train, self.y_train, *args, **kwargs)
        print("Baseline classifier performance: ", self.evaluate(self.y_valid, clf(self.x_valid)))
        return self.x_train, clf

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values."""
        pth = f"./data/influence_score_matrix-{type(self.pred_model).__name__}{self.dataset_name}.npy"

        if os.path.exists(pth) and not self.rerun:
            influence_score_matrix = np.load(pth)
            print("Loaded influence score matrix with shape: ", influence_score_matrix.shape)

        else:
            print("Calculating influence score matrix ...")
            influence_score_matrix = self._get_influence_score_list(*args, **kwargs)  # (num_train_points, num_params)
            np.save(pth, influence_score_matrix)
            print("Computed influence score matrix with shape: ", influence_score_matrix.shape)

        print("Optimizing data values...")

        # TODO: this returns only one set of samples, but we need an ordering ...
        if self.optimization_type == "value":
            self.data_values = size_guided_opt(influence_score_matrix, dv_threshold=0.05)

        elif self.only_size:
            W = value_guided_opt(influence_score_matrix, size=self.only_size)
            self.data_values = W

        else:
            W_100 = value_guided_opt(influence_score_matrix, size=100)
            W_200 = value_guided_opt(influence_score_matrix, size=200)
            W_500 = value_guided_opt(influence_score_matrix, size=500)
            W_700 = value_guided_opt(influence_score_matrix, size=700)

            self.data_values = W_100 + W_200 + W_500 + W_700

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point."""
        return self.data_values


def calc_loss(y, t):
    """Return NLL loss of logit predictions ``y`` against targets ``t``.

    Handles 1-D predictions and one-hot targets by converting to class indices.
    """

    # add batch dim maybe
    if y.dim() == 1:
        y = y.unsqueeze(0)

    # If t is one-hot encoded and one value, convert to class index
    if t.size(0) != y.size(0):
        t = t.argmax(dim=0).unsqueeze(0)

    # If t is one-hot encoded, convert to class index
    if t.dim() > 1:
        t = t.argmax(dim=1)

    y = torch.nn.functional.log_softmax(y, dim=1)
    loss = torch.nn.functional.nll_loss(y, t, weight=None, reduction='mean')

    return loss


def grad_z(z, t, model):
    """Return per-parameter gradients of the loss on sample ``(z, t)``."""

    model.eval()

    # if model is on gpu, put data also on gpu
    if model.device.type == "cuda":
        z, t = z.cuda(), t.cuda()

    y = model(z)
    loss = calc_loss(y, t)

    # Compute sum of gradients from model parameters to loss
    params = [ p for p in model.parameters() if p.requires_grad ]

    return list(grad(loss, params, create_graph=True))


def hvp(y, w, v):
    """Return the Hessian-vector product H(y, w) · v via two backprop passes."""

    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads


def s_train(model, x_train, y_train, index, damp=0.01, scale=25.0, recursion_depth=5000):
    """Estimate S_train_i = -H^{-1} ∇L(z_i) via stochastic LiSSA recursion.

    See Sec. 3 of Koh & Liang (2017), arXiv:1703.04730.
    """

    v = grad_z(x_train[index], y_train[index], model)
    h_estimate = v.copy()
    h_estimate_copy = h_estimate.copy()

    # take one random sample from train set in each recursion step
    index = np.random.choice(len(x_train), recursion_depth)

    for i in range(recursion_depth):

        x, y = x_train[index[i]], y_train[index[i]]
        y_hat = model(x)

        loss = calc_loss(y_hat, y)
        params = [p for p in model.parameters() if p.requires_grad]
        hv = hvp(loss, params, h_estimate)

        # Recursively calculate h_estimate
        with torch.no_grad():
            h_estimate = [_v + (1 - damp) * _h_e - _hv / scale for _v, _h_e, _hv in zip(v, h_estimate, hv)]

        diff = parameters_to_vector(tuple(h_estimate)) - parameters_to_vector(tuple(h_estimate_copy))
        a = (torch.norm(diff, 2).item())

        if a < 1.5e-5:
            return parameters_to_vector(tuple(h_estimate)).cpu().numpy()

        h_estimate_copy = h_estimate.copy()

    return parameters_to_vector(tuple(h_estimate)).cpu().numpy() * -1


if __name__ == "__main__":
    from opendataval.dataloader import mix_labels
    from opendataval.experiment import ExperimentMediator
    from opendataval.dataloader import DataFetcher

    from utils_run import run_multiple_models

    NUM_EXPERIMENTS = 10
    num_models = [1000, 2000]

    dataset_name = "cifar10-embeddings"
    train_count, valid_count, test_count = 1000, 500, 500
    noise_rate = 0.1
    noise_kwargs = {'noise_rate': noise_rate}

    data_fetcher = DataFetcher(dataset_name, force_download=False)
    covar_dim, label_dim = data_fetcher.covar_dim, data_fetcher.label_dim

    model_kwargs = {"layers": 3, "hidden_dim": 100}
    pred_model = ClassifierMLP(*covar_dim, *label_dim, **model_kwargs)

    metric_name = "accuracy"
    train_kwargs = {"epochs": 15, "batch_size": 250, "lr": 0.01}
    device = torch.device("cpu")

    exper_med = ExperimentMediator.setup(
        dataset_name=dataset_name,
        cache_dir="../data_files/",
        force_download=False,
        train_count=train_count,
        valid_count=valid_count,
        test_count=test_count,
        add_noise=mix_labels,
        noise_kwargs=noise_kwargs,
        train_kwargs=train_kwargs,
        pred_model=pred_model,
        metric_name=metric_name,
        random_state=42,
        # device=device,
    )

    dve_kwargs = {"optimization_type": "size", "rerun": False}
    run_multiple_models(exper_med, PruningOptimization, num_models, NUM_EXPERIMENTS, dve_kwargs, check_available=False)