"""Data valuation (pruning) by optimization of data attribution matrix."""
import warnings
from datetime import datetime
from typing import Optional, Union, Literal
from tqdm import tqdm
import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state
import torch
import cvxpy as cp

from opendataval.dataval.api import DataEvaluator, ModelMixin


def max_sum_var(S, size, max_val, alpha=0.8, l1_lambda=0.0):
    """Solve LP: maximise sum of attribution values subject to a soft cap at max_val.

    Objective: alpha * sum(w @ S) - (1-alpha) * sum(excess above max_val) - l1_lambda * ||w||_1
    w is relaxed to [0, 1] (not binary) and must sum to `size`.
    """
    n, m = S.shape
    w = cp.Variable(n)
    t = cp.Variable(m)
    values = cp.Variable(m)

    print(f"Solving LP: max_val={max_val}, size={size}, alpha={alpha}, l1_lambda={l1_lambda}")

    constraints = [
        values == w @ S,
        cp.sum(w) == size,
        t >= values - max_val,
        t >= 0,
        w >= 0,
        w <= 1,
    ]
    obj = cp.Maximize(alpha * cp.sum(values) - (1 - alpha) * cp.sum(t) - l1_lambda * cp.norm(w, 1))

    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=False, solver=cp.GUROBI)

    return w.value




class CDVM(DataEvaluator, ModelMixin):
    """Constrained Data Value Maximization (CDVM).

    Trains an ensemble of models on random subsets to build a train×test attribution
    matrix, then solves a linear program to select the most valuable training points.

    Parameters
    ----------
    num_models_or_path : int or str
        Number of random-subset models to train, or path to a precomputed `.npy`
        attribution matrix. When a path is given, training is skipped.
    prob : float or "random" or "frac"
        Inclusion probability for each training point per subset.
        ``"random"`` draws a fresh uniform probability per model.
        ``"frac"`` derives the probability from ``retention_budget / n_train``
        (requires ``retention_budget`` to be set).
    random_state : RandomState, optional
        Seed for subset sampling.
    alt_train_kwargs : dict, optional
        If set, these kwargs are forwarded to ``clf.fit`` instead of the defaults.
        Set ``{"epochs": "random"}`` to randomise the number of training epochs.
    alpha : float or "best"
        LP objective weight in [0, 1]: ``alpha`` on total attribution sum,
        ``1 - alpha`` on soft-cap violations. ``"best"`` triggers a grid search
        over candidate values.
    retention_budget : int or None
        Number of training points to retain (the LP selection budget). Must be a
        positive integer to enable LP optimisation. ``None`` (default) disables the
        LP; ``evaluate_data_values`` will raise if called without a budget set.
    max_val : float, str, or "best"
        Soft cap on per-test-point attribution in the LP objective. Accepts a
        numeric value or one of: ``"max"``, ``"mean"``, ``"max±Nstd"``,
        ``"max+s"``, ``"max+2s"``, ``"3s"``. ``"best"`` triggers a grid search.
    use_banzhaf : bool
        If ``True``, bypass the LP entirely and return the mean attribution per
        training point (equivalent to a Banzhaf value baseline).
    add_banzhaf : bool
        If ``True``, add the mean attribution scores to the LP weights before
        returning, blending LP and Banzhaf signals.
    dataset_name : str
        Prefix for output ``.npy`` files written to ``data/attr_datavals/`` and
        ``data/attr_selected_vals/``.
    l1_lambda : float
        L1 regularisation coefficient on the LP weight vector ``w``.
    """

    def __init__(
        self,
        num_models_or_path: Union[int, str] = 1000,
        prob: Union[float, Literal["random"]] = 0.03,
        random_state: Optional[RandomState] = None,
        alt_train_kwargs=None,
        alpha: float = "best",
        retention_budget: Optional[int] = None,
        max_val="best",
        use_banzhaf: bool = False,
        add_banzhaf: bool = False,
        dataset_name="",
        l1_lambda: float = 0.0
    ):
        super().__init__()
        self.num_models_or_path = num_models_or_path
        self.num_models = num_models_or_path if isinstance(num_models_or_path, int) else \
            f"{num_models_or_path.split('-')[-1]}"

        self.prob = prob
        self.random_state = check_random_state(random_state)
        self.alt_train_kwargs = alt_train_kwargs
        self.alpha = alpha
        self.retention_budget = retention_budget

        self.max_val_ = None
        self.max_val = max_val
        self.use_banzhaf = use_banzhaf
        self.add_banzhaf = add_banzhaf
        self.dataset_name = dataset_name
        self.l1_lambda = l1_lambda

        if isinstance(retention_budget, int):
            assert retention_budget > 0, f"retention_budget must be a positive integer, got {retention_budget}"

        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None

        self.num_train_points = None
        self.num_test_points = None

        self.dv_base = None
        self.attr_matrix = None
        self._dv = None

        self.trained = False

        if isinstance(num_models_or_path, str):
            self.load(num_models_or_path)
            self.set_params_for_loaded_matrix()

        self.set_max_val()


    def set_max_val(self):
        """Set max_val to the maximum value in the attribute matrix."""
        if not isinstance(self.max_val, str):
            self.max_val_ = float(self.max_val)
            return

        if self.max_val == "best":
            self.max_val_ = "best"
            return

        if self.attr_matrix is not None:
            if self.max_val == "max":
                self.max_val_ = np.max(self.attr_matrix)
            elif self.max_val == "mean":
                self.max_val_ = np.mean(self.attr_matrix)
            elif self.max_val == "max+1std":
                self.max_val_ = np.max(self.attr_matrix) + np.std(self.attr_matrix)
            elif self.max_val == "max+2std":
                self.max_val_ = np.max(self.attr_matrix) + 2*np.std(self.attr_matrix)
            elif self.max_val == "max-1std":
                self.max_val_ = np.max(self.attr_matrix) - np.std(self.attr_matrix)
            elif self.max_val == "max-2std":
                self.max_val_ = np.max(self.attr_matrix) - 2*np.std(self.attr_matrix)
            elif self.max_val == "max+1.0":
                self.max_val_ = np.max(self.attr_matrix) + 1.0
            elif self.max_val == "max+s":
                if self.retention_budget is None:
                    raise ValueError("retention_budget must be set for this max_val mode")
                self.max_val_ = np.max(self.attr_matrix) + self.retention_budget * np.mean(self.attr_matrix)
            elif self.max_val == "max+2s":
                if self.retention_budget is None:
                    raise ValueError("retention_budget must be set for this max_val mode")
                self.max_val_ = np.max(self.attr_matrix) + self.retention_budget * 2 * np.mean(self.attr_matrix)
            elif self.max_val == "3s":
                if self.retention_budget is None:
                    raise ValueError("retention_budget must be set for this max_val mode")
                self.max_val_ = self.retention_budget * 3 * np.mean(self.attr_matrix)

            else:
                raise ValueError(f"Unknown max_val: {self.max_val_}. Use 'max', 'mean', 'max+1std' or None.")

            if isinstance(self.max_val, str):
                self.max_val += "_" + str(self.max_val_)

        else:
            raise ValueError("Attribute matrix is not set. Please train the model first.")


    def set_params_for_loaded_matrix(self):
        n_models_prob_epochs = self.num_models_or_path.split("-")[-1]
        n_mdls, prob, epochs = n_models_prob_epochs.split("_")[:3]

        self.num_models = int(n_mdls)
        self.prob = float(prob) if prob != "random" else prob

    def __repr__(self) -> str:
        if self.use_banzhaf:
            return f"Banzhaf({self.num_models})"

        frac = "" if self.retention_budget is None else f", frac={self.retention_budget}"
        add_banzhaf = ", +b" if self.add_banzhaf else ""
        l1_str = f", l1_lambda={self.l1_lambda}" if self.l1_lambda != 0.0 else ""

        if isinstance(self.num_models_or_path, str) and self.num_models_or_path.endswith(".npy"):
            return f"DataAttrOptim({self.num_models}, alpha={self.alpha} {frac}, max_val={self.max_val}{add_banzhaf}{l1_str})"

        return (f"DataAttrOptim({self.num_models},"
                f"prob={self.prob}, "
                f"alpha={self.alpha} {frac}, max_val={self.max_val}{add_banzhaf}{l1_str})")

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

        if not isinstance(self.num_models_or_path, str) and isinstance(self.retention_budget, int) and self.prob == "frac":
            assert self.retention_budget <= self.num_train_points, "retention_budget must be <= num_train_points"
            self.prob = self.retention_budget / self.num_train_points
            warnings.warn("prob is derived from retention_budget / num_train_points.")

        return self

    def load(self, path: str):
        """Load the data value evaluator from the given path."""
        try:
            self.attr_matrix = np.load(path)
            self.dv_base = np.mean(self.attr_matrix, axis=1)
            self.trained = True
            print(f"Loaded matrix {path}")

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at {path}. Training from scratch.")

        return self

    def get_accuracy_and_true_predictions(self, clf):
        """Get predictions from model"""
        pred = clf.predict(self.x_valid)
        pred_np = np.argmax(pred.cpu().numpy(), axis=1)
        pred_true = pred_np == np.argmax(self.y_valid.cpu().numpy(), axis=1)
        acc = self.evaluate(pred, self.y_valid)

        return acc, pred_true

    def fit_clf(self, clf, x, y, *args, **kwargs):
        """Fit classifier."""

        if self.alt_train_kwargs is not None:
            kwargs = self.alt_train_kwargs.copy()

            if self.alt_train_kwargs["epochs"] == "random":
                kwargs["epochs"] = np.random.randint(1, 15)

            clf.fit(x, y, **kwargs)

        else:
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

    @staticmethod
    def get_msr(sample_utility, sample_counts):
        """Get mean sample reward."""
        msr = np.divide(sample_utility,
                        sample_counts,
                        out=np.zeros_like(sample_utility),
                        where=sample_counts != 0)

        return msr


    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values."""
        if self.trained:
            print("Skipped training...")
            return self

        max_acc = 0
        num_points = len(self.x_train)
        subsets = self.get_subsets()  # shape: (num_models, num_train_points)

        sample_utility_per_train_test_idx = np.zeros((num_points, 2, len(self.x_valid)), dtype=np.ushort)
        sample_counts_per_train_test_idx = np.zeros((num_points, 2, len(self.x_valid)), dtype=np.ushort)

        for i in tqdm(range(self.num_models)):
            s = subsets[i]
            clf = self.pred_model.clone()
            clf = self.fit_clf(clf, self.x_train[s.nonzero()[0]], self.y_train[s.nonzero()[0]],
                               *args, **kwargs)

            acc, pred_true = self.get_accuracy_and_true_predictions(clf)  # pred_true has shape: (num_test_points,)
            max_acc = max(max_acc, acc)

            sample_utility_per_train_test_idx[range(num_points), s] += pred_true
            sample_counts_per_train_test_idx[range(num_points), s] += np.ones_like(pred_true)

        msr_tt = np.divide(
            sample_utility_per_train_test_idx,
            sample_counts_per_train_test_idx,
            where=sample_counts_per_train_test_idx != 0,
        )

        self.attr_matrix = msr_tt[:, 1] - msr_tt[:, 0]  # shape (num_train_points, num_test_points)
        self.dv_base = np.mean(self.attr_matrix, axis=1)

        if isinstance(self.num_models_or_path, str):
            np.save(self.num_models_or_path, self.attr_matrix)

        print(f"Max accuracy during training: {max_acc}")
        return self

    def optimize_max_val(self, val_range_max=None, val_range_alpha=None) -> float:
        """Grid-search over max_val and alpha; return the LP weights for the best-scoring combination."""
        if val_range_max is None:
            val_range_max = [0.2, 0.5, 0.8, 1.2, 1.6]
        if val_range_alpha is None:
            val_range_alpha = [0.0, 0.3, 0.6, 1.0]

        def average_score(subset_, n_models=5):

            scores = []
            for _ in range(n_models):
                clf = self.pred_model.clone()
                clf = self.fit_clf(clf, self.x_train[subset_], self.y_train[subset_])
                scr = self.evaluate(clf.predict(self.x_valid), self.y_valid)
                scores.append(scr)
                
            return np.mean(scores)

        Ws = []
        scores = []

        range_alpha = val_range_alpha if self.alpha == "best" else [self.alpha]

        best_vals = []
        for m_val in val_range_max:
            for a in range_alpha:
                W = max_sum_var(self.attr_matrix, size=self.retention_budget, alpha=a, max_val=m_val, l1_lambda=self.l1_lambda)
                subset = np.argpartition(W, -self.retention_budget)[-self.retention_budget:]

                score = average_score(subset)
                Ws.append(W)
                scores.append(score)
                best_vals.append({"max_val": m_val, "alpha": a, "score": score})

        best_idx = np.argmax(scores)
        print(f"Selected {best_idx} with:", best_vals[best_idx])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pth = f"data/attr_selected_vals/{self.dataset_name}-{self.retention_budget}-{self.prob}-{self.num_models}-{timestamp}.npy"
        np.save(pth, best_vals[best_idx])

        return Ws[best_idx]

    def evaluate_data_values(self) -> np.ndarray:
        """Return per-sample data values via LP optimisation (or raw mean-shift rewards for Banzhaf mode)."""
        if self.use_banzhaf:
            return self.dv_base

        if not self.retention_budget:
            raise ValueError("retention_budget must be set to use LP optimisation.")

        if self.max_val_ == "best" or self.alpha == "best":
            # If max_val is already resolved to a float, only search over alpha
            val_range_max = None if self.max_val_ == "best" else [self.max_val_]
            W = self.optimize_max_val(val_range_max=val_range_max)
        else:
            W = max_sum_var(self.attr_matrix, size=self.retention_budget, alpha=self.alpha, max_val=self.max_val_, l1_lambda=self.l1_lambda)

        dv = W + self.dv_base if self.add_banzhaf else W

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pth = f"data/attr_datavals/{self.dataset_name}-{self.retention_budget}-{self.prob}-{self.num_models}-{timestamp}.npy"
        np.save(pth, dv)

        return dv


