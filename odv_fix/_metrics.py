"""COPIED FROM OPENDATVAL FOR DOCKERFILE"""
import torch
import torch.nn.functional as F

from opendataval.util import FuncEnum


def accuracy(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute accuracy of two one-hot encoding tensors."""

    # BugFix: If tensors are on different devices, move them to CPU
    if a.get_device() != b.get_device():
        a = a.cpu()
        b = b.cpu()

    return (a.argmax(dim=1) == b.argmax(dim=1)).float().mean().item()


def neg_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return -torch.square(a - b).sum().sqrt().item()


def neg_mse(a: torch.Tensor, b: torch.Tensor):
    return -F.mse_loss(a, b).item()


def worst_class_accuracy(a: torch.Tensor, b: torch.Tensor) -> float:
    """Returns accuracy on worst class."""

    # BugFix: If tensors are on different devices, move them to CPU
    if a.get_device() != b.get_device():
        a = a.cpu()
        b = b.cpu()

    a = a.argmax(dim=1)
    b = b.argmax(dim=1)

    classes = torch.unique(a)

    accs = []

    for cls in classes:
        mask = a == cls
        accs.append((b[mask] == a[mask]).float().mean().item())

    return min(accs)


class Metrics(FuncEnum):
    ACCURACY = FuncEnum.wrap(accuracy)
    NEG_L2 = FuncEnum.wrap(neg_l2)
    NEG_MSE = FuncEnum.wrap(neg_mse)
    WORST_CLASS_ACCURACY = FuncEnum.wrap(worst_class_accuracy)