import torch
import torch.nn as nn
import tensorly as tl


tl.set_backend("pytorch")


# different criterions for sigma selection
# obtained from https://github.com/yuhuixu1993/Trained-Rank-Pruning
class EnergyThreshold(object):
    def __init__(self, threshold, eidenval=True):
        """
        :param threshold: float, threshold to filter small valued sigma:
        :param eidenval: bool, if True, use eidenval as criterion, otherwise use singular
        """
        self.T = threshold
        assert self.T < 1.0 and self.T > 0.0
        self.eiden = eidenval

    def __call__(self, sigmas):
        """
        select proper numbers of singular values
        :param sigmas: numpy array obj which containing singular values
        :return: valid_idx: int, the number of sigmas left after filtering
        """
        if self.eiden:
            energy = sigmas**2
        else:
            energy = sigmas

        sum_e = torch.sum(energy)
        valid_idx = sigmas.size(0)
        for i in range(energy.size(0)):
            if energy[: (i + 1)].sum() / sum_e >= self.T:
                valid_idx = i + 1
                break

        return valid_idx


def svd_rank(weight, criterion):
    _, S, _ = torch.svd(weight, some=True)

    return criterion(S)


def svd_rank_linear(layer, criterion=EnergyThreshold(0.85)):
    return svd_rank(layer.weight.data, criterion)


def decompose(model: nn.Module, rank: int = 0):
    """
    Decompose a model to a CPDBlock model.

    Args:
        model (nn.Module): a Conv2d form model.

    Returns:
        model (nn.Module): a CPDBlock form model

    """

    for name, module in model._modules.items():
        if isinstance(module, nn.Linear):
            if module.out_features > 1:
                print(f"decomposing: {name}")
                if rank == 0:
                    rank = svd_rank_linear(module)
                    print(f"SVD Estimated Rank (using 90% rule): {rank}")
                model._modules[name] = decompose_linear(module, rank)

        elif len(list(module.children())) > 0:
            print(f"recursing module: {name}")
            # recurse
            model._modules[name] = decompose(module, rank)

    return model


def decompose_linear(layer, rank=1):
    print(layer)
    [U, S, V] = tl.svd_interface(
        layer.weight.data, method="randomized_svd", n_eigenvecs=rank
    )

    first_layer = nn.Linear(in_features=V.shape[1], out_features=V.shape[0], bias=False)

    bias = layer.bias is not None
    second_layer = nn.Linear(in_features=U.shape[1], out_features=U.shape[0], bias=bias)
    if bias:
        second_layer.bias.data = layer.bias.data

    first_layer.weight.data = (V.t() * S).t()
    second_layer.weight.data = U

    new_layers = [first_layer, second_layer]

    return nn.Sequential(*new_layers)
