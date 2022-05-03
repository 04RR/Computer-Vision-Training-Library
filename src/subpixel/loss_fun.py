import torch
import torch.nn as nn
import torch.nn.functional as F

class PolyLoss(nn.Module):
    def __init__(self, eps=2.0):
        super().__init__()
        self.eps = eps

    def poly_loss(self, x, target, eps, ignore_index=-100):

        # https://github.com/frgfm/Holocron/blob/94cda24216e3f2dbc7c6521b466077ed1aa0d948/holocron/nn/functional.py#L552

        logpt = F.log_softmax(x, dim=1)
        logpt = logpt.transpose(1, 0).flatten(1).gather(0, target.view(1, -1)).squeeze()
        valid_idxs = torch.ones(target.view(-1).shape[0], dtype=torch.bool, device=x.device)

        if ignore_index >= 0 and ignore_index < x.shape[1]:
            valid_idxs[target.view(-1) == ignore_index] = False

        loss = -1 * logpt + eps * (1 - logpt.exp())

        loss = loss[valid_idxs].mean()

        return loss

    def forward(self, pred, label):

        return self.poly_loss(pred, label, self.eps)
