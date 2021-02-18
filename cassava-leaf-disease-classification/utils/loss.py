import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def focal_loss(logits, y_batch, gamma=2.):
    true = y_batch.view(len(y_batch), -1)
    preds = torch.log_softmax(logits, dim=1)
    losses = -torch.gather(preds, dim=1, index=true)
    softmax = torch.softmax(logits, dim=1)
    coefs = (1 - torch.gather(softmax, dim=1, index=true))**gamma
    return torch.mean(coefs * losses)


class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, xent=.1):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.xent = xent

        self.y = torch.Tensor([1]).cuda()

    def forward(self, input, target, reduction="mean"):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)), self.y, reduction=reduction)

        cent_loss = F.cross_entropy(F.normalize(input), target, reduce=False)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * cent_loss

        if reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss