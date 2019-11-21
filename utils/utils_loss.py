import torch.nn.functional as F


def label_cross_entropy(preds, labels):
    return F.binary_cross_entropy_with_logits(preds, labels)


def label_accuracy(preds, target):
    _, preds = preds.max(-1)
    _, target = target.max(-1)
    correct = preds.int().data.eq(target.int()).sum()
    return correct.float() / (target.size(0))
