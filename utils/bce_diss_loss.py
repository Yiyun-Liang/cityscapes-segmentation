import torch

class BCEDiceLoss:

    """
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: loss.
    """

    def __init__(self, bce_weight=1., weight=None, eps=1e-7, 
                 smooth=.0, class_weights=[], threshold=0., activate=False):

        self.bce_weight = bce_weight
        self.eps = eps
        self.smooth = smooth
        self.threshold = threshold # 0 or apply sigmoid and threshold > .5 instead
        self.activate = activate
        self.class_weights = class_weights
        self.nll = torch.nn.BCEWithLogitsLoss(weight=weight)

    def __call__(self, logits, true):
        loss = self.bce_weight * self.nll(logits, true)
        if self.bce_weight < 1.:
            dice_loss = 0.
            batch_size, num_classes = logits.shape[:2]
            if self.activate:
                logits = torch.sigmoid(logits)
            logits = (logits > self.threshold).float()
            for c in range(num_classes):
                iflat = logits[:, c,...].view(batch_size, -1)
                tflat = true[:, c,...].view(batch_size, -1)
                intersection = (iflat * tflat).sum()
                
                w = self.class_weights[c]
                dice_loss += w * ((2. * intersection + self.smooth) /
                                 (iflat.sum() + tflat.sum() + self.smooth + self.eps))
            loss -= (1 - self.bce_weight) * torch.log(dice_loss)

        return loss