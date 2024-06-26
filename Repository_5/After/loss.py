import torch
import torch.nn.functional as F


class CELoss:
    def __init__(self, trg_pad_idx):
        self.trg_pad_idx = trg_pad_idx

    def __call__(self, model, src_seq, trg_seq, gold, smoothing=False):
        """Apply label smoothing if needed"""
        pred = model(src_seq, trg_seq)
        loss = self._loss_function(pred, gold, smoothing=smoothing)

        pred = pred.max(1)[1]
        gold = gold.contiguous().view(-1)
        non_pad_mask = gold.ne(self.trg_pad_idx)
        n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
        n_word = non_pad_mask.sum().item()

        return loss, n_correct, n_word

    def _loss_function(self, pred, gold, smoothing=False):
        """Calculate cross entropy loss, apply label smoothing if needed."""

        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.1
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            non_pad_mask = gold.ne(self.trg_pad_idx)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later
        else:
            loss = F.cross_entropy(
                pred, gold, ignore_index=self.trg_pad_idx, reduction="sum"
            )
        return loss
