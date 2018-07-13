import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import reporter

from .transform import BatchTransform


def calc_prec_and_recall(scores, labels):
    scores = F.sigmoid(scores).data
    xp = cuda.get_array_module(scores, labels)

    tp = np.float32(0)
    fp = np.float32(0)
    n_pos = np.float32(0)
    n_pred = np.float32(0)
    for score, label in zip(scores, labels):
        n_pos += len(label)
        label = xp.array(label)
        pred_label = xp.where(score > 0.5)[0]

        seen = xp.zeros((len(label),), dtype=np.bool)
        n_pred += len(pred_label)
        for pred_lb in pred_label:
            index = xp.where(pred_lb == label)[0]
            if len(index) > 0:
                if seen[index] == 0:
                    tp += 1
                    seen[index] = 1
                else:
                    fp += 1
            else:
                fp += 1
    return {
        'precision': tp / (tp + fp), 'recall': tp / n_pos,
        'n_pred': n_pred, 'n_pos': n_pos}


class MultiLabelClassifier(chainer.Chain):

    def __init__(self, model, loss_scale):
        super(MultiLabelClassifier, self).__init__()
        with self.init_scope():
            self.model = model
        self.loss_scale = loss_scale

    def __call__(self, x, labels):
        x = BatchTransform(self.model.mean)(x)
        x = self.xp.array(x)
        scores = self.model(x)

        B, n_class = scores.shape[:2]
        one_hot_labels = self.xp.zeros((B, n_class), dtype=np.int32)
        for i, label in enumerate(labels):
            one_hot_labels[i, label] = 1
        # sigmoid_cross_entropy normalizes the loss
        # by the size of batch and the number of classes.
        # It works better to remove the normalization factor
        # of the number of classes.
        loss = self.loss_scale * F.sigmoid_cross_entropy(
            scores, one_hot_labels)

        result = calc_prec_and_recall(scores, labels)
        reporter.report({'loss': loss}, self)
        reporter.report({'recall': result['recall']}, self)
        reporter.report({'precision': result['precision']}, self)
        reporter.report({'n_pred': result['n_pred']}, self)
        reporter.report({'n_pos': result['n_pos']}, self)
        return loss
