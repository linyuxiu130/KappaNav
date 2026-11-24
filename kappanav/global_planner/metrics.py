"""Per-source scores. A pixel whose label is one class counts in the confusion matrix; a set-labelled pixel counts in
set accuracy (argmax inside the set); drivability is scored where the label set is all DRIVABLE or all BLOCKED;
value_error is the class-balanced error of the 0-255 value on single-class pixels."""
import numpy as np

from kappanav.global_planner.taxonomy import BLOCKED, CLASSES, DRIVABLE, VALUES, bits

K = len(CLASSES)


def singles(labels):
    """Class index of single-class bitmasks, -1 elsewhere."""
    out = np.full(labels.shape, -1, np.int64)
    for k in range(K):
        out[labels == (1 << k)] = k
    return out


class Scores:
    def __init__(self):
        self.matrix = np.zeros((K, K), np.int64)
        self.value_sum, self.value_error = np.zeros(K), np.zeros(K)
        self.set_hits = self.set_pixels = 0
        self.passable_bits, self.blocked_bits = bits(DRIVABLE), bits(BLOCKED)
        self.p, self.y = [], []
        self.rng = np.random.default_rng(0)

    def add(self, posterior, passable, value, labels):
        prediction = posterior.argmax(0)
        single = singles(labels)
        valid = single >= 0
        self.matrix += np.bincount(single[valid] * K + prediction[valid], minlength=K * K).reshape(K, K)
        self.value_sum += np.bincount(single[valid], value[valid], K)
        self.value_error += np.bincount(single[valid], np.abs(value[valid] - VALUES[single[valid]]), K)
        sets = (labels != 0) & ~valid
        self.set_hits += int(((labels[sets] >> prediction[sets]) & 1).sum())
        self.set_pixels += int(sets.sum())
        yes = (labels != 0) & ((labels & ~self.passable_bits) == 0)
        no = (labels != 0) & ((labels & ~self.blocked_bits) == 0)
        index = np.flatnonzero(yes | no)
        if len(index):
            index = self.rng.choice(index, min(len(index), 20000), replace=False)
            self.p.append(passable.ravel()[index])
            self.y.append(yes.ravel()[index])

    def merge(self, other):
        """Fold another rank's scores into this one."""
        self.matrix += other.matrix
        self.value_sum += other.value_sum
        self.value_error += other.value_error
        self.set_hits += other.set_hits
        self.set_pixels += other.set_pixels
        self.p += other.p
        self.y += other.y
        return self

    def report(self):
        tp = self.matrix.diagonal()
        union = self.matrix.sum(0) + self.matrix.sum(1) - tp
        pixels = self.matrix.sum(1)
        present = pixels > 0
        iou = np.divide(tp, union, out=np.full(K, np.nan), where=union > 0)
        result = {'mIoU': float(np.nanmean(iou[present])) if present.any() else None,
                  'class_IoU': {c: (None if not present[k] else float(iou[k])) for k, c in enumerate(CLASSES)},
                  'class_value': {c: (float(self.value_sum[k] / pixels[k]) if present[k] else None)
                                  for k, c in enumerate(CLASSES)},
                  'value_error': float((self.value_error[present] / pixels[present]).mean()) if present.any() else None,
                  'single_pixels': int(self.matrix.sum()),
                  'set_accuracy': self.set_hits / self.set_pixels if self.set_pixels else None,
                  'passable_AUC': None, 'false_passable': None, 'missed_passable': None, 'calibration_error': None}
        if self.p:
            p, y = np.concatenate(self.p), np.concatenate(self.y)
            result.update(passable_AUC=auc(p, y), calibration_error=calibration_error(p, y),
                          false_passable=float((p[~y] >= .5).mean()) if (~y).any() else None,
                          missed_passable=float((p[y] < .5).mean()) if y.any() else None)
        return result


def calibration_error(p, y, bins=10):
    """Expected calibration error of P(passable): |mean p - passable share| averaged over equal-width bins of p."""
    index = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[index == b].mean() - y[index == b].mean()) * (index == b).mean()
                     for b in range(bins) if (index == b).any()))


def auc(p, y):
    if y.all() or not y.any():
        return None
    order = np.argsort(p, kind='stable')
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    positives = y.sum()
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * (len(y) - positives)))
