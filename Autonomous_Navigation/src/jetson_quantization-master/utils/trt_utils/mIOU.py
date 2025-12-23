import numpy as np

class MIoUCalculator:
    def __init__(self, num_classes=100):
        self.num_classes = num_classes

    def compute_per_frame(self, gt, pred):
        gt = np.asarray(gt).reshape(-1)
        pred = np.asarray(pred).reshape(-1)
        ious = []
        for cls in range(self.num_classes):
            gt_inds = (gt == cls)
            pred_inds = (pred == cls)
            intersection = np.logical_and(pred_inds, gt_inds).sum()
            union = np.logical_or(pred_inds, gt_inds).sum()
            if union == 0:
                continue
            ious.append(intersection / union)
        if len(ious) == 0:
            return np.nan
        return np.mean(ious)

    def compute_list(self, gts, preds):
        mious = []
        for pred, gt in zip(preds, gts):
            mIoU = self.compute_per_frame(pred, gt)
            mious.append(mIoU)
        return mious

# 사용 예시
# miou_calc = MIoUCalculator(num_classes=100)
# mious = miou_calc.compute_list(fp32_output, query_output)
