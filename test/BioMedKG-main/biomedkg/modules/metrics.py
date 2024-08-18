import torch
from torchmetrics import Metric

class EdgeWisePrecision(Metric):
    def __init__(self, class_mapping: dict, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.class_mapping = class_mapping
        self.num_classes = len(class_mapping)
        self.add_state("class_counts", default=torch.zeros(self.num_classes), dist_reduce_fx="sum")
        self.add_state("above_threshold_counts", default=torch.zeros(self.num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        for class_idx in range(self.num_classes):
            class_mask = (target == class_idx)
            self.class_counts[class_idx] += class_mask.sum()
            self.above_threshold_counts[class_idx] += (preds[class_mask] > self.threshold).sum()

    def compute(self) -> dict:
        percentages = {}
        for class_idx in range(self.num_classes):
            key_name = self.class_mapping[class_idx] + "_pre"
            if self.class_counts[class_idx] > 0:
                percentages[key_name] = (self.above_threshold_counts[class_idx] / self.class_counts[class_idx]).item()
            else:
                percentages[key_name] = 0.0
        return percentages