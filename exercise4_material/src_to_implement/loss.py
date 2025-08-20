import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalBCELoss(nn.Module):
    def __init__(self, w_parent=1.0, w_child=1.0, inconsistency_weight=1.0, threshold=0.5):
        """
        w_parent: 类1（父类）的 BCE 权重
        w_child:  类2（子类）的 BCE 权重
        inconsistency_weight: 不一致惩罚系数
        threshold: 判定子类依赖的一致性阈值（默认 0.5）
        """
        super().__init__()
        self.w_parent = w_parent
        self.w_child = w_child
        self.lambda_inc = inconsistency_weight
        self.th = threshold
        self.bce = nn.BCELoss(reduction='mean')

    def forward(self, pred, target):
        """
        pred: (bs, 2)，已 sigmoid 的概率
        target: (bs, 2)，0/1 标签
        """
        # 数值稳定
        pred = pred.clamp(min=1e-6, max=1-1e-6)

        p1, p2 = pred[:, 0], pred[:, 1]
        t1, t2 = target[:, 0], target[:, 1]

        # 基本 BCE 项
        loss_parent = self.bce(p1, t1)
        loss_child  = self.bce(p2, t2)

        # 依赖一致性惩罚：p1<th 且 p2>th 时罚；用可导的软形式
        inconsistency = F.relu(self.th - p1) * F.relu(p2 - self.th)
        loss_inconsistency = inconsistency.mean()

        # 总损失（类1加权 + 子类权重 + 不一致惩罚）
        loss = self.w_parent * loss_parent + self.w_child * loss_child + self.lambda_inc * loss_inconsistency
        return loss