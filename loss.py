import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module): # wyzanczenie straty

    def __init__(self, S=7, B=2, C=50): # Zmieniono na C=50
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        self.S = S # podział obrazu
        self.B = B # ilość boxów na ramkę podziałów
        self.C = C # ilość klas

        self.lambda_noobj = 0.5 # kara za brak obiektu
        self.lambda_coord = 5 # kara za błędną pozycję boxa

    def forward(self, predictions, target):
        # (BATCH_SIZE, S*S(C+B*5)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        iou_b1 = intersection_over_union(predictions[..., self.C+1:self.C+5], target[..., self.C+1:self.C+5])
        iou_b2 = intersection_over_union(predictions[..., self.C+6:self.C+10], target[..., self.C+1:self.C+5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        iou_maxes, bestbox = torch.max(ious, dim=0) # wybór boxa z największym IoU
        exists_box = target[..., self.C].unsqueeze(3) 

        box_predictions = exists_box * (
            (
                bestbox * predictions[..., self.C+6:self.C+10]
                + (1 - bestbox) * predictions[..., self.C+1:self.C+5]
            )
        )

        box_targets = exists_box * target[..., self.C+1:self.C+5]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        pred_box = (
            bestbox * predictions[..., self.C+5:self.C+6] + (1 - bestbox) * predictions[..., self.C:self.C+1]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., self.C:self.C+1]),
        )

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C:self.C+1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C+5:self.C+6], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1)
        )

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2),
        )

        loss = (
            self.lambda_coord * box_loss 
            + object_loss  
            + self.lambda_noobj * no_object_loss
            + class_loss 
        )

        return loss
