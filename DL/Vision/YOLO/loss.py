# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:24:07 2023

@author: sanat
"""

import torch
import torch.nn as nn
from iou import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        
    def forward(self, prediction, target):
        prediction = prediction.reshape(-1, self.S, self.S, self.C + self.B * 5)
        
        # Why 21-25 and 25-30
        # and why 21:25 for target but 21:25 and 25:30 for prediction???
        # the target will only contain one box i.e., in the grid cell
        iou_b1 = intersection_over_union(prediction[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(prediction[..., 25:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_max, best_box = torch.max(ious, dim=0)
        exist_box = target[..., 20].unsqueeze(3) # Iobj_i
        
        # ============================= #
        #     FOR  BOX  COORDINATES     #
        # ============================= #
        
        box_predictions = exist_box * (
            # best box is here 0 or 1
            # so if it's 1 it means the second box and 0 means the first box
            best_box * prediction[..., 26:30]
            + (1-best_box) * prediction[..., 21:25]
            )
        
        box_target = exist_box * target[..., 21:25]
        
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
            )
        
        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])
        
        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_target, end_dim=-2)
            )
        
        # ============================= #
        #    FOR  OBJECT  COORDINATES   #
        # ============================= #
        # for object loss
        pred_box = (
            best_box * prediction[..., 25:26] + (1 - best_box) * prediction[..., 20:21]
            )
        
        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exist_box * pred_box),
            torch.flatten(exist_box * target[..., 20:21]))
        
        # for no object
        # (N, S, S, 1) -> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1- exist_box) * prediction[..., 20:21], start_dim=1),
            torch.flatten((1 - exist_box) * target[..., 20:21], start_dim=1)
            )
        
        no_object_loss += self.mse(
            torch.flatten((1- exist_box) * prediction[..., 25:26], start_dim=1),
            torch.flatten((1 - exist_box) * target[..., 20:21], start_dim=1)
            )
        
        # for class loss
        # (N, S, S, 20) -> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exist_box * prediction[..., :20], end_dim=-2),
            torch.flatten(exist_box * target[..., :20], end_dim=-2)
            )
        
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
            )
        
        return loss
        
        
        
        
        
        
        
        
        
        
        
        
        