# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:10:35 2023

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
        prediction = prediction.reshape(-1,
                                        self.S, self.S, self.C + self.B * 5)

        iou_b1 = intersection_over_union(
            prediction[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(
            prediction[..., 21:25], target[..., 21:25])

        # best_box means which box has the highest iou with the target bounding box
        # it's value will be 0 or 1
        best_box = torch.max(iou_b1, iou_b2)

        # unsqueeze to reserve the dimension
        # for eg-> from (1, 2, 3) to (1, 2, 3, 1)
        exist_box = target[..., 20].unsqueeze(3)

        # ============================= #
        #     FOR  BOX  COORDINATES     #
        # ============================= #

        box_predictions = exist_box * (
            best_box * prediction[..., 26:30]
            + (1 - best_box) * prediction[..., 21:25]
        )

        # Now doing sqrt
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4]) + 1e-6)
        
        box_target = exist_box * target[..., 21:25]
        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions),
            torch.flatten(box_target)
        )

        # ============================= #
        #    FOR  OBJECT  COORDINATES   #
        # ============================= #

        l1 = (
            best_box * prediction[..., 25:26]
            + (1 - best_box) * prediction[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(exist_box * l1),
            torch.flatten(exist_box * target[..., 20:21])
        )

        # Now when there is no object
        # it should be the duty of both the bounding boxes to predict that
        # there exist no object
        no_object_loss = self.mse(
            torch.flatten((1 - exist_box) * prediction[..., 20:21]),
            torch.flatten((1 - exist_box) * target[..., 20:21])
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exist_box) * prediction[..., 25:26]),
            torch.flatten((1 - exist_box) * target[..., 20:21])
        )
        
        
        class_loss = self.mse(
            torch.flatten(exist_box * prediction[..., :20]),
            torch.flatten(exist_box * target[..., :20])
            )
        
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
            )
        
        return loss
        
        
        
        
        
        
        
        
        