# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:22:27 2023

@author: sanat
"""

import torch

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    
    NOTE: remember the top of image is (0, 0)
    
    Calculates intersection over union between the predicted and the actual labels
    
    Parameters:
    ------------
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
            For bounding box either we can give coordinates of the edges points of the box or
            we can provide middle coordinates of the box and it's respective height and width
        
    Returns:
    -------
        tensor: Intersection over union for all examples
    """

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    
    # boxes_preds shape is (N, 4) where N is the number of bboxes
    # why doing 0:1 instead of just 0 cause we want the shape to be reserved
    # which mean we want it be be in N, 1 rather than in N
    
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    
    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
        

    ## The coordinates of edge points of the intersection of the boxes, check NOTE in starting        
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    # clamp for edge case when there is no intersection between the two
    # so clamp will give us 0 if the subtraction is -ve i.e., the minimum value
    # we can get is 0
    intersection_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))
    
    
    ## 1e-6 is for avoiding dividing by 0.
    return intersection_area / (box1_area + box2_area - intersection_area + 1e-6)
    
    
# a = torch.tensor([[1, 2, 3, 4], [6, 7, 8, 9]])
# b = torch.tensor([[1, 4, 5, 6], [3, 5, 7, 8]])
# print(intersection_over_union(a, b))
    
    
    
    
    
    
    
    
    
    
    
    