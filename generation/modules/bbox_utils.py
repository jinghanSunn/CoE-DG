"""
Bounding box utility functions for object detection.

This module provides common functions for processing bounding boxes,
including coordinate transformations and embeddings.
"""
import torch
import numpy as np
from typing import Union


def coordinate_embeddings(
    boxes: Union[np.ndarray, torch.Tensor], 
    dim: int, 
    w: float, 
    h: float
) -> torch.Tensor:
    """
    Generate coordinate embeddings for bounding boxes using sin/cos encoding.
    
    This function transforms bounding box coordinates into high-dimensional
    embeddings using sinusoidal position encoding.
    
    Args:
        boxes: Bounding boxes in format [x1, y1, x2, y2], shape [K, 4]
        dim: Embedding dimension for sin/cos encoding
        w: Image width
        h: Image height
        
    Returns:
        Coordinate embeddings of shape [K, 4, 2*dim]
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.Tensor(boxes)
    
    num_boxes = boxes.shape[0]

    # Transform to (x_c, y_c, w, h) format
    boxes_centered = boxes.new_zeros((num_boxes, 4))
    boxes_centered[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x_center
    boxes_centered[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y_center
    boxes_centered[:, 2] = boxes[:, 2] - boxes[:, 0]        # width
    boxes_centered[:, 3] = boxes[:, 3] - boxes[:, 1]        # height

    # Normalize positions to [0, 100] range
    pos = boxes.new_zeros((num_boxes, 4))
    pos[:, 0] = boxes_centered[:, 0] / w * 100
    pos[:, 1] = boxes_centered[:, 1] / h * 100
    pos[:, 2] = boxes_centered[:, 2] / w * 100
    pos[:, 3] = boxes_centered[:, 3] / h * 100

    # Generate sin/cos embeddings
    dim_mat = 1000 ** (torch.arange(dim, dtype=boxes.dtype, device=boxes.device) / dim)
    sin_embedding = (pos.view((num_boxes, 4, 1)) / dim_mat.view((1, 1, -1))).sin()
    cos_embedding = (pos.view((num_boxes, 4, 1)) / dim_mat.view((1, 1, -1))).cos()

    return torch.cat((sin_embedding, cos_embedding), dim=-1)


def coordinate_convert(
    boxes: Union[np.ndarray, torch.Tensor], 
    w: float, 
    h: float
) -> torch.Tensor:
    """
    Convert bounding box coordinates to normalized integer positions.
    
    This function normalizes bounding box coordinates to [0, 100] range
    and converts them to integers.
    
    Args:
        boxes: Bounding boxes in format [x1, y1, x2, y2], shape [K, 4]
        w: Image width
        h: Image height
        
    Returns:
        Normalized positions of shape [K, 4]
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.Tensor(boxes)
    
    num_boxes = boxes.shape[0]

    # Normalize positions to [0, 100] range and convert to int
    pos = boxes.new_zeros((num_boxes, 4))
    pos[:, 0] = (boxes[:, 0] / w * 100).int()
    pos[:, 1] = (boxes[:, 1] / h * 100).int()
    pos[:, 2] = (boxes[:, 2] / w * 100).int()
    pos[:, 3] = (boxes[:, 3] / h * 100).int()

    return pos.cpu()


def filter_boxes_by_score(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    threshold: float = 0.7
) -> tuple:
    """
    Filter bounding boxes by confidence score.
    
    Args:
        boxes: Bounding boxes, shape [N, 4]
        scores: Confidence scores, shape [N]
        threshold: Minimum score threshold
        
    Returns:
        Tuple of (filtered_boxes, filtered_scores, indices)
    """
    idxs = torch.where(scores > threshold)[0]
    
    if len(idxs) == 0:
        return None, None, None
    
    filtered_boxes = boxes[idxs]
    filtered_scores = scores[idxs]
    
    return filtered_boxes, filtered_scores, idxs


def create_default_box(w: float, h: float) -> np.ndarray:
    """
    Create a default bounding box covering the entire image.
    
    Args:
        w: Image width
        h: Image height
        
    Returns:
        Default box [0, 0, w, h] as numpy array
    """
    return np.array([[0, 0, w, h]])


def pad_boxes(boxes: torch.Tensor, max_boxes: int, box_dim: int = 5) -> torch.Tensor:
    """
    Pad boxes tensor to fixed size.
    
    Args:
        boxes: Input boxes tensor
        max_boxes: Maximum number of boxes
        box_dim: Dimension of each box (default: 5 for [x1, y1, x2, y2, score])
        
    Returns:
        Padded boxes tensor of shape [max_boxes, box_dim]
    """
    num_boxes = boxes.shape[0]
    padded = torch.ones((max_boxes, box_dim)) * -1
    
    if num_boxes > 0:
        padded[:min(num_boxes, max_boxes)] = boxes[:max_boxes]
    
    return padded

