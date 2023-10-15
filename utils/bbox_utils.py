import tensorflow as tf
from typing import Tuple, Sequence

def convert_to_tf_format(boxes: tf.Tensor) -> tf.Tensor:
    ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=-1)
    return tf.concat([xmin, ymin, xmax, ymax], axis=-1)

def scale_boxes(boxes: tf.Tensor, from_size: Tuple[int, int], to_size: Tuple[int, int]) -> tf.Tensor:
    # Unpack the height and width of the source image (from_size) and the target image (to_size)
    from_height, from_width = from_size
    to_height, to_width = to_size

    # Calculate the width and height ratios to scale the boxes
    ratio_width = from_width / to_width
    ratio_height = from_height / to_height

    # Scale the box coordinates based on the width and height ratios
    x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
    x_min *= ratio_width
    x_max *= ratio_width
    y_min *= ratio_height
    y_max *= ratio_height

    # Concatenate the scaled coordinates to form the resulting boxes  
    return tf.concat([x_min, y_min, x_max, y_max], axis=1)

def normalize_boxes(boxes: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    image_height, image_width = image_size
    
    # Normalize box coordinates to be in the range [0, 1]
    x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
    x_min /= (image_width - 1)
    x_max /= (image_width - 1)
    y_min /= (image_height - 1)
    y_max /= (image_height - 1)

    return tf.concat([x_min, y_min, x_max, y_max], axis=1)

def regress_bboxes(boxes: tf.Tensor, regressors: tf.Tensor) -> tf.Tensor:
    # Cast input boxes and regressors to float32 for calculations
    boxes = tf.cast(boxes, tf.float32)
    regressors = tf.cast(regressors, tf.float32)
    
    # Calculate the center (Px, Py) and size (Pw, Ph) of input boxes
    Px = (boxes[..., 0] + boxes[..., 2]) / 2.0
    Py = (boxes[..., 1] + boxes[..., 3]) / 2.0
    Pw = boxes[..., 2] - boxes[..., 0]
    Ph = boxes[..., 3] - boxes[..., 1]

    # Split the regressors into individual components: dxP, dyP, dwP, dhP
    dxP = regressors[..., 0]
    dyP = regressors[..., 1]
    dwP = regressors[..., 2]
    dhP = regressors[..., 3]

    # Calculate the new center (Gx_hat, Gy_hat) and size (Gw_hat, Gh_hat) of boxes
    Gx_hat = Pw * dxP + Px
    Gy_hat = Ph * dyP + Py
    Gw_hat = Pw * tf.math.exp(dwP)
    Gh_hat = Ph * tf.math.exp(dhP)

    # Calculate the new coordinates of the boxes
    x1 = Gx_hat - (Gw_hat / 2.0)
    y1 = Gy_hat - (Gh_hat / 2.0)
    x2 = x1 + Gw_hat
    y2 = y1 + Gh_hat

    return tf.stack([x1, y1, x2, y2], axis=-1)

def clip_boxes(boxes: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    height, width = image_size

    # Cast height and width to float32
    height = tf.cast(height - 1, tf.float32)
    width = tf.cast(width - 1, tf.float32)

    # Clip box coordinates to be within the image boundaries
    x_min = tf.clip_by_value(boxes[..., 0], 0.0, width)
    y_min = tf.clip_by_value(boxes[..., 1], 0.0, height)
    x_max = tf.clip_by_value(boxes[..., 2], 0.0, width)
    y_max = tf.clip_by_value(boxes[..., 3], 0.0, height)

    return tf.stack([x_min, y_min, x_max, y_max], axis=-1)

def single_image_nms(boxes: tf.Tensor, scores: tf.Tensor, score_threshold: float = 0.05) -> tf.Tensor:
    def per_class_nms(class_idx: int) -> tf.Tensor:
        # Extract scores for a specific class
        class_scores = tf.gather(scores, class_idx, axis=-1)
        
        # Apply Non-Maximum Suppression (NMS) to the boxes of that class
        indices = tf.image.non_max_suppression(boxes=boxes, scores=class_scores, max_output_size=100, score_threshold=score_threshold)
        
        # Count the number of selected indices and crete labels for them
        n_indices = tf.constant(tf.shape(indices)[0])
        labels = tf.tile([class_idx], [n_indices])
        
        return tf.stack([indices, labels], axis=-1)

    # Ensure that the input boxes are in the required format (xmin, ymin, xmax, ymax)
    boxes = convert_to_tf_format(boxes)
    
    # Apply NMS for each class and concatenate the results
    return tf.concat([per_class_nms(c) for c in range(tf.shape(scores)[-1])], axis=0)


def nms(boxes: tf.Tensor, class_scores: tf.Tensor, score_threshold: float = 0.5) -> Tuple[Sequence[tf.Tensor], Sequence[tf.Tensor], Sequence[tf.Tensor]]:
    # Determine the batch size and the number of classes
    batch_size = tf.shape(boxes)[0]
    num_classes = tf.shape(class_scores)[-1]
    
    # Cast input boxes and class scores to float32 for calculations
    boxes = tf.cast(boxes, tf.float32)
    class_scores = tf.cast(class_scores, tf.float32)
    
    # Initialize lists to store the results for each batch
    all_boxes = []
    all_labels = []
    all_scores = []

    for batch_idx in range(batch_size):
        # Extract class scores and boxes for the current batch
        c_scores = tf.gather(class_scores, batch_idx)
        c_boxes = tf.gather(boxes, batch_idx)

        # Apply single-image NMS to select the most relevant boxes
        indices = single_image_nms(c_boxes, c_scores, score_threshold)
        
        # Extract selected labels and scores
        batch_labels = indices[:, 1]
        batch_scores = tf.gather_nd(c_scores, indices)
        batch_boxes = tf.gather(c_boxes, indices[:, 0])
        
        # Append the results for the current batch to the respective lists
        all_boxes.append(batch_boxes)
        all_scores.append(batch_scores)
        all_labels.append(batch_labels)

    # Return the results for all batches as tuples of lists
    return all_boxes, all_labels, all_scores
    
def bbox_overlap(boxes: tf.Tensor, gt_boxes: tf.Tensor) -> tf.Tensor:
    bb_x_min, bb_y_min, bb_x_max, bb_y_max = tf.split(value=boxes, num_or_size_splits=4, axis=2)
    gt_x_min, gt_y_min, gt_x_max, gt_y_max = tf.split(value=gt_boxes, num_or_size_splits=4, axis=2)

    # Calculates the intersection area
    i_xmin = tf.math.maximum(bb_x_min, tf.transpose(gt_x_min, [0, 2, 1]))
    i_xmax = tf.math.minimum(bb_x_max, tf.transpose(gt_x_max, [0, 2, 1]))
    i_ymin = tf.math.maximum(bb_y_min, tf.transpose(gt_y_min, [0, 2, 1]))
    i_ymax = tf.math.minimum(bb_y_max, tf.transpose(gt_y_max, [0, 2, 1]))
    
    i_area = (tf.math.maximum(i_xmax - i_xmin, 0) * tf.math.maximum(i_ymax - i_ymin, 0))

    # Calculates the union area
    bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
    gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)
    
    # Adds a small epsilon to avoid divide-by-zero
    u_area = bb_area + tf.transpose(gt_area, [0, 2, 1]) - i_area + 1e-8

    # Calculates IoU
    iou = i_area / u_area

    # Fills -1 for IoU entries between the padded ground truth boxes
    gt_invalid_mask = tf.less(tf.reduce_max(gt_boxes, axis=-1, keepdims=True), 0.0)
    padding_mask = tf.logical_or(tf.zeros_like(bb_x_min, dtype=tf.bool), tf.transpose(gt_invalid_mask, [0, 2, 1]))
    iou = tf.where(padding_mask, -tf.ones_like(iou), iou)

    return iou