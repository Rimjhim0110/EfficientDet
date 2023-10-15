import numpy as np
import tensorflow as tf
from typing import Any, Tuple, Sequence
from . import bbox_utils

class GenerateAnchor(object):
    def __init__(self, base_size: float, aspect_ratios: Sequence[float], stride: int = 1) -> None:
        self.base_size = base_size
        self.stride = stride
        self.aspect_ratios = aspect_ratios

        # Predefined scales for anchors
        self.anchor_scales = [2**0, 2**(1/3.0), 2**(2/3.0)]

        # Generate the anchor boxes
        self.anchors = self.generate_anchors()
    
    def __call__(self, *args: Any, **kwargs: Any) -> tf.Tensor:
        return self.tile_anchors_over_feature_map(*args, **kwargs)

    def tile_anchors_over_feature_map(self, feature_map_shape: Tuple[int, int]) -> tf.Tensor:
        def arange(limit: int) -> tf.Tensor:
            return tf.range(0., tf.cast(limit, tf.float32), dtype=tf.float32)

        stride = tf.cast(self.stride, tf.float32)
        shift_x = (arange(feature_map_shape[1]) + 0.5) * stride
        shift_y = (arange(feature_map_shape[0]) + 0.5) * stride

        shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
        shift_x = tf.reshape(shift_x, [-1])
        shift_y = tf.reshape(shift_y, [-1])

        shifts = tf.stack([shift_x, shift_y, shift_x, shift_y], axis=0)
        shifts = tf.transpose(shifts)

        all_anchors = (tf.reshape(self.anchors, [1, len(self), 4]) + tf.cast(tf.reshape(shifts, [shifts.shape[0], 1, 4]), tf.float32))
        all_anchors = tf.reshape(all_anchors, [shifts.shape[0] * len(self), 4])

        return all_anchors

    def generate_anchors(self) -> tf.Tensor:
        ratios = np.array(self.aspect_ratios)
        scales = np.array(self.anchor_scales)

        anchors = np.zeros((len(self), 4))
        anchors[:, 2:] = self.base_size * np.tile(scales, (2, len(ratios))).T
        areas = anchors[:, 2] * anchors[:, 3]

        # Correct for ratios
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

        return tf.constant(anchors, dtype=tf.float32)

    def __len__(self) -> int:
        return len(self.aspect_ratios) * len(self.anchor_scales)


@tf.function(
    input_signature=[tf.TensorSpec(shape=[None, 4], dtype=tf.float32),
                     tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32),
                     tf.TensorSpec(shape=[None, None, 4], dtype=tf.float32),
                     tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                     tf.TensorSpec(shape=None, dtype=tf.int32),
                     tf.TensorSpec(shape=None, dtype=tf.float32),
                     tf.TensorSpec(shape=None, dtype=tf.float32)])

def anchor_targets_bbox(anchors: tf.Tensor, images: tf.Tensor, bndboxes: tf.Tensor, labels: tf.Tensor, num_classes: int, negative_overlap: float = 0.4, positive_overlap: float = 0.5) -> Tuple[tf.Tensor, tf.Tensor]:
    im_shape = tf.shape(images)
    batch_size = im_shape[0]
    h = tf.cast(im_shape[1], tf.float32)
    w = tf.cast(im_shape[2], tf.float32) 

    positive_indices, ignore_indices, argmax_overlaps_inds = compute_gt_annotations(anchors, bndboxes, negative_overlap, positive_overlap)

    # Expand ignore indices with out of image anchors
    x_anchor_centre = (anchors[:, 0] + anchors[:, 2]) / 2.
    y_anchor_centre = (anchors[:, 1] + anchors[:, 3]) / 2.

    larger_x = tf.greater_equal(x_anchor_centre, w)
    lesser_x = tf.less(x_anchor_centre, 0)
    out_x = tf.logical_or(larger_x, lesser_x)

    larger_y = tf.greater_equal(y_anchor_centre, h)
    lesser_y = tf.less(y_anchor_centre, 0)
    out_y = tf.logical_or(larger_y, lesser_y)

    out_mask = tf.logical_or(out_x, out_y)
    ignore_indices = tf.logical_or(ignore_indices, out_mask)

    # Gather classification labels
    chose_labels = tf.gather_nd(labels, argmax_overlaps_inds)
    chose_labels = tf.reshape(chose_labels, [batch_size, -1])

    # Labels per anchor 
    # if is positive index add the class, else 0
    # To ignore the label add -1
    labels_per_anchor = tf.where(positive_indices, chose_labels, -1)
    labels_per_anchor = tf.where(ignore_indices, -1, labels_per_anchor)
    labels_per_anchor = tf.one_hot(labels_per_anchor, 
                                   axis=-1, depth=num_classes)
    labels_per_anchor = tf.cast(labels_per_anchor, tf.float32)

    # Add regression for each anchor
    chose_bndboxes = tf.gather_nd(bndboxes, argmax_overlaps_inds)
    chose_bndboxes = tf.reshape(chose_bndboxes, [batch_size, -1, 4])
    regression_per_anchor = bbox_transform(anchors, chose_bndboxes)
    
    # Generate extra label to add the state of the label. 
    # (It should be ignored?)
    indices = tf.cast(positive_indices, tf.float32)
    indices = tf.where(ignore_indices, -1., indices)
    indices = tf.expand_dims(indices, -1)

    labels_per_anchor = tf.concat([labels_per_anchor, indices], axis=-1)
    regression_per_anchor = tf.concat([regression_per_anchor, indices], axis=-1)

    return regression_per_anchor, labels_per_anchor

def compute_gt_annotations(anchors: tf.Tensor, annotations: tf.Tensor, negative_overlap: float = 0.4, positive_overlap: float = 0.5) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(annotations)[0]
    n_anchors = tf.shape(anchors)[0]

    # Cast and reshape inputs to expected values
    anchors = tf.expand_dims(anchors, 0)
    anchors = tf.cast(anchors, tf.float32)
    anchors = tf.tile(anchors, [batch_size, 1, 1])
    annotations = tf.cast(annotations, tf.float32)

    # Compute the ious between boxes, and get the argmax indices and max values
    overlaps = bndbox.bbox_overlap(anchors, annotations)
    argmax_overlaps_inds = tf.argmax(overlaps, axis=-1, output_type=tf.int32)
    max_overlaps = tf.reduce_max(overlaps, axis=-1)
    
    # Generate index like [batch_idx, max_overlap]	
    batched_indices = tf.ones([batch_size, n_anchors], dtype=tf.int32)
    batched_indices = tf.multiply(tf.expand_dims(tf.range(batch_size), -1),
                                  batched_indices)
    batched_indices = tf.reshape(batched_indices, [-1, 1])
    argmax_inds = tf.reshape(argmax_overlaps_inds, [-1, 1])
    batched_indices = tf.concat([batched_indices, argmax_inds], -1)

    # Assign positive indices. 
    positive_indices = tf.greater_equal(max_overlaps, positive_overlap)
    
    # Assign ignored boxes
    ignore_indices = tf.greater(max_overlaps, negative_overlap)
    ignore_indices = tf.logical_and(ignore_indices, 
                                    tf.logical_not(positive_indices))
    ignore_indices = tf.logical_or(ignore_indices, tf.less(max_overlaps, 0.))

    return positive_indices, ignore_indices, batched_indices

def bbox_transform(anchors: tf.Tensor, gt_boxes: tf.Tensor) -> tf.Tensor:
    """Compute bounding-box regression targets for an image."""

    anchors = tf.cast(anchors, tf.float32)
    gt_boxes = tf.cast(gt_boxes, tf.float32)

    Px = (anchors[..., 0] + anchors[..., 2]) / 2.
    Py = (anchors[..., 1] + anchors[..., 3]) / 2.
    Pw = anchors[..., 2] - anchors[..., 0]
    Ph = anchors[..., 3] - anchors[..., 1]

    Gx = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2.
    Gy = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2.
    Gw = gt_boxes[..., 2] - gt_boxes[..., 0]
    Gh = gt_boxes[..., 3] - gt_boxes[..., 1]

    tx = (Gx - Px) / Pw
    ty = (Gy - Py) / Ph
    tw = tf.math.log(Gw / Pw)
    th = tf.math.log(Gh / Ph)
    
    targets = tf.stack([tx, ty, tw, th], axis=-1)
    
    return targets