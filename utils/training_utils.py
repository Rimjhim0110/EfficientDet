import tensorflow as tf
import functools

from typing import Tuple
from . import anchors as anchors_utils
from .. import config
from ..typing import ObjectDetectionInstance


def compute_truth_values(images: tf.Tensor, annot: Tuple[tf.Tensor, tf.Tensor], anchors: tf.Tensor,num_classes: int) -> ObjectDetectionInstance:

    labels = annot[0]
    boxes = annot[1]
    #computing target values for regression(for better fitting anchor boxes and localizing objects) and classification (assigning label to bounding box)
    regression_target, classification_target = anchors_utils.anchor_targets_bbox(anchors, images, boxes, labels, num_classes)
    return images, (regression_target, classification_target)


def generate_anchor_boxes(anchors_config: config.AnchorsConfig,img_shape: int) -> tf.Tensor:

    anchors_gen = [anchors_utils.AnchorGenerator(size=anchors_config.sizes[i - 3], aspect_ratios=anchors_config.ratios, stride=anchors_config.strides[i - 3]) for i in range(3, 8)]
    
    #anchor box shapes at different scales relative to the input image size
    shapes = [img_shape // (2 ** x) for x in range(3, 8)]
    
    #generate list of anchor boxes
    anchors = [g((size, size, 3)) for g, size in zip(anchors_gen, shapes)]

    #stacking all anchor boxes vertically 
    return tf.concat(anchors, axis=0)


def wrap_detection_dataset(ds: tf.data.Dataset,img_size: Tuple[int, int],num_classes: int) -> tf.data.Dataset:

    anchors = generate_anchor_boxes(config.AnchorsConfig(), img_size[0])

    # Wrap datasets so they return the anchors labels
    dataset_training_head_fn = functools.partial(compute_truth_values, anchors=anchors, num_classes=num_classes)

    return ds.map(dataset_training_head_fn)
