import abc
import tensorflow as tf

import efficientdet.utils.bndbox as bb_utils
from typing import Tuple
from efficientdet.typing import Annotation, ObjectDetectionInstance


@tf.function
def horizontal_flip(image: tf.Tensor, annot: Annotation) -> ObjectDetectionInstance:
    labels = annot[0]
    boxes = annot[1]

    img_shape = tf.shape(image)
    #extract height and width from the computed image shape
    h, w = img_shape[0], img_shape[1]

    # Flip the image horizontally
    image = tf.image.flip_left_right(image)
    
    # Flip the box horizontally to match the image
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=-1)

    #width of bounding box
    bb_width = x2 - x1
    delta_W = tf.expand_dims(boxes[:, 0], axis=-1)

    #left edge flipping
    x1 = tf.cast(w, tf.float32) - delta_W - bb_width
    #right edge flipping
    x2 = tf.cast(w, tf.float32) - delta_W

    boxes = tf.stack([x1, y1, x2, y2], axis=-1)
    boxes = tf.reshape(boxes, [-1, 4])

    return image, (labels, boxes)


@tf.function
def crop(image: tf.Tensor, annot: Annotation) -> ObjectDetectionInstance:
    
    labels = annot[0]
    boxes = annot[1]

    img_shape = tf.shape(image)
    #extract height and width from the computed image shape
    h, w = img_shape[0], img_shape[1]

    # Get random crop dims of width and height
    crop_factor_w = tf.random.uniform(shape=[], minval=.4, maxval=1.)
    crop_factor_h = tf.random.uniform(shape=[], minval=.4, maxval=1.)

    crop_width = tf.cast(tf.cast(w, tf.float32) * crop_factor_w, tf.int32)
    crop_height = tf.cast(tf.cast(h, tf.float32) * crop_factor_h, tf.int32)

    #Random starting coordinates generated so that the crop stays within the bounds of the original image.
    x = tf.random.uniform(shape=[], maxval=w - crop_width - 1, dtype=tf.int32)
    y = tf.random.uniform(shape=[], maxval=h - crop_height - 1, dtype=tf.int32)

    # Crop the image and resize it back to original size
    crop_img = tf.image.crop_to_bounding_box(image, y, x, crop_height, crop_width)
    crop_img = tf.image.resize(crop_img, (h, w))
    
    # Clip the boxes to fit inside the crop
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=-1)
    
    # Cast crop coordinates to float, so they can be used for clipping
    x = tf.cast(x, tf.float32)
    crop_width = tf.cast(crop_width, tf.float32)
    y = tf.cast(y, tf.float32)
    crop_height = tf.cast(crop_height, tf.float32)
    
    #Update bound box coordinates after crop
    widths = x2 - x1
    heights = y2 - y1

    x1 = x1 - x
    y1 = y1 - y
    x2 = x1 + widths
    y2 = y1 + heights

    boxes = tf.stack([x1, y1, x2, y2], axis=-1)
    boxes = tf.reshape(boxes, [-1, 4])
    boxes = bb_utils.clip_boxes(tf.expand_dims(boxes, 0), (crop_height, crop_width))
    boxes = tf.reshape(boxes, [-1, 4])

    #Masking to exclude small bounding boxes
    widths = tf.gather(boxes, 2, axis=-1) - tf.gather(boxes, 0, axis=-1)
    heights = tf.gather(boxes, 3, axis=-1) - tf.gather(boxes, 1, axis=-1)
    areas = widths * heights
    
    #Remove boxes with area less than 1%
    min_area = .01 * (crop_height * crop_width)
    large_areas = tf.reshape(tf.greater_equal(areas, min_area), [-1])

    #Using only larger area boxes 
    boxes = tf.boolean_mask(boxes, large_areas, axis=0)
    labels = tf.boolean_mask(labels, large_areas)

    boxes = bb_utils.scale_boxes(boxes, image.shape[:-1], (crop_height, crop_width))

    return crop_img, (labels, boxes)

@tf.function
def erase(image: tf.Tensor, annot: Annotation,patch_aspect_ratio: Tuple[float, float] = (.2, .2)) -> ObjectDetectionInstance:

    img_shape = tf.shape(image)
    h, w = img_shape[0], img_shape[1]
    h = tf.cast(h, tf.float32)
    w = tf.cast(w, tf.float32)
    
    # Generate patch
    h_prop = tf.random.uniform(shape=[], minval=0, maxval=patch_aspect_ratio[0])
    w_prop = tf.random.uniform(shape=[], minval=0, maxval=patch_aspect_ratio[1])
    patch_h = tf.cast(tf.multiply(h, h_prop), tf.int32)
    patch_w = tf.cast(tf.multiply(w, w_prop), tf.int32)
    patch = tf.zeros([patch_h, patch_w], tf.float32)

    # Generate random location for patches
    x = tf.random.uniform(shape=[1], maxval=tf.cast(w, tf.int32) - patch_w, dtype=tf.int32)
    y = tf.random.uniform(shape=[1], maxval=tf.cast(h, tf.int32) - patch_h, dtype=tf.int32)
    
    # Pad patch with ones so it has the same shape as the image
    pad_vert = tf.concat([y, tf.cast(h, tf.int32) - y - patch_h], axis=0)
    pad_hor = tf.concat([x, tf.cast(w, tf.int32) - x - patch_w], axis=0)
    paddings = tf.stack([pad_vert, pad_hor])
    paddings = tf.cast(paddings, tf.int32)
    
    patch = tf.pad(patch, paddings, constant_values=1.)

    return tf.multiply(image, tf.expand_dims(patch, -1)), annot


@tf.function
def no_transform(image: tf.Tensor, annot: Annotation) -> ObjectDetectionInstance:
    return image, annot


class Augmentation(abc.ABC):

    def __init__(self, prob: float = .5) -> None:
        self.prob = prob

    @abc.abstractmethod
    def augment(self, image: tf.Tensor, annot: Annotation) -> ObjectDetectionInstance:
        raise NotImplementedError

    def __call__(self,image: tf.Tensor, annot: Annotation) -> ObjectDetectionInstance:

        image, annot = tf.cond(tf.random.uniform([1]) < self.prob,lambda: horizontal_flip(image, annot),lambda: no_transform(image, annot))

        return image, annot


class RandomHorizontalFlip(Augmentation):

    def __init__(self, prob: float = .5) -> None:
        super(RandomHorizontalFlip, self).__init__(prob=prob)

    def augment(self, image: tf.Tensor, annot: Annotation) -> ObjectDetectionInstance:
        return horizontal_flip(image, annot)


class RandomCrop(Augmentation):

    def __init__(self, prob: float = .5) -> None:
        super(RandomCrop, self).__init__(prob=prob)

    def augment(self, image: tf.Tensor, annot: Annotation) -> ObjectDetectionInstance:
        return crop(image, annot)


class RandomErase(Augmentation):

    def __init__(self, prob: float = .5, patch_proportion: Tuple[float, float] = (.2, .2)) -> None:
        super(RandomErase, self).__init__(prob=prob)
        self.patch_proportion = patch_proportion

    def augment(self, image: tf.Tensor, annot: Annotation) -> ObjectDetectionInstance:
        return erase(image, annot, self.patch_proportion)
