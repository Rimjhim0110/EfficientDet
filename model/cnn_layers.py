import tensorflow as tf
from typing import Any, Tuple, Sequence
from config import AnchorsConfig
from utils import bbox_utils, anchor_utils

class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, num_features: int, prefix: str = '') -> None:
        super(ResizeLayer, self).__init__()
        self.conv_block = ConvBlock(num_features, separable=True, kernel_size=3, padding='same', prefix=prefix+'conv_block/')

    def call(self, images: tf.Tensor, target_shape: Tuple[int, int, int, int] = None, training: bool = True) -> tf.Tensor:
        # Extract height and width from the target shape
        height, width = target_shape[1], target_shape[2]
        
        # Resize the input tensor
        resized_images = tf.image.resize(images, [height, width], method='nearest')
        
        # Apply the convolution block to the resized tensor
        output = self.conv_block(resized_images, training=training)
        
        return output


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, features: int = None, separable_conv: bool = False, activation: str = None, layer_prefix: str = '', **kwargs: Any) -> None:
        super(ConvBlock, self).__init__()

        # Create the convolution layer based on the choice of separable or standard convolution
        if separable_conv:
            layer_name = layer_prefix + 'separable_conv'
            self.conv_layer = tf.keras.layers.SeparableConv2D(filters=features, name=layer_name, **kwargs)
        else:
            layer_name = layer_prefix + 'conv'
            self.conv_layer = tf.keras.layers.Conv2D(features, name=layer_name, **kwargs)

        # Batch normalization layer to normalize the convolution output
        self.batch_norm = tf.keras.layers.BatchNormalization(name=layer_prefix+'batch_norm')

        # Activation function layer (default to linear if not specified)
        if activation == 'swish':
            self.activation = tf.keras.layers.Activation(tf.nn.swish, name=layer_prefix+'swish')
        elif activation is not None:
            self.activation = tf.keras.layers.Activation(activation, name=layer_prefix+activation)
        else:
            self.activation = tf.keras.layers.Activation('linear', name=layer_prefix+'linear')

    def call(self, x: tf.Tensor, is_training: bool = True) -> tf.Tensor:
        # Apply batch normalization to the output of the convolution layer
        conv_output = self.batch_norm(self.conv_layer(x), training=is_training)
        
        # Apply the activation function
        output = self.activation(conv_output)

        return output


class FilterDetections(object):
    def __init__(self, anchors_config: AnchorsConfig, score_threshold: float):
        self.score_threshold = score_threshold
        self.anchors_gen = [anchor_utils.GenerateAnchor(base_size=anchors_config.sizes[i - 3], aspect_ratios=anchors_config.ratios, stride=anchors_config.strides[i - 3]
                                                   ) for i in range(3, 8)] 
        # 3 to 7 pyramid levels

        # Accelerate calls
        self.regress_boxes = tf.function(bbox_utils.regress_bboxes, input_signature=[
                tf.TensorSpec(shape=[None, None, 4], dtype=tf.float32),
                tf.TensorSpec(shape=[None, None, 4], dtype=tf.float32)])

        self.clip_boxes = tf.function(bbox_utils.clip_boxes, input_signature=[
                tf.TensorSpec(shape=[None, None, 4], dtype=tf.float32),
                tf.TensorSpec(shape=None)])
    
    def __call__(self, images: tf.Tensor, regressors: tf.Tensor, class_scores: tf.Tensor) -> Tuple[Sequence[tf.Tensor], Sequence[tf.Tensor], Sequence[tf.Tensor]]:
        im_shape = tf.shape(images)
        batch_size, h, w = im_shape[0], im_shape[1], im_shape[2]

        # Create the anchors
        shapes = [w // (2 ** x) for x in range(3, 8)]
        anchors = [g((size, size, 3)) for g, size in zip(self.anchors_gen, shapes)]
        anchors = tf.concat(anchors, axis=0)
        
        # Tile anchors over batches, so they can be regressed
        anchors = tf.tile(tf.expand_dims(anchors, 0), [batch_size, 1, 1])

        # Regress anchors and clip in case they go outside of the image
        boxes = self.regress_boxes(anchors, regressors)
        boxes = self.clip_boxes(boxes, [h, w])

        # Suppress overlapping detections
        boxes, labels, scores = bbox_utils.nms(boxes, class_scores, score_threshold=self.score_threshold)

        return boxes, labels, scores
