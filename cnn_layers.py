import tensorflow as tf
from typing import Any, Tuple, Sequence

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
