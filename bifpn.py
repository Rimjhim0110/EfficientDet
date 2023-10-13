import tensorflow as tf

from typing import List
from . import cnn_layers
from utils import cascade_layers

EPSILON = 1e-5

class FastFusion(tf.keras.layers.Layer):
    def __init__(self, size: int, features: int, prefix: str = '') -> None:
        super(FastFusion, self).__init__()

        self.size = size
        self.w = self.add_weight(name=prefix + 'w', shape=(size,), initializer=tf.initializers.Ones(), trainable=True)
        self.relu = tf.keras.layers.Activation('relu', name=prefix + 'relu')
        self.conv = cnn_layers.ConvBlock(features, separable_conv=True, kernel_size=3, strides=1, padding='same', activation='swish', layer_prefix=prefix + 'conv_block/')
        self.resize = cnn_layers.ResizeLayer(features, prefix=prefix + 'resize/')
        #naming is done to identify each layer (prefix + '..')

    def call(self, inputs: List[tf.Tensor], training: bool = True) -> tf.Tensor:
        #shape of [tf.Tensor] if (batch,h,w)
      
        #last feature is resized according to the other inputs
        resampled_feature = self.resize(inputs[-1], tf.shape(inputs[0]), training=training)
        resampled_features = inputs[:-1] + [resampled_feature]

        # wi has to be larger than 0 -> relu
        w = self.relu(self.w)
        w_sum = EPSILON + tf.reduce_sum(w, axis=0)

        #using fast normalized fusion instead of softmax
        weighted_inputs = [(w[i] * resampled_features[i]) / w_sum for i in range(self.size)]
      
        weighted_sum = tf.add_n(weighted_inputs)
        return self.conv(weighted_sum, training=training)
        

class BiFPNBlock(tf.keras.layers.Layer):
    def __init__(self, features: int, prefix: str = '') -> None:
        super(BiFPNBlock, self).__init__()

        #feature fusion for intermediate level
        #ff is Feature fusion
        #td is intermediate level
        self.ff_6_td = FastFusion(2, features, prefix=prefix + 'ff_6_td_P6-P7_/')
        self.ff_5_td = FastFusion(2, features, prefix=prefix + 'ff_5_td_P5_P6_td/')
        self.ff_4_td = FastFusion(2, features, prefix=prefix + 'ff_4_td_P4_P5_td/')

        #for output
        self.ff_7_out = FastFusion(2, features, prefix=prefix + 'ff_7_out_P7_P6_td/')
        self.ff_6_out = FastFusion(3, features, prefix=prefix + 'ff_6_out_P6_P6_td_P7_out/')
        self.ff_5_out = FastFusion(3, features, prefix=prefix + 'ff_5_out_P5_P5_td_P4_out/')
        self.ff_4_out = FastFusion(3, features, prefix=prefix + 'ff_4_out_P4_P4_td_P3_out/')
        self.ff_3_out = FastFusion(2, features, prefix=prefix + 'ff_3_out_P3_P4_td/')

    def call(self, features: List[tf.Tensor], training: bool = True) -> List[tf.Tensor]:
        #feature fusion of bottom-up features from backbone 

        #features include the maps from conv at each stage in backbone
        P3, P4, P5, P6, P7 = features

        #intermediate state
        P6_td = self.ff_6_td([P6, P7], training=training)
        P5_td = self.ff_5_td([P5, P6_td], training=training)
        P4_td = self.ff_4_td([P4, P5_td], training=training)

        #out feature maps
        P3_out = self.ff_3_out([P3, P4_td], training=training)
        P4_out = self.ff_4_out([P4, P4_td, P3_out], training=training)
        P5_out = self.ff_5_out([P5, P5_td, P4_out], training=training)
        P6_out = self.ff_6_out([P6, P6_td, P5_out], training=training)
        P7_out = self.ff_7_out([P7, P6_td], training=training)

        return [P3_out, P4_out, P5_out, P6_out, P7_out]


class BiFPN(tf.keras.Model):
    def __init__(self, features: int = 64, n_blocks: int = 3, prefix: str = '') -> None:
        super(BiFPN, self).__init__()

        #pixel-wise for each feature comming from the bottom-up
        self.pixel_wise = [cnn_layers.ConvBlock(features, kernel_size=1, layer_prefix=prefix + f'pixel_wise_{i}/') for i in range(3)] 

        self.gen_P6 = cnn_layers.ConvBlock(features, kernel_size=3, strides=2, padding='same', layer_prefix=prefix + 'gen_P6/')
        self.relu = tf.keras.layers.Activation('relu', name=prefix + 'relu')

        self.gen_P7 = cnn_layers.ConvBlock(features, kernel_size=3, strides=2, padding='same', layer_prefix=prefix + 'gen_P7/')

        self.blocks = [BiFPNBlock(features, prefix=prefix + f'block_{i}/') for i in range(n_blocks)]

    def call(self, inputs: List[tf.Tensor], training: bool = True) -> List[tf.Tensor]:
        #each Pin has shape(batch, H, W, C)
        #channels are reduced using pixel-wise conv
      
        _, _, *C = inputs
        # first 2 elements of 'inputs' are not required
      
        P3, P4, P5 = [self.pixel_wise[i](C[i], training=training) for i in range(len(C))]
        P6 = self.gen_P6(C[-1], training=training)
        P7 = self.gen_P7(self.relu(P6), training=training)

        features = [P3, P4, P5, P6, P7]
        
        features = cascade_layers.cascade_layers(self.blocks, features, training=training)
        return features
