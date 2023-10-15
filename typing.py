from typing import Tuple #typing module is used to type hints and annotations 
import tensorflow as tf

Annotation = Tuple[tf.Tensor, tf.Tensor]
ObjectDetectionInstance = Tuple[tf.Tensor, Annotation]
