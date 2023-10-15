import tensorflow as tf
from typing import Tuple, Union, Sequence, Mapping
from pathlib import Path

def normalize_image(image: tf.Tensor) -> tf.Tensor:
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    return (image - mean) / std

def unnormalize_image(image: tf.Tensor) -> tf.Tensor:
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    return image * std + mean

def preprocess_image(img_path: str,img_size: Tuple[int, int],is_normalization_reqd: bool = False) -> tf.Tensor:
                
    #Reads image frm the path and reads contents of file as binary string
    img = tf.io.read_file(img_path)

    #Decode string as jpeg image with RGB Channels ( 3 channels)
    img = tf.image.decode_jpeg(img, channels=3)

    #converting image to same range pixel values 
    img = tf.image.convert_image_dtype(img, tf.float32)
    if is_normalization_reqd:
        img = normalize_image(img)
        
    return tf.image.resize(img, img_size)

def classnames_mapping(filename: Union[str, Path]) -> Tuple[Sequence[str], Mapping[str, int]]:
    class_name = Path(filename).read_text().split('\n')
    class_name = [cl.strip() for cl in class_name]

    #mapping classnames to index
    class_name_index = {cl: i for i, cl in enumerate(class_name)}

    return class_name, class_name_index
