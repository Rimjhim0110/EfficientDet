from typing import Optional
import tensorflow as tf
import efficientnet.tfkeras as efficientnet

def efficientnet_backbone(model_version: int = 0, pretrained_weights: Optional[str] = 'imagenet') -> tf.keras.Model:
    """
    Create an EfficientNet backbone model with intermediate feature outputs

    Parameters:
    model_version (int): The version of the EfficientNet architecture to use
    pretrained_weights (str, optional): If 'imagenet', use weights pre-trained on the ImageNet dataset
                                        If 'none' or any other value, initialize the model with random weights

    Returns:
    tf.keras.Model: An EfficientNet backbone model with intermediate feature outputs.
    """
    # Get the EfficientNet class corresponding to the specified model version
    efficientnet_cls = getattr(efficientnet, f'EfficientNetB{model_version:d}')
    
    # Create the base model with or without pre-trained weights
    base_model = efficientnet_cls(weights=pretrained_weights, include_top=False)

    # Retrieve the layers of the base model
    layers = base_model.layers
    intermediate_features = []

    # Iterate through the layers to find down-sampling layers and collect intermediate features
    for curr_layer, next_layer in zip(layers[:-1], layers[1:]):
        if hasattr(next_layer, 'strides') and next_layer.strides[0] == 2:
            # If the next layer has stride 2, it's a down-sampling layer, so add the current layer to the features
            intermediate_features.append(curr_layer)
    
    # Add the last layer to the intermediate features list
    intermediate_features.append(next_layer)

    # Create a new model that takes the same input as the base model and provides intermediate feature outputs
    intermediate_model = tf.keras.Model(base_model.input, outputs=[f.output for f in intermediate_features[-5:]], name=f'EfficientNetB{model_version:d}')

    return intermediate_model