import tensorflow as tf
from typing import Any, Sequence

def cascade_layers(layers: Sequence[tf.keras.layers.Layer], input_data: Any, is_training: bool = True) -> Any:

    # Initialize the output variable with the input data.
    output_data = input_data
    
    # Iterate through the layers in the sequence.
    for layer in layers:
        # Apply the current layer to the output data, specifying the training mode.
        output_data = layer(output_data, training=is_training)
    
    # Return the final output data after applying all layers in the cascade.
    return output_data