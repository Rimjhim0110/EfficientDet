import tensorflow as tf
from typing import Mapping
from . import coco

#calculate mean Average Precision for COCO
class COCOmAPCallback(tf.keras.callbacks.Callback):

    def __init__(self, validation_data: tf.data.Dataset,class_name_index: Mapping[str, int], validate_every: int = 1,mAP_print_freq: int = 10) -> None:
        self.validation_data = validation_data
        self.gtCOCO = coco.tf_data_to_COCO(validation_data, class_name_index)

        self.class_name_index = class_name_index
        self.validate_every = validate_every
        self.mAP_print_freq = mAP_print_freq

    #periodic validation
    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        #validation performed only if epoch multiple of validate_every
        if (epoch + 1) % self.validate_every == 0:
            self.model.training_mode = False
            coco.evaluate(self.model, self.validation_data, self.gtCOCO,sum(1 for _ in self.validation_data),self.print_freq)

            #again set to true after mAP evaluation completion
            self.model.training_mode = True
