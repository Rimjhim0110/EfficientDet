import json
from pathlib import Path
from typing import Mapping, Any

import click
import tensorflow as tf
import tensorflow_addons as tfa
import efficientdet

from .callbacks import COCOmAPCallback

def train(config: config.EfficientDetCompudScaling, save_checkpoint_dir: Path, ds: tf.data.Dataset, val_ds: tf.data.Dataset, class2idx: Mapping[str, int] , **kwargs: Any) -> None:

    weights_file = str(save_checkpoint_dir / 'model.h5')
    im_size = config.input_size

    steps_per_epoch = sum(1 for _ in ds)
    if val_ds is not None:
        validation_steps = sum(1 for _ in val_ds)
    else:
        validation_steps = 0

    if kwargs['from_pretrained'] is not None:
        model = EfficientDet(weights=kwargs['from_pretrained'], num_classes=len(class2idx), custom_head_classifier=True, freeze_backbone=kwargs['freeze_backbone'], training_mode=True)
        print('Training from a pretrained model...')
        print('This will override any configuration related to EfficientNet'
              ' using the defined in the pretrained model.')
    else:
        model = EfficientDet(len(class2idx), D=kwargs['efficientdet'], bidirectional=kwargs['bidirectional'], freeze_backbone=kwargs['freeze_backbone'], weights='imagenet', training_mode=True)

    if kwargs['w_scheduler']:
        lr = optim.WarmupCosineDecayLRScheduler(kwargs['learning_rate'], warmup_steps=steps_per_epoch, decay_steps=steps_per_epoch * (kwargs['epochs'] - 1), alpha=kwargs['alpha'])
    else:
        lr = kwargs['learning_rate']

    optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=4e-5)

    regression_loss_fn = loss_functions.EfficientDetHuberLoss()
    clf_loss_fn = loss_functions.EfficientDetFocalLoss()
    
    # Wrap to return anchors labels
    wrapped_ds = wrap_detection_dataset(ds, im_size=im_size, num_classes=len(class2idx))

    wrapped_val_ds = wrap_detection_dataset(val_ds, im_size=im_size, num_classes=len(class2idx))
    
    model.compile(loss=[regression_loss_fn, clf_loss_fn], optimizer=optimizer, loss_weights=[1., 1.])
    #to create model specs
    model.build([None, *im_size, 3])
    model.summary()

    if kwargs['checkpoint'] is not None:
        model.load_weights(str(Path(kwargs['checkpoint']) / 'model.h5'))

    model.save_weights(weights_file)
    kwargs.update(n_classes=len(class2idx))
    json.dump(kwargs, (save_checkpoint_dir / 'hp.json').open('w'))

    callbacks = [COCOmAPCallback(val_ds, class2idx, print_freq=kwargs['print_freq'], validate_every=kwargs['validate_freq']),
                 tf.keras.callbacks.ModelCheckpoint(weights_file, 
                                                    save_best_only=True)]

    model.fit(wrapped_ds.repeat(), validation_data=wrapped_val_ds, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=kwargs['epochs'], callbacks=callbacks, shuffle=False)

def labelme(ctx: click.Context, **kwargs: Any) -> None:
    kwargs.update(ctx.obj['common'])
    
    config = ctx.obj['config']
    save_checkpoint_dir = ctx.obj['save_checkpoint_dir']

    _, class2idx = utils.preprocessing.read_class_names(
        kwargs['classes_file'])

    im_size = config.input_size

    train_ds = labelme.build_dataset(annotations_path=kwargs['root_train'], images_path=kwargs['images_path'], class2idx=class2idx, im_input_size=im_size, shuffle=True)
    
    train_ds.map(augment.RandomHorizontalFlip())
    train_ds.map(augment.RandomCrop())

    train_ds = train_ds.padded_batch(batch_size=kwargs['batch_size'],
                                     padded_shapes=((*im_size, 3), 
                                                    ((None,), (None, 4))),
                                     padding_values=(0., (-1, -1.)))

    valid_ds = None
    if kwargs['root_valid']:
        valid_ds = data.labelme.build_dataset(annotations_path=kwargs['root_valid'], images_path=kwargs['images_path'], class2idx=class2idx, im_input_size=im_size, shuffle=False)
        
        valid_ds = valid_ds.padded_batch(batch_size=kwargs['batch_size'], padded_shapes=((*im_size, 3), 
                                                        ((None,), (None, 4))),
                                         padding_values=(0., (-1, -1.)))

    train(config, save_checkpoint_dir, 
          train_ds, valid_ds, class2idx, **kwargs)


if __name__ == "__main__":
    main()
