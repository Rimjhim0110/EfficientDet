import json
import base64
import hashlib
from pathlib import Path
from typing import Union, Any, Tuple
from urllib.parse import urlparse

from model.efficientdet import EfficientDet

def _md5(fname: Union[Path, str]) -> str:
    hash_md5 = hashlib.md5()
    with open(str(fname), "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return base64.b64encode(hash_md5.digest()).decode()


def save(model: EfficientDet, parameters: dict, save_dir: Union[str, Path], to_gcs: bool = False) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    model_fname = save_dir / 'model.tf'
    hp_fname = save_dir / 'hp.json'
    
    json.dump(parameters, hp_fname.open('w'))
    model.save_weights(str(model_fname))
    
    if to_gcs:
        from google.cloud import storage

        client = storage.Client(project='ml-generic-purpose')
        bucket = client.bucket('ml-generic-purpose-tf-models')
        prefix = save_dir.stem

        blob = bucket.blob(f'{prefix}/hp.json')
        blob.upload_from_filename(str(hp_fname))

        model.save_weights(
            f'gs://ml-generic-purpose-tf-models/{prefix}/model.tf')


def download_folder(gs_path: str) -> Union[str, Path]:
    from google import auth # type: ignore[attr-defined]
    from google.cloud import storage

    save_dir_url = urlparse(str(gs_path))
    assert save_dir_url.scheme == 'gs'

    save_dir = Path.home() / '.effdet-checkpoints' / save_dir_url.path[1:]
    save_dir.mkdir(exist_ok=True, parents=True)
    
    client = storage.Client(
        project='ml-generic-purpose',
        credentials=auth.credentials.AnonymousCredentials())
    bucket = client.bucket('ml-generic-purpose-tf-models')

    prefix = save_dir_url.path[1:] + '/'
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        blob.reload()
        name = blob.name.replace(prefix, '')
        fname = save_dir / name
        fname.parent.mkdir(parents=True, exist_ok=True)
        if not fname.exists() or (fname.exists() and 
                                    _md5(fname) != blob.md5_hash):
            blob.download_to_filename(str(fname))

    return save_dir


def load(save_dir_or_url: Union[str, Path], **kwargs: Any) -> Tuple['EfficientDet', dict]:
    save_dir_url_parsed = urlparse(str(save_dir_or_url))

    if save_dir_url_parsed.scheme == 'gs':
        save_dir = Path(download_folder(str(save_dir_or_url)))

        hp_fname = save_dir / 'hp.json'
        model_path = save_dir / 'model.h5'

    else:
        hp_fname = Path(save_dir_or_url) / 'hp.json'
        model_path = Path(save_dir_or_url) / 'model.h5'

    assert hp_fname.exists()

    with hp_fname.open() as f:
        hp = json.load(f)

    from efficientdet import EfficientDet
    from config import EfficientDetCompudScaling

    conf = EfficientDetCompudScaling(D=hp['efficientdet'])

    model = EfficientDet(hp['n_classes'], D=hp['efficientdet'], bidirectional=hp['bidirectional'], freeze_backbone=hp['freeze_backbone'], training_mode=True, weights=None, **kwargs)
    
    model.build([None, *conf.input_size, 3])

    print('Loading model weights from {}...'.format(str(model_path)), end='')
    model.load_weights(str(model_path))
    print(' done')
    
    model.training_mode = False
    
    for l in model.layers:
        l.trainable = False
    model.trainable = False

    return model, hp
