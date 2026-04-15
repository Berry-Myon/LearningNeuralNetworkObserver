import os
import pickle
import numpy as np
import torch


_utils_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_utils_dir))
_checkpoint_dir = os.path.join(_project_root, 'Checkpoints')


def ensure_checkpoint_dir():
    os.makedirs(_checkpoint_dir, exist_ok=True)
    return _checkpoint_dir


def _normalize_checkpoint_value(value):
    if isinstance(value, np.ndarray):
        return torch.tensor(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _normalize_checkpoint_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_normalize_checkpoint_value(v) for v in value)
    if isinstance(value, list):
        return [_normalize_checkpoint_value(v) for v in value]
    return value


def _torch_load_compatible(checkpoint_path, map_location):
    try:
        return torch.load(checkpoint_path, map_location=map_location)
    except pickle.UnpicklingError:
        # Backward compatibility for checkpoints created before PyTorch 2.6
        # when torch.load defaulted to weights_only=False.
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions may not expose weights_only; retry plainly.
        return torch.load(checkpoint_path, map_location=map_location)


def get_checkpoint_path(model_name, stage):
    ensure_checkpoint_dir()
    return os.path.join(_checkpoint_dir, f'{model_name.lower()}_{stage}.pt')


def save_checkpoint(model_name, stage, model, optimizer=None, epoch=None, loss=None, extra_state=None):
    payload = {
        'model_name': model_name,
        'stage': stage,
        'model_state_dict': model.state_dict(),
    }
    if optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        payload['epoch'] = int(epoch)
    if loss is not None:
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().cpu().item()
        payload['loss'] = float(loss)
    if extra_state is not None:
        payload.update(extra_state)

    checkpoint_path = get_checkpoint_path(model_name, stage)
    torch.save(_normalize_checkpoint_value(payload), checkpoint_path)
    return checkpoint_path


def load_checkpoint(model_name, stage, device=None):
    checkpoint_path = get_checkpoint_path(model_name, stage)
    if not os.path.exists(checkpoint_path):
        return None
    map_location = device if device is not None else 'cpu'
    return _torch_load_compatible(checkpoint_path, map_location)


def save_named_checkpoint(name, payload):
    ensure_checkpoint_dir()
    checkpoint_path = os.path.join(_checkpoint_dir, f'{name}.pt')
    torch.save(_normalize_checkpoint_value(payload), checkpoint_path)
    return checkpoint_path


def load_named_checkpoint(name, device=None):
    checkpoint_path = os.path.join(_checkpoint_dir, f'{name}.pt')
    if not os.path.exists(checkpoint_path):
        return None
    map_location = device if device is not None else 'cpu'
    return _torch_load_compatible(checkpoint_path, map_location)


def load_pretrained_parameters(model, model_name, device=None):
    checkpoint = load_checkpoint(model_name, 'pretrain', device=device)
    if checkpoint is None:
        raise FileNotFoundError(
            f'No pretrain checkpoint found for {model_name}. '
            f'Expected {get_checkpoint_path(model_name, "pretrain")}.'
        )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return get_checkpoint_path(model_name, 'pretrain')
