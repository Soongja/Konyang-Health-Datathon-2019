import os
import torch
from collections import OrderedDict


def get_initial_checkpoint(config):
    checkpoint_dir = os.path.join(config.LOCAL_TRAIN_DIR, 'checkpoints')
    checkpoints = [checkpoint
                   for checkpoint in os.listdir(checkpoint_dir)
                   if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth')]
    if checkpoints:
        return os.path.join(checkpoint_dir, list(sorted(checkpoints))[-1])
    return None


def load_checkpoint(config, model, checkpoint):
    print('load checkpoint from', checkpoint)
    checkpoint = torch.load(checkpoint)
    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict)

    last_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else -1
    score = checkpoint['score'] if 'score' in checkpoint else -1
    loss = checkpoint['loss'] if 'loss' in checkpoint else float('inf')

    return last_epoch, score, loss


def save_checkpoint(config, model, epoch, score, loss, weights_dict=None, name=None):
    checkpoint_dir = os.path.join(config.LOCAL_TRAIN_DIR, 'checkpoints')

    if name:
        checkpoint_path = os.path.join(checkpoint_dir, '%s.pth' % name)
    else:
        _score = '%.10f' % score
        _score = _score.replace(".", "")
        _score = _score[:4] + '.' + _score[4:8] + '.' + _score[8:]
        checkpoint_path = os.path.join(checkpoint_dir, 'epoch_%02d_score%s_loss%.4f.pth' % (epoch, _score, loss))

    if weights_dict is None:
        weights_dict = {
            'state_dict': model.state_dict(),
            'epoch' : epoch,
            'score': score,
            'loss': loss
        }
    torch.save(weights_dict, checkpoint_path)
