# checkpoint_utils.py

import os
import torch
from collections import deque

def save_checkpoint(agent, path='checkpoint.pth'):
    ckpt = {
        'model_state': agent.model.state_dict(),
        'optim_state': agent.trainer.optimizer.state_dict(),
        'epsilon':     agent.epsilon,
        'n_games':     agent.n_games,
        # only save replay if you really need it
        'memory':      list(agent.memory),
    }
    torch.save(ckpt, path)


def load_checkpoint(agent, path='checkpoint.pth', max_memory=100_000):
    """
    Loads whatever layers will fit and restores optimizer + schedule.
    Returns True if any checkpoint was loaded, False otherwise.
    """
    if not os.path.exists(path):
        return False

    data = torch.load(path, weights_only=False)

    
    pretrained = data.get('model_state', {})
    model_dict = agent.model.state_dict()
    compatible = {
        k: v for k, v in pretrained.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }
    model_dict.update(compatible)
    agent.model.load_state_dict(model_dict)
    print(f"→ Loaded {len(compatible)}/{len(model_dict)} compatible layers")


    try:
        agent.trainer.optimizer.load_state_dict(data['optim_state'])
    except Exception:
        print("⚠️ Optimizer state not fully loaded (shapes may differ)")
    agent.epsilon = data.get('epsilon', getattr(agent, 'epsilon', 0))
    agent.n_games = data.get('n_games', getattr(agent, 'n_games', 0))

    agent.memory = deque(maxlen=max_memory)

    return True
