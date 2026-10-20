"""Central seeding. Search runs: seeds 0-9. Final trainings: seeds 0-4."""
import random
import numpy as np

def set_all_seeds(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed
