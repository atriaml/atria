def to_serializable(obj):
    import numpy as np
    import torch

    # NumPy scalar (np.float32, np.int64, etc.)
    if isinstance(obj, np.generic):
        return obj.item()

    # NumPy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Torch tensor
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    # Dict
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}

    # List / tuple
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]

    # Anything else
    return obj
