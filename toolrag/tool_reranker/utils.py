import random
import numpy as np
import torch

# ... (rest of your imports and code) ...

def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across different libraries.

    Args:
        seed (int): The integer seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ... (rest of your code) ...

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed) # Call the function here
    train(args)

    gc.collect()
    torch.cuda.empty_cache()