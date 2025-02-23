import torch 

def set_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    return device

def set_seed(seed: int = 42) -> None:
    print(f"Seed: {seed}, CUDA available: {torch.cuda.is_available()}, MPS available: {torch.mps.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}, Current device: {torch.cuda.current_device()}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.manual_seed_all(seed)
    elif torch.mps.is_available():
        torch.mps.manual_seed(seed)
