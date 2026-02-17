#train/test split + walk-forward CV
import numpy as np

def time_train_test_split(n: int, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns indices for a chronological split:
    - Train = first (1-test_size) portion
    - Test  = last test_size portion
    
    Why it matters:
    - Random splits leak future information into training.
    - Time split simulates realistic forecasting.
    """
    test_n = int(np.floor(n * test_size))
    if test_n < 30:
        raise ValueError("Test set too small. Increase date range or test_size.")
    train_end = n - test_n
    return np.arange(train_end), np.arange(train_end, n)
