from pathlib import Path


class Config:
    # Paths
    DATA_DIR = Path("data")
    CHECKPOINT_DIR = Path("checkpoints")

    # Files
    MEMORY_FILE = DATA_DIR / "ddpg_mem.p"
    SCORES_FILE = CHECKPOINT_DIR / "scores.pkl"
    CHECKPOINT_PATH = CHECKPOINT_DIR / "ddpg_checkpoint"

    # Training parameters
    EPISODES = 1000
    MAX_STEPS = 100000
    SAVE_FREQUENCY = 10

    # Testing parameters
    TEST_EPISODES = 1000
    TEST_MAX_STEPS = 10000

    def __init__(self) -> None:
        # Create necessary directories
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
