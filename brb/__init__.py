import atexit
import shutil
from pathlib import Path

import fprs
import numpy as np

brb_seed = 42
brb_random_state = np.random.RandomState(seed=42)


def set_random_state(
        seed: int
        ) -> None:
    global brb_seed, brb_random_state
    brb_seed = seed
    brb_random_state.seed(seed)
    fprs.set_random_state(seed)


brb_tmp_directory = ""


def set_brb_tmp_directory(
        new_path: str
        ) -> None:
    global brb_tmp_directory
    brb_tmp_directory = new_path
    Path(brb_tmp_directory).mkdir(parents=True, exist_ok=True)


set_brb_tmp_directory("/tmp/brb")


def exit_handler():
    global brb_tmp_directory
    if Path(brb_tmp_directory).exists():
        shutil.rmtree(brb_tmp_directory, ignore_errors=True)


atexit.register(exit_handler)
