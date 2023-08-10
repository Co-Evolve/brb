import fprs
import numpy as np

seed = 42
brb_random_state = np.random.RandomState(seed=42)


def set_random_state(
        seed_value: int
        ) -> None:
    global seed, brb_random_state
    seed = seed_value
    brb_random_state.seed(seed_value)
    fprs.set_random_state(seed)
