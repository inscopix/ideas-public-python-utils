import numpy as np
import pandas as pd
from beartype import beartype


@beartype
def _synthetic_two_gaussian(
    *,
    a_mean: float = 0.05,
    b_mean: float = 0.05,
    a_std: float = 0.1,
    b_std: float = 0.1,
    n_mice: int = 10,
    N: int = 100,
    mean_error: float = 0.05,
    random_seed: int = 1984,
) -> pd.DataFrame:
    """generate some synthetic data for testing the hierarchical
    bootstrap"""

    df = dict(
        mouse_id=[],
        value=[],
        treatment=[],
    )

    np.random.seed(random_seed)

    mouse_id = 0

    for _ in np.arange(n_mice):
        this_N = N + np.random.randint(N)

        mouse_id += 1

        a = np.random.normal(
            loc=a_mean + np.random.normal(scale=mean_error),
            scale=a_std,
            size=this_N,
        )
        df["mouse_id"].extend(np.ones(this_N) * mouse_id)
        df["value"].extend(a)
        df["treatment"].extend(np.repeat("Green", this_N))

        mouse_id += 1

        b = np.random.normal(
            loc=b_mean + np.random.normal(scale=mean_error),
            scale=b_std,
            size=this_N,
        )
        df["mouse_id"].extend(np.ones(this_N, dtype=int) * mouse_id)
        df["value"].extend(b)
        df["treatment"].extend(np.repeat("Blue", this_N))

    df = pd.DataFrame(df)

    df["mouse_id"] = df["mouse_id"].astype("category")
    df["treatment"] = df["treatment"].astype("category")

    return df
