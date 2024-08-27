import warnings

import numpy as np
import pandas as pd
import scipy
from beartype import beartype

from ideas.types import NumpyVector


def _quiet_difference(x, y, axis) -> float:
    """difference in means between two things

    small helper functions that finds the difference
    in means between two np arrays and never raises
    a runtime warning


    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        return np.nanmean(x, axis=axis) - np.nanmean(y, axis=axis)


@beartype
def permutation_test(
    x: NumpyVector,
    y: NumpyVector,
    permutation_type: str = "independent",
    random_state=1984,
) -> float:
    """performs a one-tailed permutation test to check if means of
    x and y are different

    If x and y are the same length, and you want to perform a paired
    permutation test, use permutation_type = "samples"

    For details on how to use this, and possible arguments, see
    the docs for the underlying scipy function:

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
    """

    alternative = "less"
    if np.nanmean(x) > np.nanmean(y):
        alternative = "greater"

    res = scipy.stats.permutation_test(
        (x, y),
        _quiet_difference,
        vectorized=True,
        alternative=alternative,
        permutation_type=permutation_type,
        random_state=random_state,
    )

    pvalue = res.pvalue
    return pvalue


def _df_to_nan_padded_array(
    df: pd.DataFrame,
    subject_column: str,
    value_column: str,
) -> (np.array, np.array):
    """helper function to convert a (ragged) dataframe to np.array

    Do not use this function. This is meant to be called by
    hierarchical_bootstrap

    ### Arguments:

    - df: ragged dataframe
    - subject_column: column that identifies the subject
    - value_column: column that identifies the actual value/data

    """

    max_columns = df.groupby(subject_column)[value_column].count().max()
    subjects = df[subject_column].unique()
    n_subjects = len(subjects)

    x = np.full((n_subjects, max_columns), np.nan)
    n_samples = np.full(n_subjects, 0).astype(int)

    for i in np.arange(n_subjects):
        y = df[df[subject_column] == subjects[i]][value_column]
        x[i, : len(y)] = y
        n_samples[i] = len(y)

    return x, n_samples


@beartype
def hierarchical_bootstrap(
    df: pd.DataFrame,
    return_all_samples: bool = False,
    *,
    condition_column: str,
    subject_column: str,
    value_column: str,
    n_bootstrap: int = 1000,
    summary_func=np.nanmean,
) -> dict:
    """performs a hierarchical bootstrap of data to compare two conditions

    This is similar to the original hierarchical_bootstrap
    but uses numpy arrays to
    speed up computation. Experimental.

    performs a bootstrap of hierarchical data as described in [Saravanan,
    Berman & Sober](https://pubmed.ncbi.nlm.nih.gov/33644783/)

    ### Assumptions:

    1. data is hierarchical, with data split into two conditions
    2. each condition has several subjects (animals)
    3. each animal has multiple neurons recorded


    ### Arguments:

    - df: pandas data frame with data
    - return_all_samples: indicate if should additionally return a dict
    with all boostrap sample values; default is False
    - condition_column: name of column that identifies condition.
    - subject_column: name of column that identifies subject (animal)
    - value_column: Name of column that identifies value to compare
    - n_bootstrap: How many times to bootstrap?
    - summary_func: function to summarize statistics f(vector) -> scalar

    ### Returns

    Dictionary with bootstrapped values
    Dictionary with entire bootstrapped samples (optional, if
    return_all_samples == True)


    """
    # check for nans in value_column
    if df[value_column].isna().sum() > 0:
        raise ValueError(f"Remove rows from df with nans in {value_column}.")

    # check if return_all_samples is bool
    if not isinstance(return_all_samples, bool):
        raise ValueError("return_all_samples should be of Boolean type")

    # get unique conditions
    conditions = df[condition_column].unique()

    # figure out how many subjects
    subjects = dict()
    for condition in conditions:
        subjects[condition] = df[df[condition_column] == condition][
            subject_column
        ].unique()

    # preallocate outputs
    results = dict()
    for condition in conditions:
        results[condition] = np.full((n_bootstrap), np.nan)

    # if requested, return all results
    if return_all_samples:
        all_samples = dict()
        for condition in conditions:
            all_samples[condition] = [[]] * n_bootstrap

    for condition in conditions:
        this_df = df[df[condition_column] == condition]

        x, n_samples = _df_to_nan_padded_array(
            this_df,
            subject_column=subject_column,
            value_column=value_column,
        )

        for i in np.arange(n_bootstrap):
            resampled_subjects = np.random.randint(
                0, x.shape[0], size=x.shape[0]
            )
            resampled_x = np.full_like(x, np.nan)

            for j, _ in enumerate(resampled_subjects):
                subject = resampled_subjects[j]

                # turns out this is EXTREMELY slow
                # leaving this here as a warning to those who
                # follow to never use this pattern
                # n_samples = sum(~np.isnan(x[subject, :]))

                resampled_observations = np.random.randint(
                    0, n_samples[j], size=n_samples[j]
                )

                resampled_x[j, : n_samples[j]] = x[
                    subject, resampled_observations
                ]

            # run the summary function on this
            resampled_x_flat = resampled_x.flatten()
            results[condition][i] = summary_func(resampled_x_flat)

            # if requested, return all samples
            if return_all_samples:
                all_samples[condition][i] = [
                    s for s in resampled_x_flat if ~np.isnan(s)
                ]

    if return_all_samples:
        results["all_samples"] = all_samples

    return results
