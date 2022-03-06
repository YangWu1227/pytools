# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #

import pandas as pd
import datatable as dt
import numpy as np
from category_encoders import OrdinalEncoder as ce_OrdinalEncoder
from category_encoders import OneHotEncoder as ce_OneHotEncoder

# ----------------------------- Standard library ----------------------------- #

import os
import sys
import pickle
from itertools import compress
from re import sub
import keyword
from collections import namedtuple

# ------------------------------- Intra-package ------------------------------ #

from pycitizen.exceptions import ColumnNameKeyWordError, ColumnNameStartWithDigitError, InvalidIdentifierError

# ---------------------------------------------------------------------------- #
#                               Cleaning helpers                               #
# ---------------------------------------------------------------------------- #

# ------------------ Function to check column name integrity ----------------- #


def check_col_nms(df):
    """
    This function is a helper that checks the integrity of the column names. 

    Parameters
    ----------
    df : DataFrame

    Raises
    ------
    TypeError
        The argument 'df' must be a DataFrame.
    ColumnNameKeyWordError
        Columns name cannot contain reserved keywords like 'def', 'for'.
    ColumnNameStartWithDigitError
        Column names cannot begin with a digit.
    InvalidIdentifierError
        Catch-all exception for invalid identifiers.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("'df' must be a DataFrame")

    col_nms = df.columns.tolist()

    # Check column names are not keywords
    kw_index = [keyword.iskeyword(col) for col in col_nms]
    if any(kw_index):
        raise ColumnNameKeyWordError(list(compress(col_nms, kw_index)))

    # Check column names do not start with digit
    digit_index = [col[0].isdigit() for col in col_nms]
    if any(digit_index):
        raise ColumnNameStartWithDigitError(
            list(compress(col_nms, digit_index)))

    # Catch-all check
    invalid_identifier_index = [not col.isidentifier() for col in col_nms]
    if any(invalid_identifier_index):
        raise InvalidIdentifierError(
            list(compress(col_nms, invalid_identifier_index)))

# --------------------- Function that cleans column names -------------------- #


def clean_col_nms(df, inplace=False):
    """
    This helper function removes any invalid character, e.g. special characters and white spaces, in a column name and removes 
    leading characters until a character from a-z or A-Z is matched. Note this function does not replace python keywords or reserved 
    words if they exist as column names. Use the `rename()` method of pandas DataFrame or the datatable rename `{"A": "col_A"}` syntax 
    to clean the column names if `check_col_nms()` reveals such invalid columns names. 

    Parameters
    ----------
    df : Dataframe
    inplace : bool, optional
        Whether to return a new DataFrame, by default False.

    Returns
    -------
    DataFrame
        A DataFrame with transformed column names or None if inplace=True.

    Raises
    ------
    TypeError
        The argument 'df' must be a DataFrame.
    TypeError
        The argument 'inplace' must be a boolean.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("'df' must be a DataFrame")
    if not isinstance(inplace, bool):
        raise TypeError("'inplace' must be a single boolean")

    # Create a copy if inplace=False
    if (not inplace):
        df = df.copy()

    original_col_nms = df.columns.tolist()
    # \W matches any character that is not a 'word character' (alphanumeric & underscore). Equivalent to [^A-Za-z0-9_]
    new_col_nms = (sub('\W', '', col.lower())
                   for col in original_col_nms)
    # Remove leading characters until a letter
    # [^ ] is negated set, matching any character that is not in this set
    # + is a quantifier, matching the preceding element one or more times
    new_col_nms = [sub('^[^a-zA-Z]+', '', col) for col in new_col_nms]
    # Assign new columns
    df.columns = new_col_nms

    # Return copy
    if (not inplace):
        return df
    else:
        return None

# -------------- Function to create a tuple of frequency tables -------------- #


def freq_tbl(df, dropna=False, **kwargs):
    """
    This function creates a sequence of freqency tables of the text fields in a DataFrame,
    which can be examined to identify misspellings and case inconsistencies. You may pass 
    extra keyword arguments for the underlying pandas function. See `?pandas.DataFrame.value_counts`
    for options (note that the `subset` argument is not permitted). 

    Parameters
    ----------
    df : DataFrame
    dropna : bool, optional
        Whether to drop missing values, by default False.

    Returns
    -------
    tuple of DataFrame
        A namedtuple of frequency tables containing counts (or proportions if extra keyword arguments are specified) per category for each text field.

    Raises
    ------
    ValueError
        Only 'normalize', 'sort', 'ascending' are supported as extra keyword arguments.
    """
    # Check keyword args
    if not all((kwarg in ('normalize', 'sort', 'ascending') for kwarg in kwargs.keys())):
        raise ValueError(
            "Only 'normalize', 'sort', and 'ascending' are supported as extra keyword arguments")
    # Keep only text columns (including mixed type with 'object' dtype)
    df = df.select_dtypes(exclude=np.number)
    # Generator of frequency tables
    gen_of_freq = (pd.DataFrame(df[col].value_counts(dropna=dropna, **kwargs))
                   for col in df.columns)
    # Create subclass constructor
    freq = namedtuple('freq', df.columns)
    # Use constructor to create a named tuple
    freq_tbl = freq._make(gen_of_freq)

    return freq_tbl
