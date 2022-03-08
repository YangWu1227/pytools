# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import datatable as dt
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

from pycitizen.exceptions import ColumnDtypeInferError, ColumnNameKeyWordError, ColumnNameStartWithDigitError, InvalidIdentifierError, InvalidColumnDtypeError
from pycitizen.predicates import is_sequence, is_sequence_str

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
    """
    # Create a copy if inplace=False
    if (not inplace):
        df = df.copy()

    original_col_nms = df.columns.tolist()
    # [^a-zA-Z0-9_] matches any character that is not a 'word character' (alphanumeric & underscore), which equivalent to \W
    new_col_nms = (sub('[^a-zA-Z0-9_]', '', col.lower())
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

# ----------------------- Function for case conversion ----------------------- #


def case_convert(df, cols=None, to='lower', inplace=False):
    """
    This helper function converts the cases in the columns of a DataFrame; the aim is to address case inconsistencies 
    in string columns. Use this function to ensure values in a DataFrame are consistent with regards to case, which helps 
    reduce the chance of committing further errors later on in the cleaning process.

    Parameters
    ----------
    df : DataFrame
    cols : str or Sequence of str, optional
        A single column name or a sequence of column names, by default None, which converts all columns that can be inferred as having 'string' dtypes.
    to : str, optional
        The direction or type of case conversion. One of 'lower', 'upper', 'title', or 'capitalize', by default 'lower'.
    inplace : bool, optional
        Whether to return a new DataFrame, by default False.

    Returns
    -------
    DataFrame
        A DataFrame with transformed columns or None if inplace=True.

    Raises
    ------
    TypeError
        The argument 'cols' must be registered as a Sequence or a single string.
    InvalidColumnDtypeError
        User supplied columns contain non-string columns.
    ValueError
        Direction or type of case conversion must either be 'lower', 'upper', 'title', or 'capitalize'.
    """
    if not is_sequence_str(cols) and cols is not None:
        raise TypeError(
            "'cols' must be a sequence like a list or tuple or a single string")
    # Selecting by column names df[cols] cannot use tuples
    if isinstance(cols, tuple):
        cols = list(cols)

    # Create a copy if inplace=False
    if (not inplace):
        df = df.copy()

    # If user does not specify columns, default to using all columns that are inferred as 'string'
    if cols == None:
        bool_is_str = [pd.api.types.infer_dtype(
            value=df[col], skipna=True) == 'string' for col in df.columns.tolist()]
        cols = list(compress(df.columns.tolist(), bool_is_str))
    else:
        # If cols is a single string, then create an interable object before list comprehension
        if isinstance(cols, str):
            bool_is_str = [pd.api.types.infer_dtype(
                value=df[col], skipna=True) == 'string' for col in (cols, )]
        else:
            # If a sequence of strings, then apply list comprehension
            bool_is_str = [pd.api.types.infer_dtype(
                value=df[col], skipna=True) == 'string' for col in cols]
        # If user supplies columns, check input column data types
        if not all(bool_is_str):
            raise InvalidColumnDtypeError(col_nms=list(
                compress(cols, [not element for element in bool_is_str])), dtype='string')

    # Use pd.DataFrame since, if user passes a single str as 'cols', df[cols] would be a series, and lambda 'x' would be the elements
    # Apply relies on the Series 'str' attribute
    if (to == 'upper'):
        df[cols] = pd.DataFrame(df[cols]).apply(lambda x: x.str.upper())
    elif (to == 'lower'):
        df[cols] = pd.DataFrame(df[cols]).apply(lambda x: x.str.lower())
    elif (to == 'title'):
        df[cols] = pd.DataFrame(df[cols]).apply(lambda x: x.str.title())
    elif (to == 'capitalize'):
        df[cols] = pd.DataFrame(df[cols]).apply(lambda x: x.str.capitalize())
    else:
        raise ValueError(
            "'to' must either by 'lower', 'upper', 'title', or 'capitalize'")

    # Return copy
    if (not inplace):
        return df
    else:
        return None

# ------------------- Function for correcting misspellings ------------------- #


def correct_misspell(df, cols, mapping, inplace=False):
    """
    This function corrects for potential spelling mistakes in the fields. 
    Users should first identify columns containing misspellings with the 
    help of the `freq_tbl()` function. Then, call this function to correct 
    for any misspellings by passing a dictionary.  

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing spelling errors.
    cols : str or Sequence of str
        A single column name or a sequence of column names, which must be inferred as having 'string' dtypes.
    mapping : dict
        A dictionary can be used to specify different replacement values for different existing values. For example, {'a': 'b', 'y': 'z'} replaces the value 'a' with 'b' and 'y' with 'z'.
    inplace : bool, optional
        Whether to return a new DataFrame, by default False.

    Returns
    -------
    DataFrame
        A DataFrame with transformed columns or None if inplace=True.

    Raises
    ------
    TypeError
        The argument 'cols' must be registered as a Sequence or a single string.
    InvalidColumnDtypeError
        User supplied columns contain non-string columns.
    """
    # Create a copy if inplace=False
    if (not inplace):
        df = df.copy()

    # Check input type
    if not is_sequence_str(cols):
        raise TypeError(
            "'cols' must be a sequence like a list or tuple or a single string")
    # Cast to list object
    if isinstance(cols, tuple):
        cols = list(cols)

    bool_is_str = [pd.api.types.infer_dtype(
        value=df[col], skipna=True) == 'string' for col in cols]
    # Check input column data types
    if not all(bool_is_str):
        raise InvalidColumnDtypeError(col_nms=list(
            compress(cols, [not element for element in bool_is_str])), dtype='string')

    # Replace values based on user supplied dictionary
    df[cols] = df[cols].replace(to_replace=mapping)

    # Return copy
    if (not inplace):
        return df
    else:
        return None
