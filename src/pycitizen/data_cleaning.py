# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #

import pandas as pd
import numpy as np

# ----------------------------- Standard library ----------------------------- #

import os
import sys
import pickle
from itertools import compress
from re import sub
import keyword
from collections import namedtuple
from typing import List, Dict, Tuple, Union, Optional, NamedTuple

# ------------------------------- Intra-package ------------------------------ #

from pycitizen.exceptions import (ColumnDtypeInferError,
                                  ColumnNameKeyWordError,
                                  ColumnNameStartWithDigitError,
                                  InvalidIdentifierError,
                                  InvalidColumnDtypeError,
                                  InvalidMappingKeys,
                                  InvalidMappingValues)
from pycitizen.utils import (is_sequence,
                             is_sequence_str,
                             is_list_str,
                             is_string,
                             is_encode_map)

# ---------------------------------------------------------------------------- #
#                               Cleaning helpers                               #
# ---------------------------------------------------------------------------- #

# ------------------ Function to check column name integrity ----------------- #


def check_col_nms(df: pd.DataFrame) -> None:
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


def clean_col_nms(df: pd.DataFrame, inplace: Optional[bool] = False) -> Union[pd.DataFrame, None]:
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
    # Remove trailing and leading white spaces
    new_col_nms = (col.strip() for col in original_col_nms)
    # Replace white spaces with "_"
    # \s+ matches 1 or more whitespace characters (spaces, tabs, line breaks)
    new_col_nms = (sub(r'\s+', '_', col) for col in new_col_nms)
    # [^a-zA-Z0-9_] matches any character that is not a 'word character' (alphanumeric & underscore), which is equivalent to \W
    new_col_nms = (sub('[^a-zA-Z0-9_]', '', col.lower())
                   for col in new_col_nms)
    # Remove leading characters until a letter is matched
    # ^ matches the beginning of the string
    # [^ ] is negated set, matching any character that is not in this set
    # + is a quantifier, matching the preceding element one or more times
    new_col_nms = (sub('^[^a-zA-Z]+', '', col) for col in new_col_nms)
    # Remove trailing characters until a letter is matched
    new_col_nms = [sub('[^a-zA-Z]+$', '', col) for col in new_col_nms]
    # Assign new columns
    df.columns = new_col_nms

    # Return copy
    if (not inplace):
        return df
    else:
        return None

# -------------- Function to create a tuple of frequency tables -------------- #


def freq_tbl(df: pd.DataFrame, dropna: Optional[bool] = False, **kwargs: str) -> NamedTuple:
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


def case_convert(df: pd.DataFrame,
                 cols: Optional[Union[List[str], Tuple[str], str]] = None,
                 to: Optional[str] = 'lower',
                 inplace: Optional[bool] = False) -> Union[pd.DataFrame, None]:
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
        bool_is_str = [is_string(df[col]) for col in df.columns.tolist()]
        cols = list(compress(df.columns.tolist(), bool_is_str))
    else:
        # If cols is a single string, then use the string to select column from df directly
        if isinstance(cols, str):
            bool_is_str = [is_string(df[cols])]
        else:
            # If cols is a sequence of strings, then apply list comprehension
            bool_is_str = [is_string(df[col]) for col in cols]
        # Take the 'bool_is_str' list from either one of the two branches above, check input column data types
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


def correct_misspell(df, cols, mapping, inplace=False) -> Union[pd.DataFrame, None]:
    """
    This function corrects for potential spelling mistakes in the fields. 
    Users should first identify columns containing misspellings with the 
    help of the `freq_tbl()` function. Then, call this function to correct 
    for any misspellings by passing a dictionary. For pattern matching with
    regular expressions, the `pd.DataFrame.replace()` method is still
    preferred and should offer more functionality. Note: the function does 
    not error if 'mapping' contains values that do not exist in 'df'.

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
        The argument 'mapping' must be dictionary object.
    TypeError
        The argument 'cols' must be registered as a Sequence or a single string.
    InvalidColumnDtypeError
        User supplied columns contain non-string columns.
    """
    if (not isinstance(mapping, dict)):
        raise TypeError("The argument 'mapping' must be a dictionary object")

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

    # If cols is a single string, then use the string to select column from df directly
    if isinstance(cols, str):
        bool_is_str = [is_string(df[cols])]
    else:
        # If a sequence of strings, then apply list comprehension
        bool_is_str = [is_string(df[col]) for col in cols]

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

# ------- Function for identifying columns that contain missing values ------- #


def find_missing(df: pd.DataFrame, axis: Optional[int] = 0) -> pd.Series:
    """
    This is a helper function that identifies columns or rows in a DataFrame
    that contain missing values.

    Parameters
    ----------
    df : DataFrame
    axis : int, optional
        Whether to identify rows or columns that contain missing values, by default 0 (columns).

    Returns
    -------
    Series of bool
        Boolean series indicating columns or rows with missing values.

    Raises
    ------
    TypeError
        The argument 'axis' must be an integer.
    ValueError
        The argument 'axis' must either be 1 (rows) or 0 (columns).
    """
    if not isinstance(axis, int):
        raise TypeError("The argument 'axis' must be an integer")

    # The lambda function simply returns the true elements of the boolean series
    if axis == 1:
        return df.isna().any(axis=1)[lambda x: x]
    elif axis == 0:
        return df.isna().any(axis=0)[lambda x: x]
    else:
        raise ValueError("'axis' must either be 1 (rows) or 0 (columns)")

# ---------------- Function to relocate columns in a DataFrame --------------- #


def relocate(df: pd.DataFrame,
             to_move: Union[List[str], str],
             before: Union[str, None] = None,
             after: Union[str, None] = None) -> pd.DataFrame:
    """
    This function reorders the columns in a DataFrame based on a reference column, which is either 
    specified as `before` or `after`. Only one of 'before' and 'after' should be supplied as a string.

    Parameters
    ----------
    df : pd.DataFrame
    to_move : str or list of str 
        A single or a list of column names to relocate.
    before : str or None, optional
        A reference column before which the columns in 'to_move' should be relocated, by default None.
    after : str or None, optional
        A reference column after which the columns in 'to_move' should be relocated, by default None.

    Returns
    -------
    DataFrame
        A DataFrame with reordered columns.

    Raises
    ------
    TypeError
        The argument 'to_move' must either be a list or a single string.
    TypeError
        Must supply only one of 'before' and 'after' as a string.
    """
    # Check input
    if not is_list_str(to_move):
        raise TypeError(
            "'to_move' must be a sequence like a list or a single string")
    # Exclusive or (exclusive disjunction where this true if and only if the two booleans differ (one is true, the other is false))
    if isinstance(before, str) == isinstance(after, str):
        raise TypeError(
            "must supply only one of 'before' and 'after' as a string")

    # If to_move is a string
    if isinstance(to_move, str):
        # Strings are immutable so this creates a new object that 'to_move' references
        to_move = [to_move]

    # If the reference column is included in 'to_move', remove it
    if before in to_move:
        to_move.remove(before)
    elif after in to_move:
        to_move.remove(after)

    # List of all column names
    cols = df.columns.tolist()

    # If we wish to move cols before the reference column
    if isinstance(before, str):
        # Select columns before the reference column (not including the reference column)
        seg1 = cols[:cols.index(before)]
        # Segment 2 includes columns to move 'before' the reference column
        to_move.append(before)
        seg2 = to_move

    # If we wish to move cols after the reference column
    if isinstance(after, str):
        # Select columns before the reference column (including the reference column)
        seg1 = cols[:cols.index(after) + 1]
        # Segment 2 simply includes columns to move
        seg2 = to_move

    # In either case, 'before' or 'after', we need to ensure columns to move are not in segment 1 (that is, drop 'to_move' from their original positions)
    # Segment 2 in either case should always include columns to move
    seg1 = [col for col in seg1 if col not in seg2]
    # Finally, select the rest of the columns--- those that are not in seg1 and seg2
    seg3 = [col for col in cols if col not in seg1 + seg2]

    # For 'before', seg2 includes columns to move plus the reference column
    # Thus, seg2 created from the 'before' branch ensures that columns to move will appear 'before' the reference column
    # For 'after', adding seg1 (with reference column) + seg2 (to_move) in this order ensures that columns to move will appear 'after' the reference column
    # Finally, seg3 adds the rest of the columns to the end
    return df[seg1 + seg2 + seg3]


# ---------------------------------------------------------------------------- #
#       Persistent storage and standard operating procedure for cleaning       #
# ---------------------------------------------------------------------------- #


class EncodeMap(object):
    """
    A class for mapping dictionary storage and cleaning standard operating procedure.
    """

    def __init__(self, mapping: List[dict], data: Optional[pd.DataFrame] = None) -> None:
        self.data = data
        self.mapping = mapping

    def to_csv(self, path: str) -> None:
        """
        This function writes the list of mappings as a csv file to disk.

        Parameters
        ----------
        path : str
            The path and file name for writing the list of mappings as a csv file.
        """
        mapping_df = pd.DataFrame.from_dict(self.mapping)
        mapping_df.rename(columns={
            "col": "Column Name", "mapping": "Description (any manipulations, recodes, etc)"}, inplace=True)
        mapping_df.to_csv(path, index=False)


# ---------------------------------------------------------------------------- #
#                              Encoding functions                              #
# ---------------------------------------------------------------------------- #

# ----------------------- Function for ordinal encoding ---------------------- #


def encode(df: pd.DataFrame, mapping: List[dict], mapping_path: Optional[str] = None, inplace: Optional[bool] = False) -> Union[pd.DataFrame, None]:
    """
    This function transforms specified columns using coding schemes for categorical variables. This may be a string-to-string rollup of the categories or 
    placing the categories on a likert scale, in which an integer vector is used to represent the categories in the columns. The results of this transformation 
    will be inaccurate if the columns contain misspellings or case inconsistencies in the categories. The `freq_tbl()` helper may be helpful for identifying 
    the presence of such issues. Then, `case_convert()` and `correct_misspell()` may be useful in addressing those issues. All these cleaning helpers are a part 
    of the recommended preprocessing steps, which precede this transformation.

    Parameters
    ----------
    df : DataFrame
    mapping : list of dict or dict
        This must be a mapping of categories to labels for the encoding.
        The dict contains a list of keys 'col' and values 'mapping'.
        The value of each 'col' should be a column name.
        The value of each 'mapping' should be a dictionary of 'original_label' to 'encoded_label'.
        example mapping [{'col': 'col1', 'mapping': {'a': 1, 'b': 2}}, {'col': 'col2', 'mapping': {'x': 1, 'y': 2}}]
    mapping_path : str, optional
        File path to write the list of mappings as a csv file, by default None, which does not write the mappings to disk.
    inplace : bool, optional
        Whether to return a new DataFrame, by default False.

    Returns
    -------
    DataFrame
        A DataFrame with transformed columns or None if inplace=True.
    """
    # If passed a single dict, convert to list
    if isinstance(mapping, dict):
        mapping = [mapping]

    # Check mapping
    if not is_encode_map(mapping=mapping)[0]:
        raise InvalidMappingKeys
    elif not is_encode_map(mapping=mapping)[1]:
        raise InvalidMappingValues

    # Create a copy if inplace=False
    if (not inplace):
        df = df.copy()

    for col_mapping_pair in mapping:
        # May switch to get() to not raise an error if the keys do not exist
        col = col_mapping_pair['col']
        col_mapping = col_mapping_pair['mapping']

        # If None exists, then convert it to np.NaN in prepartion for 'na_action' in map()
        df[col] = pd.Series(
            [obs if obs is not None else np.NaN for obs in df[col]], index=df[col].index)

        # Use 'na_action' to not pass np.NaN to the mapping correspondence
        df[col] = df[col].map(col_mapping, na_action='ignore')

        # Integer column with missing values is cast to floating-point dtype
        # Try converting to nullable integer array if possible
        # Use errors='ignore' to return original object on error
        df[col] = df[col].astype('Int64', errors='ignore')

    if (mapping_path is not None):
        EncodeMap(mapping=mapping).to_csv(mapping_path)

    # Return copy
    if (not inplace):
        return df
    else:
        return None

# ------------------------- One-hot or dummy encoding ------------------------ #


def onehot_encode(df: pd.DataFrame, cols: Union[List[str], Tuple[str], str]) -> pd.DataFrame:
    """
    Use this function to onehot (or dummy) transform categorical 
    features, producing one feature per category, each as a binary 
    indicator (0-1 allocation). Note: The results of this transformation will 
    be inaccurate if the columns contain misspellings and case inconsistencies. 
    The `freq_tbl()` helper may be helpful for identifying the presence of 
    such issues. Then, `case_convert()` and `correct_misspell()` may be 
    useful in addressing those issues.

    Parameters
    ----------
    df : DataFrame
    col : Sequence of str or str
        A single or sequence of column names to be transformed.

    Returns
    -------
    DataFrame
        The transformed DataFrame containing newly generated binary features.

    Raises
    ------
    TypeError
        The argument 'df' must be a DataFrame.
    TypeError
        The argument 'cols' must either be registered as a Sequence or a str object.
    """
    # Be more restrictive with typing--- only act on DataFrames
    if not (isinstance(df, pd.DataFrame)):
        raise TypeError("'df' must be be a DataFrame")
    # Check input
    if not (is_sequence_str(cols)):
        raise TypeError(
            "'cols' must be a sequence like a list or tuple or a single string")

    # Subset
    subset = df[cols]
    # Create prefixes
    if isinstance(cols, str):
        prefix = cols
    else:
        prefix = subset.columns.tolist()

    # Transform
    transformed_df = pd.get_dummies(
        data=subset,
        prefix=prefix,
        dummy_na=False,
        dtype=np.uint8
    )

    # Column-bind to original data frame
    cbind_df = pd.concat(
        objs=[df, transformed_df],
        # Columns
        axis=1
    )

    return cbind_df
