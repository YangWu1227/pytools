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
from typing import List, Dict, Tuple, Union, Optional, NamedTuple

# ------------------------------- Intra-package ------------------------------ #

from pycitizen.exceptions import ColumnDtypeInferError, ColumnNameKeyWordError, ColumnNameStartWithDigitError, InvalidIdentifierError, InvalidColumnDtypeError
from pycitizen.utils import is_sequence, is_sequence_str

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
        bool_is_str = [pd.api.types.infer_dtype(
            value=df[col], skipna=True) == 'string' for col in df.columns.tolist()]
        cols = list(compress(df.columns.tolist(), bool_is_str))
    else:
        # If cols is a single string, then use the string to select column from df directly
        if isinstance(cols, str):
            bool_is_str = [pd.api.types.infer_dtype(
                value=df[cols], skipna=True) == 'string']
        else:
            # If cols a sequence of strings, then apply list comprehension
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
        bool_is_str = [pd.api.types.infer_dtype(
            value=df[cols], skipna=True) == 'string']
    else:
        # If a sequence of strings, then apply list comprehension
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

# ---------------------------------------------------------------------------- #
#                              Encoding functions                              #
# ---------------------------------------------------------------------------- #

# ----------------------- Function for ordinal encoding ---------------------- #


def likert_encode(df, mapping, return_class=False):
    """
    This function transforms specified columns using likert scale. Note that
    this function is defined to handle cases where some columns contain missing
    values. Use the helper function contain_null() to determine whether columns
    that are to be transformed contain any missing values.
    Note: The results of this transformation may be inaccurate if
    the columns contain misspellings and/or case inconsistencies. Use
    freq_tbl() to identify those issues. Then, use case_convert() and
    correct_misspell() to address those issues before proceeding with
    the transformation.

    Parameters
    ----------
    df : Dataframe
    mapping : A list of dictionaries: 
        - This must be a mapping of categories to labels for the encoding.
        - The dict contains a list of keys 'col' and values 'mapping'.
        - The value of each 'col' should be a feature name.
        - The value of each 'mapping' should be a dictionary of 'original_label' to 'encoded_label'.
        Example mapping: [
            {'col': 'col_1', 'mapping': {'a': 1, 'b': 2}},
            {'col': 'col_2', 'mapping': {'x': 1, 'y': 2}}
        ]
    return_class : bool, optional
        Whether to return an object of class EcodedObject, which then may be serialized, by default False

    Returns
    -------
    _type_
        _description_
    """
    # Create a copy
    df = df.copy()

    # Instantiate encoder
    encoder = ce_OrdinalEncoder(
        return_df=True,
        mapping=mapping,
        handle_unknown='return_nan'
    )
    # Fit and Transform
    transformed_df = encoder.fit_transform(X=df)
    # Select columns with float64 dtype
    float_cols = transformed_df.select_dtypes(include=np.float64)
    # Cast float64 to int64
    transformed_df[float_cols.columns] = float_cols.astype("Int64")
    # Map pd.NA to None
    transformed_df.replace(to_replace={pd.NA: None}, inplace=True)

    # Create a class
    class EncodedObject:
        def __init__(self, df, mapping):
            self.df = df
            self.mapping = mapping
    # Declare global variable
    global encoded_object
    # Instantiate an object of class EcodedObject
    encoded_object = EncodedObject(transformed_df, mapping)

    if return_class:
        return encoded_object
    else:
        return encoded_object.df

# ------------------------- One-hot or dummy encoding ------------------------ #


def onehot_encode(df, col):
    """
    Use this function to onehot (or dummy) transform categorical 
    features, producing one feature per category, each as a binary 
    indicator (0-1 allocation). Note: The results of this transformation will 
    be inaccurate if the columns contain misspellings and case inconsistencies. 
    Always use case_convert() and/or correct_misspell() to address those issues
    first and foremost. Post-transformation processing, such as reordering
    the columns, can be carried out using the helper function relocate().

    Parameters
    ----------
    df : DataFrame
    col : Sequence of str
        A sequence of column names to be transformed.

    Returns
    -------
    DataFrame
        The transformed DataFrame containing newly generated binary features.

    Raises
    ------
    TypeError
       The supplied columns must all be string columns.
    """
    # Create a copy
    df = df.copy()
    # Subset
    subset = df[col]
    # Check columns data types
    if (not all(subset.apply(lambda x: x.dtype == object))):
        raise TypeError("'col' must be a list of text columns")

    # Instantiate encoder
    encoder = ce_OneHotEncoder(
        return_df=True,
        use_cat_names=True,
        # This needs to be discussed
        handle_unknown='error'
    )
    # Fit and Transform
    transformed_df = encoder.fit_transform(X=subset)
    # Column-bind to original data frame
    new_transformed_df = pd.concat(
        objs=[df, transformed_df],
        # Columns
        axis=1,
        copy=False
    )

    return new_transformed_df

# ------------------------- String-to-string encoding ------------------------ #


def str_encode(df, mapping):
    """
    This function recodes categorical variables as standardized strings.
    For example, 'female' = F, 'high school graduate no college' = 'HS, no BA', etc.
    For encoding text variables that are ordinal in nature, use likert_case(), which 
    recodes text columns as numerical columns. This function is meant to complement
    likert_encode(), and is restricted to string-to-string encoding. 
    The user interface of this function is consistent with likert_encode() in that it
    has the same arguments. Thus, there is only one pattern to learn and users may opt
    to use one over the other depending on the data cleaning task at hand.

    Parameters
    ----------
    df : DataFrame
    mapping : A list of dictionaries:
    - This must be a mapping of categories to labels for the encoding.
    - The dict contains a list of of keys 'col' and values 'mapping'.
    - The value of each 'col' should be the feature name.
    - The value of each 'mapping' should be a dictionary of 'original_label' to 'encoded_label'.
    Example mapping: [
            {'col': 'col_1', 'mapping': {'a': 'str2', 'b': 'str2'}},
            {'col: 'col_2', 'mapping': {'x': 'str3', 'y': 'str4'}}
    ]

    Returns
    -------
    DataFrame
        A DataFrame with specified columns transformed as standardized strings.

    Raises
    ------
    TypeError
        Supplied columns must all be text fields.
    TypeError
        This function does NOT support encoding string to numerical.
    """
    # Create copy
    df = df.copy()

    # This step extracts column names into a list
    # The mapping argument is a list of dicts each structured as {'col': 'col_name', 'mapping':{}}
    # We do not want the keys ('col' and 'mapping'), but the values ('col_name', {})
    # The values can be accessed via dict_obj.values(), which returns objects of class 'dict_values'
    # The first element of each 'dict_values' is 'col_name', and the second is the mapping dict {}
    # For each 'dict_values', extract the first element [0]
    # This list comprehension then returns a list of 'col_names'
    col_names = [list(dict_obj.values())[0] for dict_obj in mapping]
    # Same for mapping dict {}, which is the second element of each 'dict_values'
    mapping_dicts = [list(dict_obj.values())[1] for dict_obj in mapping]

    # Enforce the rule that the mapping dicts must map 'string' to 'string'
    # This makes this function's goal restrictive, but it may prevent unanticipated bugs
    # Let likert_encode() handle string to integer mapping
    # Check column data types, if not all text columns, raise an error
    if (not all(df[col_names].apply(lambda x: x.dtype == object))):
        raise TypeError(
            "'col' in the 'mapping' dictionary must be a list of text columns"
        )

    # The comprehension below returns a list of sublists
    # Each sublist contains the values for each mapping dict in mapping_dicts
    list_of_sublists = [list(map_dict.values()) for map_dict in mapping_dicts]
    # Flatten the nested list above to get all values in one single list
    # The first part is (for sublist in list_of_sublists), which gets each sublist
    # Then, (value for value in sublist) gets the value from each sublist
    # These values are then stored in the outer-most single list []
    all_values = [value for sublist in list_of_sublists for value in sublist]
    # Test if all values are strings, if not, raise an error
    if (not all([isinstance(val, str) for val in all_values])):
        raise TypeError(
            "'mapping' must map string to string; use likert_encode() for mapping text columns to to integer columns"
        )

    # We can now use the map() method of pandas series to encode
    # Each iteration, we replace values in col_names[i] as specified by mapping_dicts[i]
    for i in range(len(mapping)):
        df[col_names[i]] = (
            df[col_names[i]].map(mapping_dicts[i], na_action='ignore')
        )

    return df
