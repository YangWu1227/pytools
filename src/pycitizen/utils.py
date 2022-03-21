# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #

from pandas.api.types import is_string_dtype
from pandas import Series

# ----------------------------- Standard library ----------------------------- #

from collections.abc import Sequence, ByteString
from typing import List

# ---------------------------------------------------------------------------- #
#                    Helper predicates for input validation                    #
# ---------------------------------------------------------------------------- #

# --------------------------------- Builtins --------------------------------- #


def is_sequence(seq: Sequence) -> bool:
    """
    This helper returns `True` if the input is a `collections.abc.Sequence` (except strings or bytestrings).

    Parameters
    ----------
    seq : Sequence of objects
        An input sequence to be tested.

    Returns
    -------
    bool
        `True` if the sequence is a `collections.abc.Sequence` and not a string or bytestring.
    """
    return isinstance(seq, Sequence) and not isinstance(seq, (str, ByteString, range))


def is_sequence_str(seq: Sequence) -> bool:
    """
    This helper returns `True` if the input is a `collections.abc.Sequence` (except bytestrings).

    Parameters
    ----------
    seq : Sequence of objects
        An input sequence to be tested.

    Returns
    -------
    bool
        `True` if the sequence is a `collections.abc.Sequence` and not a bytestring.
    """
    return isinstance(seq, Sequence) and not isinstance(seq, (ByteString, range))

# ----------------------------- Helper for pandas ---------------------------- #


def is_string(col: Series) -> bool:
    """
    This helper checks whether the provided Series is of the string dtype.

    Parameters
    ----------
    col : Series
        A pandas Series to be tested.

    Returns
    -------
    bool
        `True` if the Series is of the string dtype.
    """
    return is_string_dtype(col)

# -------------- Helpers for encoding functions input validation ------------- #


def is_encode_map(mapping: List[dict]) -> bool:

    # Check that the keys are correct
    keys = all((list(dict_.keys()) == ['col', 'mapping'] for dict_ in mapping))

    # Check that the values are correct
    vals = all((isinstance(list(dict_.values())[0], str) and isinstance(
        list(dict_.values())[1], dict) for dict_ in value_map_dict_str))

    return keys, vals
