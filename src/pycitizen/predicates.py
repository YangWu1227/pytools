# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #

# ----------------------------- Standard library ----------------------------- #

from collections.abc import Sequence, ByteString

# ------------------- Helper functions for input validation ------------------ #


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
