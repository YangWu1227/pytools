# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #

# ----------------------------- Standard library ----------------------------- #

from typing import List

# ---------------------------------------------------------------------------- #
#                          AWS utils module exceptions                         #
# ---------------------------------------------------------------------------- #

# ------------------- Column dtype cannot be inferred error ------------------ #


class ColumnDtypeInferError(Exception):
    """
    Exception raised when a data frame contains column(s) with dtype(s) that cannot be inferred.
    """

    def __init__(self, col_nms: List[str]) -> None:
        self.col_nms = col_nms

    def __str__(self) -> str:
        return f'The dtypes of the following columns cannot be inferred: {self.col_nms}'

# ---------------------------------------------------------------------------- #
#                        Data cleaning module exceptions                       #
# ---------------------------------------------------------------------------- #

# ------------------------------- Keyword error ------------------------------ #


class ColumnNameKeyWordError(Exception):
    """
    Exception raised when column names are keywords.
    """

    def __init__(self, col_nms: List[str]) -> None:
        self.col_nms = col_nms

    def __str__(self) -> str:
        return f'Columns {self.col_nms} are keywords of the language, and cannot be used as ordinary identifiers'

# -------------------------- Start with digit error -------------------------- #


class ColumnNameStartWithDigitError(Exception):
    """
    Exception raised when column names start with digits.
    """

    def __init__(self, col_nms: List[str]) -> None:
        self.col_nms = col_nms

    def __str__(self) -> str:
        return f'Columns {self.col_nms} must not start with digits'

# ------------------------ Catch-all identifier error ------------------------ #


class InvalidIdentifierError(Exception):
    """
    Exception raised when column names are invalid identifiers.
    """

    def __init__(self, col_nms: List[str]) -> None:
        self.col_nms = col_nms

    def __str__(self) -> str:
        return f'Columns {self.col_nms} are invalid identifiers'

# ----------------- Columns do not have correct dtypes error ----------------- #


class InvalidColumnDtypeError(Exception):
    """
    Exception raised when user supplied columns do not have the dtypes that the function expects.
    """

    def __init__(self, col_nms: List[str], dtype: str) -> None:
        self.col_nms = col_nms
        self.dtype = dtype

    def __str__(self) -> str:
        return f"Columns {self.col_nms} are invalid as dtype '{self.dtype}' is expected"

# ------------------------ Invalid mapping input error ----------------------- #


class InvalidMappingKeys(Exception):
    """
    Exception raised when user supplied invalid mapping keys to encoding functions.
    """

    def __str__(self) -> str:
        return "The argument 'mapping' must be a list of dictionaries with 'col' and 'mapping' as the only two keys"


class InvalidMappingValues(Exception):
    """
    Exception raised when user supplied invalid mapping values to encoding functions.
    """

    def __str__(self) -> str:
        return "The argument 'mapping' must be a list of dictionaries with a string and a dictionary as the only two values"
