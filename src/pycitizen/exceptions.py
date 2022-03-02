# ---------------------------------------------------------------------------- #
#                          AWS utils module exceptions                         #
# ---------------------------------------------------------------------------- #

# ------------------- Column dtype cannot be inferred error ------------------ #


class ColumnDtypeInferError(Exception):
    """
    Exception raised when a data frame contains column(s) with dtype(s) that cannot be inferred.
    """

    def __init__(self, col_nms):
        self.col_nms = col_nms

    def __str__(self):
        return f'The dtypes of the following columns cannot be inferred: {self.col_nms}'

# ---------------------------------------------------------------------------- #
#                        Data cleaning module exceptions                       #
# ---------------------------------------------------------------------------- #

# ------------------------------- Keyword error ------------------------------ #


class ColumnNameKeyWordError(Exception):
    """
    Exception raised when column names are keywords.
    """

    def __init__(self, col_nms):
        self.col_nms = col_nms

    def __str__(self):
        return f'Columns {self.col_nms} are keywords of the language, and cannot be used as ordinary identifiers'

# -------------------------- Start with digit error -------------------------- #


class ColumnNameStartWithDigitError(Exception):
    """
    Exception raised when column names start with digits.
    """

    def __init__(self, col_nms):
        self.col_nms = col_nms

    def __str__(self):
        return f'Columns {self.col_nms} must not start with digits'

# ------------------------ Catch-all identifier error ------------------------ #


class InvalidIdentifierError(Exception):
    """
    Exception raised when column names are invalid identifiers.
    """

    def __init__(self, col_nms):
        self.col_nms = col_nms

    def __str__(self):
        return f'Columns {self.col_nms} are invalid identifiers'
