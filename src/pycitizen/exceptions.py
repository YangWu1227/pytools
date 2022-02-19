# ---------------------------------------------------------------------------- #
#                                  Base error                                  #
# ---------------------------------------------------------------------------- #

class Error(Exception):
    """
    Base class for exceptions in this module.
    """
    pass

# ---------------------------------------------------------------------------- #
#                          AWS utils module exceptions                         #
# ---------------------------------------------------------------------------- #

# ------------------- Column dtype cannot be inferred error ------------------ #


class ColumnDtypeInferError(Error):
    """
    Exception raised when a data frame contains column(s) with dtype(s) that cannot be inferred.
    """
    pass
