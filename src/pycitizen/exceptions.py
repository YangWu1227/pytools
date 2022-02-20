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
