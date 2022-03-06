# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import pytest

# ----------------------------- Standard library ----------------------------- #

import os
from re import escape
from collections import namedtuple

# ------------------------------- Intra-package ------------------------------ #

import pycitizen.data_cleaning as dc
from pycitizen.exceptions import ColumnNameKeyWordError, ColumnNameStartWithDigitError, InvalidIdentifierError


# ---------------------------------------------------------------------------- #
#                           Test data for the module                           #
# ---------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def test_data():
    return pd.DataFrame({
        'likert_encode': ('A', 'A', 'B', 'C', 'D', 'D', 'C', pd.NA, 'C', 'E'),
        'str_encode': ('bachelor', 'highschool', pd.NA, 'grad', 'grad', pd.NA, 'highschool', 'college', 'college', 'bachelor'),
        'onhot_encode': ('A',) * 5 + ('B',) * 5,
        'case_convert': ('Upper',) * 5 + ('lower',) * 3 + (pd.NA, pd.NA),
        'misspell': ('republican',) * 4 + ('repulican',) + ('democrat',) * 3 + ('democract',) * 2
    })

# ---------------------------------------------------------------------------- #
#                        Tests for column names helpers                        #
# ---------------------------------------------------------------------------- #

# -------------------- Test data for column names helpers -------------------- #


@pytest.fixture(scope='class')
def check_col_nms_test_df():
    return (
        # Non-dataframe
        pd.Series((1, 2)),
        # Start with digit
        pd.DataFrame({'123col': (1, 2)}),
        # Keyword
        pd.DataFrame({'def': (1, 2)}),
        # Special characters
        pd.DataFrame({'^3edf': (1, 2)}),
        pd.DataFrame({'price ($)': (1, 2)}),
        pd.DataFrame({'percent%': (1, 2)})
    )


class TestColumnNmsHelpers:
    """
    Check the exceptions raised when column names helper functions fail.
    """

    def test_check_col_nms(self, check_col_nms_test_df):
        """
        Check that check_col_nms() raises exceptions when data frames contain invalid identifiers.
        """
        # Non-dataframe
        with pytest.raises(TypeError, match="'df' must be a DataFrame"):
            dc.check_col_nms(check_col_nms_test_df[0])
        # Start with digit
        with pytest.raises(ColumnNameStartWithDigitError, match=escape("Columns ['123col'] must not start with digits")):
            dc.check_col_nms(check_col_nms_test_df[1])
        # Keyword
        with pytest.raises(ColumnNameKeyWordError, match=escape("Columns ['def'] are keywords of the language, and cannot be used as ordinary identifiers")):
            dc.check_col_nms(check_col_nms_test_df[2])
        # Special characters
        with pytest.raises(InvalidIdentifierError, match=escape("Columns ['^3edf'] are invalid identifiers")):
            dc.check_col_nms(check_col_nms_test_df[3])
        with pytest.raises(InvalidIdentifierError, match=escape("Columns ['price ($)'] are invalid identifiers")):
            dc.check_col_nms(check_col_nms_test_df[4])
        with pytest.raises(InvalidIdentifierError, match=escape("Columns ['percent%'] are invalid identifiers")):
            dc.check_col_nms(check_col_nms_test_df[5])

    def test_clean_col_nms(self, check_col_nms_test_df):
        """
        Check that clean_col_nms() fixes invalid identifiers.
        """
        # Digits '123col' are first replace with "_" and then "_" are removed until letter 'c' is matched
        assert dc.clean_col_nms(
            check_col_nms_test_df[1]).columns.tolist() == ['col']
        # Digit and special character are removed from '^3edf'
        assert dc.clean_col_nms(
            check_col_nms_test_df[3]).columns.tolist() == ['edf']
        # White spaces and special characters are removed from 'price ($)' and 'percent%'
        assert dc.clean_col_nms(
            check_col_nms_test_df[4]).columns.tolist() == ['price']
        assert dc.clean_col_nms(
            check_col_nms_test_df[5]).columns.tolist() == ['percent']

# ---------------------------------------------------------------------------- #
#                          Tests for freq_tbl function                         #
# ---------------------------------------------------------------------------- #


class TestFreqTable:
    """
    Tests for the freq_tbl helper function.
    """

    # ------------------------ Tests for exceptions raised ----------------------- #

    def test_freq_tbl_errors(self):
        """
        Check that freq_tbl() raises exceptions when function fails (non **kwargs).
        """
        with pytest.raises(AttributeError, match="object has no attribute 'select_dtypes'"):
            dc.freq_tbl(pd.Series(('A', 'B')), False)
        with pytest.raises(TypeError, match='boolean value of NA is ambiguous'):
            dc.freq_tbl(pd.DataFrame({'col': ("A", "B", "A")}), pd.NA)
        with pytest.raises(ValueError):
            dc.freq_tbl(pd.DataFrame(
                {'col': ("A", "B", "A")}), pd.Series(('A', 'B')))

    @pytest.mark.parametrize(
        "df, dropna",
        [(pd.DataFrame({'col': ("A", "B", "A")}), False)],
        scope='function'
    )
    def test_freq_tbl_kwargs(self, df, dropna):
        """
        Test that exception is raised when invalid **kwargs are provided to freq_tbl().
        """
        with pytest.raises(ValueError, match="Only 'normalize', 'sort', and 'ascending' are supported as extra keyword arguments"):
            dc.freq_tbl(df, dropna, wrong_keyword=True)
        with pytest.raises(ValueError, match="Only 'normalize', 'sort', and 'ascending' are supported as extra keyword arguments"):
            dc.freq_tbl(df, dropna, subset=True)

    # -------------------------- Tests for functionality ------------------------- #

    @pytest.mark.parametrize(
        "sort, normalize",
        [
            (True, False),
            (False, True)
        ],
        scope='function'
    )
    def test_freq_tbl(self, test_data, sort, normalize):
        """
        Test that freq_tbl() returns the correct class and length given a test dataframe.
        """

        # Tuple
        assert isinstance(dc.freq_tbl(
            test_data, sort=sort, normalize=normalize), tuple) == True
        # Check '_fileds' attributes match test data columns names
        assert dc.freq_tbl(
            test_data, sort=sort, normalize=normalize)._fields == tuple(test_data.columns)
        # Check length
        assert len(dc.freq_tbl(test_data, sort=sort, normalize=normalize)) == 5
