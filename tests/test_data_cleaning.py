# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #

from io import StringIO
from turtle import right
import pandas as pd
import numpy as np
import pytest

# ----------------------------- Standard library ----------------------------- #

import os
from io import StringIO
from re import escape
from collections import namedtuple

# ------------------------------- Intra-package ------------------------------ #

import pycitizen.data_cleaning as dc
from pycitizen.exceptions import (ColumnDtypeInferError,
                                  ColumnNameKeyWordError,
                                  ColumnNameStartWithDigitError,
                                  InvalidIdentifierError,
                                  InvalidColumnDtypeError,
                                  InvalidMappingKeys,
                                  InvalidMappingValues)


# ---------------------------------------------------------------------------- #
#                           Test data for the module                           #
# ---------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def test_data():
    return pd.DataFrame({
        'likert_encode': ('A', 'A', 'B', 'C', 'D', 'D', 'C', pd.NA, 'C', 'E'),
        'str_encode': ('bachelor', 'highschool', pd.NA, 'grad', 'grad', pd.NA, 'highschool', 'college', 'college', 'bachelor'),
        'onehot_encode': ('A',) * 5 + ('B',) * 5,
        'case_convert': ('Upper',) * 5 + ('lower',) * 3 + (pd.NA, pd.NA),
        'misspell': ('republican',) * 4 + ('repulican',) + ('democrat',) * 3 + ('democract',) * 2,
        'invalid_case_convert': tuple(range(0, 10))
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
        pd.DataFrame({'percent%': (1, 2)}),
        # Trailing and leading White spaces
        pd.DataFrame({' trailing   leading    ': (1, 2)})
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
        with pytest.raises(InvalidIdentifierError, match=escape("Columns [' trailing   leading    '] are invalid identifiers")):
            dc.check_col_nms(check_col_nms_test_df[6])

    def test_clean_col_nms(self, check_col_nms_test_df):
        """
        Check that clean_col_nms() fixes invalid identifiers.
        """
        # Digits '123col' become 'col' since leading characters are moved until a letter is matched
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
        # Leading and trailing white spaces are removed first, then while spaces in between are replace with '_'
        assert dc.clean_col_nms(
            check_col_nms_test_df[6]).columns.tolist() == ['trailing_leading']


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

        # Unique error messages due to polymorphism
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
            (False, True),
            (True, True),
            (False, False)
        ],
        scope='function'
    )
    def test_freq_tbl(self, test_data, sort, normalize):
        """
        Test that freq_tbl() returns the correct class and length given a test dataframe.
        """

        # Outputs
        tbls = dc.freq_tbl(test_data, dropna=True,
                           sort=sort, normalize=normalize)
        tbls_true_false = dc.freq_tbl(test_data, dropna=True,
                                      sort=True, normalize=False)
        tbls_false_true = dc.freq_tbl(test_data, dropna=True,
                                      sort=False, normalize=True)

        # Expected for two branches
        expected_index = {
            # Sort=True
            'sort_true': (
                ['C', 'A', 'D', 'B', 'E'],
                ['bachelor', 'highschool', 'grad', 'college'],
                ['A', 'B'],
                ['Upper', 'lower'],
                ['republican', 'democrat', 'democract', 'repulican']
            ),
            # Sort=False
            'sort_false': (
                ['A', 'B', 'C', 'D', 'E'],
                ['bachelor', 'highschool', 'grad', 'college'],
                ['A', 'B'],
                ['Upper', 'lower'],
                ['republican', 'repulican', 'democrat', 'democract']
            )
        }
        expected_values = {
            # Normalize=False
            'normalize_false': [
                np.array([[3], [2], [2], [1], [1]]),
                np.array([[2], [2], [2], [2]]),
                np.array([[5], [5]]),
                np.array([[5], [3]]),
                np.array([[4], [3], [2], [1]])
            ],
            # Normalize=True
            'normalize_true': [
                np.array([[0.2], [0.1], [0.3], [0.2], [0.1]]),
                np.array([[0.2], [0.2], [0.2], [0.2]]),
                np.array([[0.5], [0.5]]),
                np.array([[0.6], [0.4]]),
                np.array([[0.4], [0.1], [0.3], [0.2]])
            ]
        }

        # Tuple
        assert isinstance(tbls, tuple) == True
        # Check '_fileds' attributes match test data string columns names
        assert tbls._fields == (
            'likert_encode', 'str_encode', 'onehot_encode', 'case_convert', 'misspell')
        # Check length
        assert len(tbls) == 5

        # Branch (sort=True and normalize=False)
        for num, tbl in enumerate(tbls_true_false):
            assert all(tbl.index == expected_index['sort_true'][num])
            assert all(tbl.values ==
                       expected_values['normalize_false'][num])

        # Branch (sort=False and normalize=True)
        for num, tbl in enumerate(tbls_false_true):
            assert all(tbl.index == expected_index['sort_false'][num])
            assert all(tbl.round(1).values ==
                       expected_values['normalize_true'][num])

# ---------------------------------------------------------------------------- #
#                        Tests for case_convert function                       #
# ---------------------------------------------------------------------------- #


class TestCaseConvert:
    """
    Tests for the case_convert helper function.
    """

    # --------------------- Tests that exceptions are raised --------------------- #

    def test_case_convert_errors(self, test_data):
        """
        Tests that case_convert raises exceptions when 'df', 'cols' and 'to' are passed invalid inputs.
        """

        # Invalid input for 'df', which creates unique error messages due to polymorphism
        with pytest.raises(AttributeError, match="no attribute 'columns'"):
            dc.case_convert(df=pd.Series(('Upper', 'lower', 'case')))
        with pytest.raises(AttributeError, match="no attribute 'columns'"):
            dc.case_convert(df=['Upper', 'Word'])
        with pytest.raises(AttributeError, match="no attribute 'copy'"):
            dc.case_convert(df=('str', ))

        # Range offset for 'cols'
        with pytest.raises(TypeError, match="'cols' must be a sequence like a list or tuple or a single string"):
            dc.case_convert(df=test_data, cols=range(0, 3))

        # Invalid inputs for 'to'
        # Numeric
        with pytest.raises(ValueError, match="'to' must either by 'lower', 'upper', 'title', or 'capitalize'"):
            dc.case_convert(df=test_data, cols='case_convert', to=3)
        # Boolean
        with pytest.raises(ValueError, match="'to' must either by 'lower', 'upper', 'title', or 'capitalize'"):
            dc.case_convert(df=test_data, cols='case_convert', to=True)
        # List
        with pytest.raises(ValueError, match="'to' must either by 'lower', 'upper', 'title', or 'capitalize'"):
            dc.case_convert(df=test_data, cols='case_convert',
                            to=['lower', 'upper'])
        # Single element tuple
        with pytest.raises(ValueError, match="'to' must either by 'lower', 'upper', 'title', or 'capitalize'"):
            dc.case_convert(
                df=test_data, cols='case_convert', to=('lower', ))

    # ------------------- Tests that custom exception is raised ------------------ #

    def test_case_convert_custom_error(self, test_data):
        """
        Test that when user passes non 'string' columns in 'cols' the function raises InvalidColumnDtypeError.
        """

        with pytest.raises(InvalidColumnDtypeError, match=escape("Columns ['invalid_case_convert'] are invalid as dtype 'string' is expected")):
            dc.case_convert(
                test_data, ['invalid_case_convert', 'case_convert'])

    # -------------------------- Tests for functionality ------------------------- #

    @pytest.mark.parametrize(
        "cols, to",
        [
            # No user supplied columns
            (None, 'lower'),
            (None, 'upper'),
            (None, 'title'),
            (None, 'capitalize'),
            # User supplied columns
            ('case_convert', 'lower'),
            (('case_convert', 'str_encode'), 'upper'),
            (['misspell', 'case_convert'], 'title'),
            ('misspell', 'capitalize')
        ],
        scope='function'
    )
    def test_case_convert(self, test_data, cols, to):
        """
        Test that case_convert returns expected results given inputs with branches.
        """

        # Test branches
        type(dc.case_convert(test_data, cols=cols, to=to)) == type(pd.DataFrame())

        # Test two branches
        branch1 = dc.case_convert(test_data, cols=None, to='lower')
        branch2 = dc.case_convert(test_data, cols=(
            'case_convert', 'str_encode'), to='upper')

        # Branch 1
        pd.testing.assert_frame_equal(
            left=branch1,
            right=pd.DataFrame({
                'likert_encode': ('a', 'a', 'b', 'c', 'd', 'd', 'c', pd.NA, 'c', 'e'),
                'str_encode': ('bachelor', 'highschool', pd.NA, 'grad', 'grad', pd.NA, 'highschool', 'college', 'college', 'bachelor'),
                'onehot_encode': ('a',) * 5 + ('b',) * 5,
                'case_convert': ('upper',) * 5 + ('lower',) * 3 + (pd.NA, pd.NA),
                'misspell': ('republican',) * 4 + ('repulican',) + ('democrat',) * 3 + ('democract',) * 2,
                'invalid_case_convert': tuple(range(0, 10))
            })
        )

        # Branch 2
        pd.testing.assert_frame_equal(
            left=branch2,
            right=pd.DataFrame({
                'likert_encode': ('A', 'A', 'B', 'C', 'D', 'D', 'C', pd.NA, 'C', 'E'),
                'str_encode': ('BACHELOR', 'HIGHSCHOOL', pd.NA, 'GRAD', 'GRAD', pd.NA, 'HIGHSCHOOL', 'COLLEGE', 'COLLEGE', 'BACHELOR'),
                'onehot_encode': ('A',) * 5 + ('B',) * 5,
                'case_convert': ('UPPER',) * 5 + ('LOWER',) * 3 + (pd.NA, pd.NA),
                'misspell': ('republican',) * 4 + ('repulican',) + ('democrat',) * 3 + ('democract',) * 2,
                'invalid_case_convert': tuple(range(0, 10))
            })
        )


# ---------------------------------------------------------------------------- #
#                      Tests for correct_misspell function                     #
# ---------------------------------------------------------------------------- #


class TestMisspell:
    """
    Tests for the correct_misspell helper function.
    """

    # ------------------------ Tests for exceptions raised ----------------------- #

    def test_correct_misspell_errors(self, test_data):
        """
        Tests that correct_misspell raises exceptions when 'df', 'cols' and 'mapping' are passed invalid inputs.
        """

        # Invalid input for 'df', which creates unique error messages due to polymorphism, but usually AttributeError or KeyError
        # First place that may raise an exception is the '.copy' method
        # The second place is keyword when subsetting df[cols] or df[col]
        with pytest.raises(AttributeError):
            dc.correct_misspell((1, 2, 3), cols='misspell',
                                mapping={'democract': 'democrat'})
        with pytest.raises(KeyError):
            dc.correct_misspell(pd.Series(
                ['misspell', 'correct_spell']), cols='misspell', mapping={'democract': 'democrat'})

        # Range offset for 'cols'
        with pytest.raises(TypeError, match="'cols' must be a sequence like a list or tuple or a single string"):
            dc.correct_misspell(df=test_data, cols=range(
                0, 3), mapping={'democract': 'democrat'})
        # Supplying unknown cols should return KeyErrors (handled by pandas)
        with pytest.raises(KeyError):
            dc.correct_misspell(df=test_data, cols='does_not_exist', mapping={
                                'democract': 'democrat'})
        with pytest.raises(KeyError):
            dc.correct_misspell(df=test_data, cols=['misspell', 'does_not_exist2'], mapping={
                                'democract': 'democrat'})

        # Invalid inputs for 'mapping'
        # Invalid types
        with pytest.raises(TypeError, match="The argument 'mapping' must be a dictionary object"):
            dc.correct_misspell(df=test_data, cols=[
                                'misspell'], mapping=['str'])

    # ------------------- Tests that custom exception is raised ------------------ #

    def test_correct_misspell_custom_error(self, test_data):
        """
        Test that when user passes non 'string' columns in 'cols' the function raises InvalidColumnDtypeError.
        """

        with pytest.raises(InvalidColumnDtypeError, match=escape("Columns ['invalid_case_convert'] are invalid as dtype 'string' is expected")):
            dc.correct_misspell(df=test_data, cols=[
                                'misspell', 'invalid_case_convert'], mapping={'democract': 'democrat'})

    # -------------------------- Tests for functionality ------------------------- #

    @pytest.mark.parametrize(
        "cols, mapping",
        [
            # Single str
            ['misspell', {'repulican': 'republican',
                          'democract': 'democrat'}],
            # Sequence of str (tuple)
            [('misspell', 'case_convert'), {'Upper': 'Changed'}]
        ],
        scope='function'
    )
    def test_correct_misspell(self, test_data, cols, mapping):
        """
        Test that correct_misspell returns expected results given inputs with branches.
        """

        # Test types
        type(dc.correct_misspell(df=test_data, cols=cols,
             mapping=mapping)) == type(pd.DataFrame())

        # Single str branch
        pd.testing.assert_frame_equal(
            left=pd.DataFrame(
                dc.correct_misspell(df=test_data, cols='misspell', mapping={
                    'repulican': 'republican', 'democract': 'democrat'}).misspell
            ),
            right=pd.DataFrame(
                {'misspell': ('republican',) * 5 + ('democrat',) * 5}
            )
        )

        # Sequence of str (tuple) branch
        # This triggers the extra casting branch--- tuple -> list
        pd.testing.assert_frame_equal(
            left=pd.DataFrame(
                dc.correct_misspell(df=test_data, cols=('misspell', 'case_convert'), mapping={
                                    'Upper': 'Changed'}).case_convert
            ),
            right=pd.DataFrame(
                {'case_convert': ('Changed',) * 5 + ('lower',)
                 * 3 + (pd.NA, pd.NA)}
            )
        )

        # Sequence of str (list) branch
        # No casting from tuple -> list
        pd.testing.assert_frame_equal(
            left=pd.DataFrame(
                dc.correct_misspell(df=test_data, cols=['misspell', 'case_convert'], mapping={
                                    'Upper': 'Changed'}).case_convert
            ),
            right=pd.DataFrame(
                {'case_convert': ('Changed',) * 5 + ('lower',)
                 * 3 + (pd.NA, pd.NA)}
            )
        )


# ---------------------------------------------------------------------------- #
#                        Tests for find_missing function                       #
# ---------------------------------------------------------------------------- #


class TestFindMissing:
    """
    Tests for the find_missing helper function.
    """

    # ------------------------ Tests for exceptions raised ----------------------- #

    def test_find_missing_errors(self, test_data):
        """
        Tests that find_missing raises exceptions when 'df' and 'axis' are passed invalid inputs.
        """

        # Unique exceptions due to polymorphism, but mostly attribute errors
        # Places where exceptions will arise are the isna() and any() methods
        # If a series is supplied than the lambda function will error due to series.isna().any() returning only a single bool
        with pytest.raises(AttributeError):
            dc.find_missing((1, 2, 3), axis=0)
        with pytest.raises(IndexError):
            dc.find_missing(pd.Series((1, 2, 3)), axis=0)
        with pytest.raises(AttributeError):
            dc.find_missing({'dict': (1, 2, 3)}, axis=0)

        # Errors for 'axis'
        # Usually type errors with message--- 'The argument 'axis' must be an integer'
        # Value error if supplied anything other than 0 or 1
        with pytest.raises(TypeError):
            dc.find_missing(test_data, axis={'3': 9})
        with pytest.raises(ValueError):
            dc.find_missing(test_data, axis=10)

    # -------------------------- Tests for functionality ------------------------- #

    def test_find_missing(self, test_data):
        """
        Test that find_missing returns expected results given inputs with branches.
        """

        # Return for rows
        pd.testing.assert_series_equal(
            left=dc.find_missing(test_data, axis=1),
            right=pd.Series(data=(True,) * 5, index=(2, 5, 7, 8, 9))
        )

        # Return for columns
        pd.testing.assert_series_equal(
            left=dc.find_missing(test_data, axis=0),
            right=pd.Series(
                data=(True,) * 3, index=('likert_encode', 'str_encode', 'case_convert'))
        )


# ---------------------------------------------------------------------------- #
#                       Tests for Likert encode function                       #
# ---------------------------------------------------------------------------- #

# ---------------------------- Fixture for mapping --------------------------- #

@pytest.fixture(scope='class')
def mapping():
    return [
        {
            'col': 'likert_encode',
            'mapping': {
                'A': 1,
                'B': 2,
                'C': 3,
                'D': 4,
                'E': 5
            }
        },
        {
            'col': 'onehot_encode',
            'mapping': {
                'A': 1,
                'B': 2
            }
        }
    ]

# ---------------------- Fixture for temporary csv file ---------------------- #


@pytest.fixture(scope="session")
def csv_file(tmpdir_factory):
    path = tmpdir_factory.mktemp("data").join("dict.csv")
    return path


class TestLikert:
    """
    Tests for the likert_encode function.
    """

    # ------------------------ Tests for exceptions raised ----------------------- #

    def test_likert_encode_errors(self, test_data, mapping):
        """
        Test that likert_encode raises exceptions when arguments are passed invalid inputs.
        """

        # --------- For 'df', the most common exception is the keyword error --------- #

        with pytest.raises(KeyError):
            dc.likert_encode(pd.Series(('A', 'B', 'C')), mapping)
        with pytest.raises(KeyError):
            dc.likert_encode({'df': 1}, mapping)

        # ---------------------- Custom exceptions with mapping ---------------------- #

        # Wrong key for 'col'
        with pytest.raises(InvalidMappingKeys, match="The argument 'mapping' must be a list of dictionaries with 'col' and 'mapping' as the only two keys"):
            dc.likert_encode(
                test_data, [{
                    'wrong_key': 'onehot_encode',
                    'mapping': {'A': 1, 'B': 2}}
                ])
        # Wrong key for 'mapping'
        with pytest.raises(InvalidMappingKeys, match="The argument 'mapping' must be a list of dictionaries with 'col' and 'mapping' as the only two keys"):
            dc.likert_encode(
                test_data, [{
                    'col': 'onehot_encode',
                    'wrong_values': {'A': 1, 'B': 2}}
                ])
        # Invalid values in 'mapping'
        # The first value should be a str object
        with pytest.raises(InvalidMappingValues, match="The argument 'mapping' must be a list of dictionaries with a string and a dictionary as the only two values"):
            dc.likert_encode(
                test_data, [{
                    'col': 3,
                    'mapping': {'A': 1, 'B': 2}}
                ])
        # The second value should be a dict
        with pytest.raises(InvalidMappingValues, match="The argument 'mapping' must be a list of dictionaries with a string and a dictionary as the only two values"):
            dc.likert_encode(
                test_data, [{
                    'col': 'onehot_encode',
                    'mapping': (1, 2, 3)}
                ])

        # If mapping is not a list of dictionary, an attribute error will be raised since other objects do not have the 'keys' attribute
        with pytest.raises(AttributeError):
            dc.likert_encode(test_data, [3])
        with pytest.raises(AttributeError):
            dc.likert_encode(test_data, pd.Series((1, 2, 3)))

        # If unknown columns are specified in 'col', key errors should be raised
        with pytest.raises(KeyError):
            dc.likert_encode(test_data,         {
                'col': 'unknown',
                'mapping': {
                    'A': 1,
                    'B': 2
                }
            })

        # ---------------------- Invalid inputs for mapping_path --------------------- #

        # Invalid map_path leads to ValueError in pandas' to_csv method
        with pytest.raises(ValueError):
            dc.likert_encode(test_data, mapping, mapping_path=True)
        with pytest.raises(ValueError):
            dc.likert_encode(test_data, mapping,
                             mapping_path=['false', 'path'])

    # -------------------------- Tests for functionality ------------------------- #

    def test_likert_encode(self, test_data, mapping, csv_file):
        """
        Test that likert_encode returns correct output given a set of inputs.
        """

        # --------------------------------- Base case -------------------------------- #

        # Encoded columns should have 'Int64' as dtypes
        pd.testing.assert_frame_equal(
            left=dc.likert_encode(test_data, mapping)[[
                'likert_encode', 'onehot_encode']],
            right=pd.DataFrame({
                'likert_encode': pd.array((1, 1, 2, 3, 4, 4, 3, pd.NA, 3, 5), dtype=pd.Int64Dtype()),
                'onehot_encode': pd.array([1] * 5 + [2] * 5, dtype=pd.Int64Dtype())
            })
        )

        # ----------------------------- Single dictionary ---------------------------- #

        pd.testing.assert_series_equal(
            left=dc.likert_encode(test_data, mapping={
                'col': 'onehot_encode',
                'mapping': {
                    'A': 1,
                    'B': 2
                }
            })['onehot_encode'],
            right=pd.Series(
                pd.array([1] * 5 + [2] * 5, dtype=pd.Int64Dtype()), name='onehot_encode'
            )
        )

        # ------------------ Test saving mapping dictionary to disk ------------------ #

        dc.likert_encode(test_data, mapping, mapping_path=csv_file)
        # Read file from disk and check column names
        assert pd.read_csv(csv_file).columns.tolist() == [
            'Column Name', 'Description (any manipulations, recodes, etc)']
        # Check content of file written to disk
        pd.testing.assert_frame_equal(
            left=pd.read_csv(csv_file),
            right=pd.DataFrame({
                'Column Name': ('likert_encode', 'onehot_encode'),
                'Description (any manipulations, recodes, etc)': ("{'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}", "{'A': 1, 'B': 2}")
            })
        )

        # -------------------------------- Edge cases -------------------------------- #

        # If unknown mappings are specified in 'mapping', encoding should return NA's
        pd.testing.assert_series_equal(
            left=dc.likert_encode(test_data, {
                'col': 'onehot_encode',
                'mapping': {'not_found': 1, 'unknown': 2}})['onehot_encode'],
            right=pd.Series([pd.NA] * 10, name='onehot_encode', dtype='Int64')
        )

        # Values that are not in the dictionary (as keys) are converted to NaN and NA
        pd.testing.assert_series_equal(
            # Everything other than 'A' and 'B' should be NA
            left=dc.likert_encode(test_data, {
                'col': 'likert_encode',
                'mapping': {'A': 1, 'B': 2}})['likert_encode'],
            right=pd.Series((1, 1, 2, pd.NA, pd.NA, pd.NA, pd.NA,
                            pd.NA, pd.NA, pd.NA), name='likert_encode', dtype='Int64')
        )

        # Columns containing None should be treated as np.NaN
        pd.testing.assert_series_equal(
            # Everything other than 'A' and 'B' should be NA
            left=dc.likert_encode(
                pd.DataFrame({'None': ('A', None, None, 'B')}), mapping={
                    'col': 'None',
                    'mapping': {'A': 1, 'B': 2}})['None'],
            right=pd.Series((1, pd.NA, pd.NA, 2), name='None', dtype='Int64')
        )

        # Mapping to float instead of integers should return a series with dtype as 'float64'
        # This is because df[col].astype(errors='ignore') should return original object on error
        pd.testing.assert_series_equal(
            # Everything other than 'A' and 'B' should be NA
            left=dc.likert_encode(
                pd.DataFrame({'float': ('A', None, 'C', 'B', 'D')}), mapping={
                    'col': 'float',
                    'mapping': {'A': 1, 'B': 2, 'C': -999, 'D': 2.34}})['float'],
            right=pd.Series((1, np.NaN, -999, 2, 2.34),
                            name='float', dtype='float64')
        )

# ---------------------------------------------------------------------------- #
#                       Tests for onehot encode function                       #
# ---------------------------------------------------------------------------- #


class TestOnehot:
    """
    Tests for the onehot_encode function.
    """

    # -------------------------------- Exceptions -------------------------------- #

    def test_onehot_encode_errors(self, test_data):
        """
        Test that onehot_encode raises exceptions when arguments are passed invalid inputs.
        """

        # ---------------------------- Enforced exceptions --------------------------- #

        # Invalid type for 'df'
        with pytest.raises(TypeError, match="'df' must be be a DataFrame"):
            dc.onehot_encode(
                pd.Series(('A', 'B', 'C'), name='col'), cols='col')

        # Invalid type for 'cols'
        with pytest.raises(TypeError, match="'cols' must be a sequence like a list or tuple or a single string"):
            dc.onehot_encode(test_data, cols=range(1, 3))

        # -------------------------- Not enforced exceptions ------------------------- #

        # Supply columns that do not exist in 'df'
        with pytest.raises(KeyError):
            dc.onehot_encode(test_data, cols=['do_not_exist', 'onehot_encode'])

        # Attempts to select columns to encode using integer indices
        with pytest.raises(KeyError):
            dc.onehot_encode(test_data, cols=list(range(1, 3)))

        # Attempts to select columns to encode using boolean indices
        with pytest.raises(KeyError):
            dc.onehot_encode(test_data, cols=(True, False))

        # -------------------------- Tests for functionality ------------------------- #

    def test_onehot_encode(self, test_data):
        """
        Test that onehot_encode returns correct output given a set of inputs.
        """

        # --------------------------------- Base case -------------------------------- #

        pd.testing.assert_frame_equal(
            left=dc.onehot_encode(test_data, 'onehot_encode')[
                ['onehot_encode', 'onehot_encode_A', 'onehot_encode_B']],
            right=pd.DataFrame({
                'onehot_encode': pd.array(['A'] * 5 + ['B'] * 5, dtype='object'),
                'onehot_encode_A': pd.array([1] * 5 + [0] * 5, dtype=np.uint8),
                'onehot_encode_B': pd.array([0] * 5 + [1] * 5, dtype=np.uint8)
            })
        )

        # ------------------------------ Numeric columns ----------------------------- #

        # Float with missing
        pd.testing.assert_frame_equal(
            left=dc.onehot_encode(
                pd.DataFrame({
                    'col': pd.array((2.1, 2.2, np.NaN), dtype='float')
                }), 'col'),
            right=pd.DataFrame({
                'col': pd.array((2.1, 2.2, np.NaN), dtype='float'),
                'col_2.1': pd.array((1, 0, 0), dtype=np.uint8),
                'col_2.2': pd.array((0, 1, 0), dtype=np.uint8),
                'col_nan': pd.array((0, 0, 1), dtype=np.uint8)
            })
        )

        # Nullable Integer with missing
        pd.testing.assert_frame_equal(
            left=dc.onehot_encode(
                pd.DataFrame({
                    'col': pd.Series((1, 2, pd.NA), dtype='Int64')
                }), 'col'),
            right=pd.DataFrame({
                'col': pd.array((1, 2, pd.NA), dtype='Int64'),
                'col_1': pd.array((1, 0, 0), dtype=np.uint8),
                'col_2': pd.array((0, 1, 0), dtype=np.uint8),
                'col_nan': pd.array((0, 0, 1), dtype=np.uint8)
            })
        )

        # -------------------------- String and categorical -------------------------- #

        # String with missing
        pd.testing.assert_frame_equal(
            left=dc.onehot_encode(
                pd.DataFrame({
                    'col': pd.array(('A', 'B', pd.NA), dtype='string')
                }), 'col'),
            right=pd.DataFrame({
                'col': pd.array(('A', 'B', pd.NA), dtype='string'),
                'col_A': pd.array((1, 0, 0), dtype=np.uint8),
                'col_B': pd.array((0, 1, 0), dtype=np.uint8),
                'col_nan': pd.array((0, 0, 1), dtype=np.uint8)
            })
        )

        # Categorical with missing
        pd.testing.assert_frame_equal(
            left=dc.onehot_encode(
                pd.DataFrame({
                    'col': pd.array(["a", "b", "c", "a", pd.NA], dtype="category")
                }), 'col'),
            right=pd.DataFrame({
                'col': pd.array(["a", "b", "c", "a", pd.NA], dtype="category"),
                'col_a': pd.array((1, 0, 0, 1, 0), dtype=np.uint8),
                'col_b': pd.array((0, 1, 0, 0, 0), dtype=np.uint8),
                'col_c': pd.array((0, 0, 1, 0, 0), dtype=np.uint8),
                'col_nan': pd.array((0, 0, 0, 0, 1), dtype=np.uint8)
            })
        )

        # Boolean with missing
        pd.testing.assert_frame_equal(
            left=dc.onehot_encode(
                pd.DataFrame({
                    'col': pd.array([True, False, pd.NA], dtype="boolean")
                }), 'col'),
            right=pd.DataFrame({
                'col': pd.array([True, False, pd.NA], dtype="boolean"),
                'col_False': pd.array((0, 1, 0), dtype=np.uint8),
                'col_True': pd.array((1, 0, 0), dtype=np.uint8),
                'col_nan': pd.array((0, 0, 1), dtype=np.uint8)
            })
        )
