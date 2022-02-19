import pycitizen.aws_utils as au
from pycitizen.exceptions import ColumnDtypeInferError
import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope='module')
def read_test_frame():
    return pd.read_csv('tests/test_data_frame.csv')

# ---------------------------------------------------------------------------- #
#                         Non-AWS application programs                         #
# ---------------------------------------------------------------------------- #


class TestInputValidation:
    """
    Check the exceptions raised when functions fail.
    """

    # ------------------------ Tests for create_statement ------------------------ #

    @pytest.mark.parametrize(
        "df, tbl_name, primary_key",
        [
            # Case 1 (wrong type for df)
            (pd.Series((2, 3, 4, 5)), 'nm', 'pkey'),
            # Case 2 (wrong type for tbl_name)
            (pd.DataFrame({'a': (1, 2), 'b': ('a', 'b')}),
             {'test': 2, "case": "4"}, 'pkey'),
            # Case 3 (wrong type for primary_key)
            (pd.DataFrame({'a': (1, 2), 'b': ('a', 'b')}),
             'nm', ["key1", "key2"])
        ],
        scope='function'
    )
    def test_create_statement_type_error(self, df, tbl_name, primary_key):
        with pytest.raises(TypeError, match="'df' must be a data frame and 'tbl_name' and 'primary_key' must be string objects"):
            au.create_statement(df, tbl_name, primary_key)

    @pytest.mark.parametrize(
        "df, tbl_name, primary_key",
        [
            # Case 1 (df contains 'bool')
            (pd.DataFrame({'a': (False, True), 'b': ('a', 'b')}),
             'nm', 'pkey'),
            # Case 2 (df contains 'categorical')
            (pd.DataFrame(pd.Series(["a", "b", "c", "a"], dtype="category")),
             'nm', 'pkey')
        ],
        scope='function'
    )
    def test_create_statement_col_dtype_error(self, df, tbl_name, primary_key):
        with pytest.raises(ColumnDtypeInferError, match="'df' contains columns with dtypes that cannot be inferred see documentation for supported dtypes"):
            au.create_statement(df, tbl_name, primary_key)

    # ------------------------ Tests for create_statements ----------------------- #

    @pytest.mark.parametrize(
        "df_seq, tbl_names, primary_keys",
        [
            # Case 1 (single df error)
            (
                pd.DataFrame(
                    {'a': (1, 2), 'b': ('a', 'b')}),
                ('nm1', 'nm2'),
                ['pkey1', 'pkey2']
            ),
            # Case 2 (single tbl_name error)
            (
                [pd.DataFrame(
                    {'a': (1, 2), 'b': ('a', 'b')}),
                 pd.DataFrame(
                    {'a': (1, 2), 'b': ('a', 'b')})],
                'single_table_name',
                ['pkey1', 'pkey2']
            ),
            # Case 3 (single primary_keys error)
            (
                [pd.DataFrame(
                    {'a': (1, 2), 'b': ('a', 'b')}),
                 pd.DataFrame(
                    {'a': (1, 6), 'b': ('a', 'b')})],
                ('nm1', 'nm2'),
                'single_primary_key_error'
            )
        ],
        scope='function'
    )
    def test_create_statementS_type_error(self, df_seq, tbl_names, primary_keys):
        with pytest.raises(TypeError, match="'df_seq', 'tbl_names', and 'primary_keys' must be sequences like lists or tuples"):
            au.create_statements(df_seq, tbl_names, primary_keys)

    @pytest.mark.parametrize(
        "df_seq, tbl_names, primary_keys",
        [(
            [pd.DataFrame(
                {'a': (1, 2), 'b': ('a', 'b')}),
             pd.DataFrame(
                {'a': (1, 6), 'b': ('a', 'b')})],
            ('nm1', 'nm2'),
            # Wrong length
            ['key'] * 3
        )],
        scope='function'
    )
    def test_create_statementS_len_error(self, df_seq, tbl_names, primary_keys):
        with pytest.raises(ValueError, match="'args' must be sequences of the same lengths"):
            au.create_statements(df_seq, tbl_names, primary_keys)

    # -------------------------- Tests for create_tables ------------------------- #

    @pytest.mark.parametrize(
        "commands",
        [
            # Case 1
            ["command1", "command2"],
            # Case 2
            "single_statement"

        ],
        scope='function'
    )
    def test_create_tables_type_error(self, commands):
        with pytest.raises(TypeError, match="'commands' must be a tuple"):
            au.create_tables(commands, db_name="name", host="host",
                             port="port", user="user", db_password="pass")

    @pytest.mark.parametrize(
        "commands", [("command1", b'commands2')],
        scope='function'
    )
    def test_create_tables_element_error(self, commands):
        with pytest.raises(TypeError, match="All CREATE TABLE statements in 'commands' must be string objects"):
            au.create_tables(commands, db_name="name", host="host",
                             port="port", user="user", db_password="pass")

    # --------------------------- Tests for copy_tables -------------------------- #

    @pytest.mark.parametrize(
        "tbl_names, paths",
        [
            # Case 1 (wrong type for tbl_names)
            ("single_tbl_name", ("path1", "path2")),
            # Case 2 (wrong type for paths)
            (["tbl_name1", "tbl_name2"], "single_path"),
            # Case 3 (both are not sequences)
            ("single_name", "single_path")
        ],
        scope='function'
    )
    def test_copy_tables_type_error(self, tbl_names, paths):
        with pytest.raises(TypeError, match="Both 'table_names' and 'paths' must be sequences like lists or tuples"):
            au.copy_tables(tbl_names, paths, access_key="abcd", secret_key="efgh", db_name="name", host="host",
                           port="port", user="user", db_password="pass")

    @pytest.mark.parametrize(
        "tbl_names, paths", [(['tbl1', 'tbl2'], ['path'] * 3)],
        scope='function'
    )
    def test_copy_tables_len_error(self, tbl_names, paths):
        with pytest.raises(ValueError, match="'args' must be sequences of the same lengths"):
            au.copy_tables(tbl_names, paths, access_key="abcd", secret_key="efgh", db_name="name", host="host",
                           port="port", user="user", db_password="pass")

    # --------------------------- Tests for rename_col --------------------------- #

    @pytest.mark.parametrize(
        "tbl_names, old_nms, new_nms",
        [
            # Case 1 (single tbl_name)
            (
                "tbl_name",
                ('old1', 'old2'),
                ('new1', 'new2')
            ),
            # Case 2 (single strings)
            (
                "tbl_name",
                "old",
                "new"
            )
        ],
        scope='function'
    )
    def test_rename_col_type_error(self, tbl_names, old_nms, new_nms):
        with pytest.raises(TypeError, match="tbl_names', 'old_nms', and 'new_nms' must be sequences like lists or tuples"):
            au.rename_col(tbl_names, old_nms, new_nms, db_name="name", host="host",
                          port="port", user="user", db_password="pass")

    @pytest.mark.parametrize(
        "tbl_names, old_nms, new_nms",
        [(('tbl1', 'tbl2'), ('old1', 'old2'), ['new'] * 3)],
        scope='function'
    )
    def test_rename_col_len_error(self, tbl_names, old_nms, new_nms):
        with pytest.raises(ValueError, match="'args' must be sequences of the same lengths"):
            au.rename_col(tbl_names, old_nms, new_nms, db_name="name", host="host",
                          port="port", user="user", db_password="pass")

    # --------------------------- Tests for rename_tbl --------------------------- #

    @pytest.mark.parametrize(
        "old_nms, new_nms",
        [
            # Case 1 (single old name)
            (
                'old1',
                ('new1', 'new2')
            ),
            # Case 2 (single strings)
            (
                "old",
                "new"
            )
        ],
        scope='function'
    )
    def test_rename_tbl_type_error(self, old_nms, new_nms):
        with pytest.raises(TypeError, match="Both 'old_nms', and 'new_nms' must be sequences like lists or tuples"):
            au.rename_tbl(old_nms, new_nms, db_name="name", host="host",
                          port="port", user="user", db_password="pass")

    @pytest.mark.parametrize(
        "old_nms, new_nms", [(('old1', 'old2'), ['new'] * 3)],
        scope='function'
    )
    def test_rename_tbl_len_error(self, old_nms, new_nms):
        with pytest.raises(ValueError, match="'args' must be sequences of the same lengths"):
            au.rename_tbl(old_nms, new_nms, db_name="name", host="host",
                          port="port", user="user", db_password="pass")
