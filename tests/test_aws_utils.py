import pycitizen.aws_utils as au
from pycitizen.exceptions import ColumnDtypeInferError
from psycopg2.extensions import connection
from pytest_mock_resources import create_redshift_fixture, Statements
import pytest
import pandas as pd
import numpy as np

# --------------------------------- Test data -------------------------------- #


@pytest.fixture(scope='module')
def read_test_frame():
    return pd.read_csv(
        'tests/test_data_frame.csv',
        parse_dates=['date', 'date_with_na'],
        dtype={
            'int8': np.int8,
            'int16_missing': 'Int64',
            'int32': np.int32,
            'int64': np.int64,
            'float16': np.float16,
            'float32': np.float32,
            'float64': np.float64
        }
    )

# -- Test data containing columns with dtypes that have not been implemented - #


@pytest.fixture(scope='class')
def df():
    return pd.DataFrame({'col': pd.Series([1, 2, 3, pd.NA], dtype="Int64"),
                         'key': ('a', 'b', 'c', 'd'),
                         'err1': (True, False, True, False),
                         'err2': pd.Series(["a", "b", "c"], dtype="string"),
                         'err3': pd.Series(["a", "b", "c", "a"], dtype="category")})

# ---------------------------------------------------------------------------- #
#                     Test that specific errors are raised                     #
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
        """
        Exception raised that 'df' must be a data frame and 'tbl_name' and 'primary_key' must be string objects.
        """
        with pytest.raises(TypeError, match="'df' must be a data frame and 'tbl_name' and 'primary_key' must be string objects"):
            au.create_statement(df, tbl_name, primary_key)

    def test_create_statement_col_dtype_error(self, df):
        """
        Exception raised that 'df' contains columns with dtypes that cannot be inferred.
        """
        with pytest.raises(ColumnDtypeInferError, match="The dtypes of the following columns cannot be inferred"):
            au.create_statement(df, 'test', 'key')

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
        """
        Exception raised that 'df_seq', 'tbl_names', and 'primary_keys' must be sequences like lists or tuples.
        """
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
        """
        Exception raised that 'df_seq', 'tbl_names', and 'primary_keys' must have equal lengths.
        """
        with pytest.raises(ValueError, match="'df_seq', 'tbl_names', and 'primary_keys' must have equal lengths"):
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
        """
        Exception raised that 'commands' must be a tuple. This may be changed to allow more flexibility in the future.
        """
        with pytest.raises(TypeError, match="'commands' must be a tuple"):
            au.create_tables(commands, db_name="name", host="host",
                             port="port", user="user", db_password="pass")

    @pytest.mark.parametrize(
        "commands", [("command1", b'commands2')],
        scope='function'
    )
    def test_create_tables_element_error(self, commands):
        """
        Exception raised that all elements in 'commands' must be strings.
        """
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
        """
        Exception raised that 'table_names' and 'paths' must be sequences like lists or tuples.
        """
        with pytest.raises(TypeError, match="'table_names' and 'paths' must be sequences like lists or tuples"):
            au.copy_tables(tbl_names, paths, access_key="abcd", secret_key="efgh", db_name="name", host="host",
                           port="port", user="user", db_password="pass")

    @pytest.mark.parametrize(
        "tbl_names, paths", [(['tbl1', 'tbl2'], ['path'] * 3)],
        scope='function'
    )
    def test_copy_tables_len_error(self, tbl_names, paths):
        """
        Exception raised that 'table_names' and 'paths' must have equal lengths.
        """
        with pytest.raises(ValueError, match="'table_names' and 'paths' must have equal lengths"):
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
        """
        Exception raised that 'tbl_names', 'old_nms', and 'new_nms' must be sequences like lists or tuples.
        """
        with pytest.raises(TypeError, match="'tbl_names', 'old_nms', and 'new_nms' must be sequences like lists or tuples"):
            au.rename_col(tbl_names, old_nms, new_nms, db_name="name", host="host",
                          port="port", user="user", db_password="pass")

    @pytest.mark.parametrize(
        "tbl_names, old_nms, new_nms",
        [(('tbl1', 'tbl2'), ('old1', 'old2'), ['new'] * 3)],
        scope='function'
    )
    def test_rename_col_len_error(self, tbl_names, old_nms, new_nms):
        """
        Exception raised that 'tbl_names', 'old_nms', and 'new_nms' must have equal lengths.
        """
        with pytest.raises(ValueError, match="'tbl_names', 'old_nms', and 'new_nms' must have equal lengths"):
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
        """
        Exception raised that 'old_nms' and 'new_nms' must be sequences like lists or tuples.
        """
        with pytest.raises(TypeError, match="'old_nms' and 'new_nms' must be sequences like lists or tuples"):
            au.rename_tbl(old_nms, new_nms, db_name="name", host="host",
                          port="port", user="user", db_password="pass")

    @pytest.mark.parametrize(
        "old_nms, new_nms", [(('old1', 'old2'), ['new'] * 3)],
        scope='function'
    )
    def test_rename_tbl_len_error(self, old_nms, new_nms):
        """
        Exception raised that 'old_nms' and 'new_nms' must have equal lengths.
        """
        with pytest.raises(ValueError, match="'old_nms' and 'new_nms' must have equal lengths"):
            au.rename_tbl(old_nms, new_nms, db_name="name", host="host",
                          port="port", user="user", db_password="pass")

# ---------------------------------------------------------------------------- #
#    Test that create statement functions infer column 'dtypes' accordingly    #
# ---------------------------------------------------------------------------- #

# ----------------------- Test data for two edge cases ----------------------- #


@pytest.fixture(scope='class')
def df1():
    return pd.DataFrame(
        {'col': (1, 2, 3, pd.NA),
         'key': ('a', 'b', 'c', 'd')}
    )


@pytest.fixture(scope='class')
def df2():
    return pd.DataFrame(
        {'col': ('1', '2.342', '3', 'str'),
         'key': ('a', 'b', 'c', 'd')}
    )


@pytest.fixture(scope='class')
def df3():
    return pd.DataFrame(
        {'col': pd.to_datetime(['2018-10-26', '2018-10-26', '2022-10-22', '2022-12-27']),
         'key': ('a', 'b', 'c', 'd')}
    )


class TestCreateStatement:
    """
    Check that create_statement and create_statements generate sql command(s) with correctly inferred data type(s). Also test the return object(s) is of the expected class(es).
    """

    # ----------------------------- Single statement ----------------------------- #

    def test_create_statement_class(self, read_test_frame):
        """
        Check that the single create statement function returns a single string object.
        """
        assert isinstance(au.create_statement(
            read_test_frame, 'test', 'key'), str)

    def test_create_statement_output(self, read_test_frame):
        """
        Check that the single create statement function returns expected output given fixture dtypes.
        """
        assert au.create_statement(read_test_frame, 'test', 'key') == 'CREATE TABLE test (date DATE, date_with_na DATE, key VARCHAR(6) NOT NULL, varchar_long VARCHAR(36), varchar_missing VARCHAR(1), int8 INTEGER, int16_missing INTEGER, int32 INTEGER, int64 INTEGER, float16 REAL, float32 REAL, float64_missing REAL, mix_str_int_missing VARCHAR(2), PRIMARY KEY (key))'

    def test_create_statement_edge(self, df1, df2):
        """
        Check some edge cases--- int column with pd.NA and string column with numbers.
        """
        assert au.create_statement(
            df1, 'test1', 'key') == 'CREATE TABLE test1 (col INTEGER, key VARCHAR(1) NOT NULL, PRIMARY KEY (key))'
        assert au.create_statement(
            df2, 'test2', 'key') == 'CREATE TABLE test2 (col VARCHAR(5), key VARCHAR(1) NOT NULL, PRIMARY KEY (key))'

    # ---------------------------- Multiple statements --------------------------- #

    def test_create_statementS_class(self, read_test_frame, df1, df2):
        """
        Check that the vectorized create statement function returns a tuple of strings.
        """
        commands = au.create_statements((read_test_frame, df1, df2),
                                        ('test1', 'test2', 'test3'), ('key', 'key', 'key'))
        elements = [isinstance(command, str) for command in commands]
        assert isinstance(commands, tuple)
        assert all(elements) == True

    def test_create_statementS_output(self, read_test_frame, df1, df2, df3):
        """
        Check that the vectorized create statement function returns expected output given a set of fixure inputs.
        """
        assert au.create_statements((read_test_frame, df1, df2, df3),
                                    ('test1', 'test2', 'test3', 'test4'), ('key', 'key', 'key', 'key')) == (
                                        'CREATE TABLE test1 (date DATE, date_with_na DATE, key VARCHAR(6) NOT NULL, varchar_long VARCHAR(36), varchar_missing VARCHAR(1), int8 INTEGER, int16_missing INTEGER, int32 INTEGER, int64 INTEGER, float16 REAL, float32 REAL, float64_missing REAL, mix_str_int_missing VARCHAR(2), PRIMARY KEY (key))',
                                        'CREATE TABLE test2 (col INTEGER, key VARCHAR(1) NOT NULL, PRIMARY KEY (key))',
                                        'CREATE TABLE test3 (col VARCHAR(5), key VARCHAR(1) NOT NULL, PRIMARY KEY (key))',
                                        'CREATE TABLE test4 (col DATE, key VARCHAR(1) NOT NULL, PRIMARY KEY (key))'
        )


# ---------------------------------------------------------------------------- #
#                      Test classes defined in the module                      #
# ---------------------------------------------------------------------------- #

# --------------------------- Fixure redshift table -------------------------- #

# Three columns and ten rows
statements = Statements(
    """
    CREATE TABLE test(
      id INTEGER PRIMARY KEY,
      str VARCHAR (1),
      num INTEGER
    );
    """,
    """
    INSERT INTO test (id, str, num) VALUES (1, 'A', 3), (2, 'B', 4), (3, 'D', 3), (4, 'Z', 4), (5, 'A', 3), (6, 'C', 4), (7, 'F', 3), (8, 'B', 4), (9, 'D', 3), (10, 'G', 4);
    """
)

# ---------------------- Tests for the MyRedshift class ---------------------- #

redshift = create_redshift_fixture(
    statements,
    scope='class'
)


def test_MyRedShift(redshift):
    """
    Tests for MyRedshift class constructor, attributes, and methods.
    """
    # Obtain credentials
    credentials = redshift.pmr_credentials.as_psycopg2_kwargs()
    params = (
        credentials['dbname'],
        credentials['host'],
        credentials['port'],
        credentials['user'],
        credentials['password']
    )
    # Instantiate class
    db = au.MyRedShift(*params)

    # Test that class constructor returns expected object
    assert isinstance(db, au.MyRedShift)

    # Check attributes
    assert db.db_name == params[0]
    assert db.host == params[1]
    assert db.port == params[2]
    assert db.user == params[3]
    assert db.db_password == params[4]

    # Test get_params() method
    assert isinstance(db.get_params(), tuple)
    assert db.get_params() == params

    # Test connect() method
    assert isinstance(db.connect(), connection)

    # Test read_tbl() method
    # Other branch chunksize!=None currently cannot be tested due to pandas internals incompatible with pytest_mock_resources
    pd.testing.assert_frame_equal(
        db.read_tbl(tbl='test', chunksize=None),
        pd.DataFrame({
            'id': tuple(range(1, 11)),
            'str': ('A', 'B', 'D', 'Z', 'A', 'C', 'F', 'B', 'D', 'G'),
            'num': (3, 4) * 5
        })
    )

    # Test read_query() method
    pd.testing.assert_frame_equal(
        db.read_query(sql="SELECT * FROM test;", chunksize=None),
        pd.DataFrame({
            'id': tuple(range(1, 11)),
            'str': ('A', 'B', 'D', 'Z', 'A', 'C', 'F', 'B', 'D', 'G'),
            'num': (3, 4) * 5
        })
    )

# ----------------------- Tests for the AwsCreds class ----------------------- #


@pytest.fixture
def aws_creds():
    return ('access', 'secret')


class TestAwsCreds:
    """
    Tests for the AwsCreds class constructor, attributes, and methods.
    """

    def test_AwsCreds_class_constructor(self, aws_creds):
        """
        Tests for the AwsCreds class constructor.
        """
        # Instantiate
        creds = au.AwsCreds(*aws_creds)

        # Attributes
        assert creds.access_key == aws_creds[0]
        assert creds.secret_key == aws_creds[1]

        # Method
        assert isinstance(creds.get_params(), tuple)
        assert creds.get_params() == aws_creds
