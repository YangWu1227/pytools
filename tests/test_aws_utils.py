# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #

from psycopg2.extensions import connection
from pytest_mock_resources import create_redshift_fixture, Statements
import boto3
from moto import mock_s3
import pytest
import pandas as pd
import numpy as np

# ----------------------------- Standard library ----------------------------- #

from io import StringIO
import os

# ------------------------------- Intra-package ------------------------------ #

import pycitizen.aws_utils as au
from pycitizen.exceptions import ColumnDtypeInferError

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

# ------------------------- Redshift database fixture ------------------------ #

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

# ---------------------------------------------------------------------------- #
#                        Tests for the MyRedshift class                        #
# ---------------------------------------------------------------------------- #

redshift = create_redshift_fixture(
    statements,
    scope='module'
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
    # Instantiate class instance
    db = au.MyRedShift(*params)

    # ------------ Test that class constructor returns expected object ----------- #

    assert isinstance(db, au.MyRedShift)

    # ----------------------------- Check attributes ----------------------------- #

    assert db.db_name == params[0]
    assert db.host == params[1]
    assert db.port == params[2]
    assert db.user == params[3]
    assert db.db_password == params[4]

    # ------------------------- Test get_params() method ------------------------- #

    assert isinstance(db.get_params(), tuple)
    assert db.get_params() == params

    # --------------------------- Test connect() method -------------------------- #

    assert isinstance(db.connect(), connection)

    # -------------------------- Test read_tbl() method -------------------------- #

    # Other branch chunksize!=None cannot be tested currently due to pandas internals being incompatible with pytest_mock_resources
    pd.testing.assert_frame_equal(
        db.read_tbl(tbl='test', chunksize=None),
        pd.DataFrame({
            'id': tuple(range(1, 11)),
            'str': ('A', 'B', 'D', 'Z', 'A', 'C', 'F', 'B', 'D', 'G'),
            'num': (3, 4) * 5
        })
    )

    # -------------------------- Test read_tbl() method -------------------------- #

    pd.testing.assert_frame_equal(
        db.read_query(sql="SELECT * FROM test;", chunksize=None),
        pd.DataFrame({
            'id': tuple(range(1, 11)),
            'str': ('A', 'B', 'D', 'Z', 'A', 'C', 'F', 'B', 'D', 'G'),
            'num': (3, 4) * 5
        })
    )

    # ------------------------ Test create_tbl() method -------------------------- #

    # -------------------------------- Exceptions -------------------------------- #

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
    def test_create_tbl_type_error(self, commands):
        """
        Exception raised that 'commands' must be a tuple. This may be changed to allow more flexibility in the future.
        """
        with pytest.raises(TypeError, match="'commands' must be a tuple"):
            db.create_tbl(commands)

    @pytest.mark.parametrize(
        "commands", [("command1", b'commands2')],
        scope='function'
    )
    def test_create_tbl_element_error(self, commands):
        """
        Exception raised that all elements in 'commands' must be strings.
        """
        with pytest.raises(TypeError, match="All CREATE TABLE statements in 'commands' must be string objects"):
            db.create_tbl(commands)

    # -------------------------- Test copy_tbl() method -------------------------- #

    # -------------------------------- Exceptions -------------------------------- #

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
    def test_copy_tbl_type_error(self, tbl_names, paths):
        """
        Exception raised that 'tbl_names' and 'paths' must be sequences like lists or tuples.
        """
        with pytest.raises(TypeError, match="'tbl_names' and 'paths' must be sequences like lists or tuples"):
            db.copy_tbl(tbl_names, paths)

    @pytest.mark.parametrize(
        "tbl_names, paths", [(['tbl1', 'tbl2'], ['path'] * 3)],
        scope='function'
    )
    def test_copy_tbl_len_error(self, tbl_names, paths):
        """
        Exception raised that 'tbl_names' and 'paths' must have equal lengths.
        """
        with pytest.raises(ValueError, match="'tbl_names' and 'paths' must have equal lengths"):
            db.copy_tbl(tbl_names, paths)

    # ------------------------- Test rename_col() method ------------------------- #

    # -------------------------------- Exceptions -------------------------------- #

    @pytest.mark.parametrize(
        "tbl_names, old_nms, new_nms",
        [
            # Case 1 (set tbl_name)
            (
                {"tbl_name"},
                ('old1', 'old2'),
                ('new1', 'new2')
            ),
            # Case 2 (tbl_names is a sequence while the other two are strings)
            (
                ("tbl_name",),
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
            db.rename_col(tbl_names, old_nms, new_nms)

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
            db.rename_col(tbl_names, old_nms, new_nms)

    # ------------------------- Test rename_tbl() method ------------------------- #

    # -------------------------------- Exceptions -------------------------------- #

    @pytest.mark.parametrize(
        "old_nms, new_nms",
        [
            # Case 1 (set old name)
            (
                {'old1', 'old2'},
                ('new1', 'new2')
            ),
            # Case 2 (arguments must either be all strings or all sequences)
            (
                ["old"],
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
            db.rename_tbl(old_nms, new_nms)

    @pytest.mark.parametrize(
        "old_nms, new_nms", [(('old1', 'old2'), ['new'] * 3)],
        scope='function'
    )
    def test_rename_tbl_len_error(self, old_nms, new_nms):
        """
        Exception raised that 'old_nms' and 'new_nms' must have equal lengths.
        """
        with pytest.raises(ValueError, match="'old_nms' and 'new_nms' must have equal lengths"):
            db.rename_tbl(old_nms, new_nms)

# ---------------------------------------------------------------------------- #
#                         Tests for the AwsCreds class                         #
# ---------------------------------------------------------------------------- #


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
        # Instantiate a class instance
        creds = au.AwsCreds(*aws_creds)

        # Attributes
        assert creds.access_key == aws_creds[0]
        assert creds.secret_key == aws_creds[1]

        # Method
        assert isinstance(creds.get_params(), tuple)
        assert creds.get_params() == aws_creds


# ---------------------------------------------------------------------------- #
#                   Tests for database interaction functions                   #
# ---------------------------------------------------------------------------- #

# --------------------------- Create table commands -------------------------- #

@pytest.fixture(scope='function')
def create_commands():
    """
    Fixture for create_tbl() function.
    """
    return (
        'CREATE TABLE test1 (col INTEGER, key VARCHAR(1) NOT NULL, PRIMARY KEY (key))',
        'CREATE TABLE test2 (col2 VARCHAR(5), col2_2 REAL, key2 VARCHAR(1) NOT NULL, PRIMARY KEY (key2))',
        'CREATE TABLE test3 (col3 INTEGER, key3 VARCHAR(1) NOT NULL, PRIMARY KEY (key3))',
        'CREATE TABLE test4 (col4 VARCHAR(5), col4_4 REAL, key4 VARCHAR(1) NOT NULL, PRIMARY KEY (key4))',
    )

# ----------------- Tests for database interaction functions ----------------- #


def test_database_interaction(redshift, create_commands):
    """
    Tests for create_tbl(), rename_tbl(), and rename_col(). The copy_tbl() function
    is currently tested only manually outside of this automated testing framework.
    """
    # Database connection parameters
    credentials = redshift.pmr_credentials.as_psycopg2_kwargs()
    params = (
        credentials['dbname'],
        credentials['host'],
        credentials['port'],
        credentials['user'],
        credentials['password']
    )
    # Instantiate a class instance
    db = au.MyRedShift(*params)

    # ----------------------------- Test create table ---------------------------- #

    db.create_tbl(
        create_commands
    )
    # Retrieve created tables
    df1 = db.read_tbl('test1', None)
    df2 = db.read_tbl('test2', None)
    # Check if columns match those listed in the create statements
    assert df1.columns.tolist() == ['col', 'key']
    assert df2.columns.tolist() == ['col2', 'col2_2', 'key2']

    # ----------------------------- Test rename table ---------------------------- #

    # ---------------------------------- Case 1 ---------------------------------- #

    # Change table names (multiple statements)
    db.rename_tbl(('test1', 'test2'), ('new1', 'new2'))
    # Verify by reading the tables using their new names and checking if they contain original columns
    assert db.read_tbl('new1', None).columns.tolist() == ['col', 'key']
    assert db.read_tbl('new2', None).columns.tolist() == [
        'col2', 'col2_2', 'key2']

    # ---------------------------------- Case 2 ---------------------------------- #

    # Change table name (single statement)
    db.rename_tbl('test3', 'new3')
    # Verify by reading the table using its new name and checking if it contain original columns
    assert db.read_tbl('new3', None).columns.tolist() == ['col3', 'key3']

    # ---------------------------- Test rename column ---------------------------- #

    # ---------------------------------- Case 1 ---------------------------------- #

    # Change columns names (multiple statements)
    # Change 'col' to 'new_col' in table 'new1'
    # Change 'col2' to 'new_col2' in table 'new2'
    db.rename_col(
        ('new1', 'new2'),
        ('col', 'col2'),
        ('new_col', 'new_col2')
    )
    # Verify if column names have been changed
    df1_new = db.read_tbl('new1', None)
    df2_new = db.read_tbl('new2', None)
    assert df1_new.columns.tolist() == ['new_col', 'key']
    assert df2_new.columns.tolist() == ['new_col2', 'col2_2', 'key2']

    # ---------------------------------- Case 2 ---------------------------------- #

    # Change columns name (single statement)
    # Change column 'col3' in table 'new3' to 'new_col3'
    db.rename_col(
        'new3',
        'col3',
        'new_col3'
    )
    # Verify if column name has been changed
    df3_new = db.read_tbl('new3', None)
    assert df3_new.columns.tolist() == ['new_col3', 'key3']

    # ---------------------------------- Case 3 ---------------------------------- #

    # Change column names (recycle the first string to match those of the other two)
    # Change columns 'col4' and 'col4_4' in table 'test4' to 'new_col4' and 'new_col4_4'
    db.rename_col(
        'test4',
        ['col4', 'col4_4'],
        ('new_col4', 'new_col4_4')
    )
    # Verify if column name has been changed
    df4_new = db.read_tbl('test4', None)
    assert df4_new.columns.tolist() == ['new_col4', 'new_col4_4', 'key4']

# ---------------------------------------------------------------------------- #
#                       Tests for S3 interaction function                      #
# ---------------------------------------------------------------------------- #

# ------------------------------ Fixtures for s3 ----------------------------- #


@pytest.fixture(scope='function')
def aws_credentials():
    """
    Mocked AWS credentials for moto.
    """
    os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'


@pytest.fixture(scope='function')
def s3(aws_credentials):
    """
    Mocked S3 client object.
    """
    with mock_s3():
        yield boto3.client('s3', region_name='us-east-1')


def test_s3_interaction(redshift, s3, read_test_frame):
    """
    Tests for upload_file() function. 
    """
    # Create bucket
    s3.create_bucket(Bucket='test_bucket')
    # Upload file to s3
    au.upload_file(
        'tests/test_data_frame.csv',
        'test_bucket',
        'testing',
        'testing'
    )
    # Get file from s3 bucket
    bucket_obj = s3.get_object(Bucket='test_bucket', Key='test_data_frame.csv')
    result = bucket_obj['Body'].read()
    # Credit to https://gist.github.com/ghandic/dbde264a0d666a415bbf1bdcc3645aec
    df = pd.read_csv(
        StringIO(result.decode('utf-8')),
        parse_dates=['date', 'date_with_na'],
        dtype={
            'int8': np.int8,
            'int16_missing': 'Int64',
            'int32': np.int32,
            'int64': np.int64,
            'float16': np.float16,
            'float32': np.float32,
            'float64': np.float64
        })

    # Test that the columns are consistent pre-and-post upload
    assert df.columns.tolist() == read_test_frame.columns.tolist()
    # Test that the shapes are consistent pre-and-post upload
    assert df.shape == read_test_frame.shape
    # Test that the data are consistent
    pd.testing.assert_frame_equal(
        df,
        read_test_frame
    )
