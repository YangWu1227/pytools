# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #


import pandas as pd
import numpy as np
import os
import sys
import threading
from pycitizen.exceptions import ColumnDtypeInferError
from collections.abc import Sequence, ByteString
import psycopg2 as py
import boto3
from botocore.exceptions import ClientError, NoCredentialsError


# --------------------------- SQL commands creation -------------------------- #

# ---------------------------------------------------------------------------- #
#         Function that finds the max string length in each text column        #
# ---------------------------------------------------------------------------- #


def _max_len_tbl(df):
    """
    This function takes a data frame object and finds the maximum string
    length in each text column.

    Args:
        df: DataFrame.

    Returns:
        Series: A pandas data frame containing max string lengths for all text columns in 'df.'
    """
    # Exclude the explicit non-object columns first
    # There may be so mixed type 'object' columns remaining that are really numeric columns
    df = df.select_dtypes(include=object)
    # The implementation below requires that missing values are represented as np.NaN rather than None
    # Now the mixed type 'object' columns with pd.None or pd.NA representing missing numeric values should become numeric types
    df_obj = df.replace(to_replace={None: np.NaN}, inplace=False)
    # Create a frame containing max string length for each object column
    max_len_frame = (pd.DataFrame(df_obj.select_dtypes(include=object)
                                  .replace(to_replace={np.NaN: None}, inplace=False)
                                  .apply(lambda col: col.str.len().max()))
                     .reset_index()
                     .rename(columns={'index': 'col', 0: 'dtype'})
                     .astype({'dtype': int}, copy=False)
                     .set_index('col'))
    return max_len_frame


# ---------------------------------------------------------------------------- #
#                      Function that finds float variables                     #
# ---------------------------------------------------------------------------- #


def _float_cols(df):
    """
    This function takes a data frame object and finds the float columns.

    Args:
        df: DataFrame.

    Returns:
        Series: A list of float column names in 'df.'
    """
    # The list comprehension returns a boolean list, which is then used to select the numeric columns
    float_cols = (df.iloc[:, [~pd.to_numeric(df[col], errors='coerce').isna().all() for col in df.columns.tolist()]]
                  .select_dtypes(include=[np.float16, np.float32, np.float64, np.float128])
                  .columns.tolist())
    return float_cols


# ---------------------------------------------------------------------------- #
#                    Function that finds datetime variables                    #
# ---------------------------------------------------------------------------- #


def _datetime_cols(df):
    """
    This function takes a data frame object and finds the datatime64 columns.

    Args:
        df: DataFrame.

    Returns:
        Series: A list of datetime64 column names in 'df.'
    """
    datetime_cols = (df.select_dtypes(include='datetime64')
                     .columns.tolist())
    return datetime_cols

# ---------------------------------------------------------------------------- #
#                  Function to create a CREATE TABLE statement                 #
# ---------------------------------------------------------------------------- #


def create_statement(df, tbl_name, primary_key):
    """
    This function generates a single CREATE TABLE statement given a data frame and a table name. The CREATE
    TABLE statement is used to stage a shell of a table into which data will be copied either from S3 or directly
    from pandas after cleaning. Currently, only columns with dtype `int` (8, 16, 32, 64), `float` (16, 32, 64, 128),
    `datetime64` or `object` can be inferred. The experimental `StringDtype` extension dtype for Pandas dataframe is
    not currently implemented. See the Pandas [documentation](https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes)
    on basic dtypes.

    Args:
        df: DataFrame.
        tbl_name (str): Name of the table to be created.
        primary_key (str): Primary key of the table. Note this column must be a string column and must not contain NULL values.

    Raises:
        TypeError: The argument 'df' must be a dataframe object and 'tbl_name' and 'primary_key' must be string objects.
        ColumnDtypeInferError: The argument 'df' contains one or more columns that cannot be inferred. Currently, only columns with dtype `int`, `float`, `datetime64` or `object` columns can be inferred.

    Returns:
        SQL commands [str]: A CREATE TABLE statement to be passed to create_tables().
    """
    if (not isinstance(df, pd.DataFrame)) or (not isinstance(tbl_name, str)) or (not isinstance(primary_key, str)):
        raise TypeError(
            "'df' must be a data frame and 'tbl_name' and 'primary_key' must be string objects")

    col_dtypes = (str(val) for val in set(df.dtypes.to_dict().values()))

    if not all((dtype in ('datetime64[ns]', 'object', 'int8', 'int16', 'int32', 'int64', 'Int64', 'float16', 'float32', 'float64', 'float128') for dtype in col_dtypes)):
        raise ColumnDtypeInferError(
            "'df' contains columns with dtypes that cannot be inferred see documentation for supported dtypes")

    # Create a dataframe containing all columns as rows in the order based on the order of columns in 'df'
    df_all = pd.DataFrame(
        {'col': df.columns}
    ).set_index('col')
    # Create a dataframe containing text columns in 'df' and each of their associated max string length
    # This dataframe has: 'col' (index for joining with df_all) and 'dtype' (max string length as np.int32)
    df_varchar = _max_len_tbl(df)
    # Format the 'dtype' column in 'df_varchar', e.g., 3 becomes 'VARCHAR(3),'
    df_varchar['dtype'] = df_varchar['dtype'].map(
        lambda x: 'VARCHAR(' + str(x) + ')'
    )
    # Create a list of float column names in 'df'
    float_col_nms = _float_cols(df)
    # Create a list of datetime column names in 'df'
    datetime_col_nms = _datetime_cols(df)
    # Format 'primary_key' to add 'NOT NULL,' in addition to 'VARCHAR(int)'
    df_varchar['dtype'] = np.where(
        df_varchar.index == primary_key,
        df_varchar.dtype + ' NOT NULL,',
        df_varchar.dtype + ','
    )
    # Left join 'df_varchar' onto 'df_all' on the index 'col', preserving the left (df_all) index order
    df_commands = (df_all.join(df_varchar, how='left', sort=False))
    # Specify float columns as 'DOUBLE PRECISION'
    df_commands.iloc[(
        col in float_col_nms for col in df.columns)] = 'REAL,'
    # Specify datetime columns as 'TIMESTAMPTZ'
    df_commands.iloc[(
        col in datetime_col_nms for col in df.columns)] = 'TIMESTAMPTZ,'
    # Reset index
    df_commands.reset_index(inplace=True)
    # Replace all NaN's with INTEGER (those are the non-text columns)
    df_commands.replace(to_replace={np.nan: 'INTEGER,'}, inplace=True)
    # Add a row to specify primary key column
    df_commands = df_commands.append(
        other={'col': 'PRIMARY KEY', 'dtype': f'({primary_key})'}, ignore_index=True)

    # Create lists for joining each pair of 'col' and 'dtype', e.g., 'col_1' + 'VARCHAR(3)'
    col_list, dtype_list = (df_commands.col.tolist(),
                            df_commands.dtype.tolist())
    # Create a list, each of whose element is a joined string, e.g., 'col_1 VARCHAR(3)'
    commands_list = [col + ' ' + dtype for col,
                     dtype in zip(col_list, dtype_list)]
    # Join elements of 'commands_list', using white space ' ' as a separator
    commands = ' '.join(commands_list)
    # Add 'CREATE tbl_name' and wrap commands in ()
    final_commands = f'CREATE TABLE {tbl_name} ({commands})'

    return final_commands


# ---------------------------------------------------------------------------- #
#                     Helper function for input validation                     #
# ---------------------------------------------------------------------------- #

def is_sequence(seq):
    """
    Returns `True` if the input is a collections.Sequence (except strings or bytestrings).

    Args:
      seq: an input sequence.

    Returns:
      `True` if the sequence is a collections.Sequence and not a string or bytestring.
    """
    return isinstance(seq, Sequence) and not isinstance(seq, (str, ByteString, range))

# ---------------------------------------------------------------------------- #
#             Function to generate multiple CREATE TABLE statements            #
# ---------------------------------------------------------------------------- #


def create_statements(df_seq, tbl_names, primary_keys):
    """
    This function is a vectorized version of `create_statement()`, which takes a sequence of
    data frames, a sequence of table names, and a sequence of primary keys, returning a tuple of
    CREATE TABLE statements. The output of this function can then be passed to `create_tables()`,
    which creates shell tables in the database with column data types specified.

    Args:
        df_seq (seq): A sequence of data frame objects.
        tbl_names (seq): A sequence of names of tables to be created.
        primary_keys (seq): A sequence of primary keys.

    Raises:
        TypeError: The arguments 'df_seq', 'tbl_names', and 'primary_keys' must be registered as Sequences.
        ValueError: The sequences 'df_seq', 'tbl_names', and 'primary_keys' must have equal lengths.

    Returns:
        List of SQL commands [tuple]: A tuple of CREATE TABLE statements to be passed to create_tables().
    """
    if not (is_sequence(df_seq) and is_sequence(tbl_names) and is_sequence(primary_keys)):
        raise TypeError(
            "'df_seq', 'tbl_names', and 'primary_keys' must be sequences like lists or tuples")
    if not len(df_seq) == len(tbl_names) == len(primary_keys):
        raise ValueError(
            "'df_seq', 'tbl_names', and 'primary_keys' must have equal lengths")
    # Generator comprehension
    tuple_of_commands = tuple((create_statement(df, tbl_name, primary_key)
                               for df, tbl_name, primary_key in zip(df_seq, tbl_names, primary_keys)))

    return tuple_of_commands


# ---------------------------------------------------------------------------- #
#                             AWS credentials class                            #
# ---------------------------------------------------------------------------- #

class AwsCreds(object):
    """
    A class for storing AWS access key and secret key.

    Attributes:
            access_key (str): Access key.
            secret_key (str): Secret key.

    Methods:
            get_params(): A method for obtaining credentials as a tuple that may be unpacked and passed as function arguments.
    """

    def __init__(self, access_key, secret_key):
        """
        A parameterized class constructor.

        Args:
            access_key (str): Access key.
            secret_key (str): Secret key.
        """
        self.access_key = access_key
        self.secret_key = secret_key

    def get_params(self):
        """
        A method for obtaining credentials as a tuple that may be unpacked and passed as function arguments.
        """
        return self.access_key, self.secret_key


# -------------------- Interacting with Redshift database -------------------- #

# ---------------------------------------------------------------------------- #
#                                Redshift class                                #
# ---------------------------------------------------------------------------- #


class MyRedShift(object):
    """
    A class for interacting with Redshift database.

    Attributes:
            db_name (str): Database name.
            host (str): Database host address.
            port (str): Connection port number.
            user (str): User name used to authenticate.
            db_password (str): Database password.

    Methods:
            get_params(): A method for obtaining connection parameters as a tuple that may be unpacked and passed as function arguments.
            connect(): A method for creating a database session and instantiating a connection object.
            read_tbl(): A method for reading a table into a Pandas DataFrame.
            read_query(): A method for reading a single query output into a Pandas DataFrame.
    """

    def __init__(self, db_name, host, port, user, db_password):
        """
        A parameterized class constructor

        Args:
            db_name (str): Database name.
            host (str): Database host address.
            port (str): Connection port number.
            user (str): User name used to authenticate.
            db_password (str): Database password.
        """
        self.db_name = db_name
        self.host = host
        self.port = port
        self.user = user
        self.db_password = db_password

    def get_params(self):
        """
        A method for obtaining connection parameters as a tuple that may be unpacked and passed as function arguments.
        """
        return self.db_name, self.host, self.port, self.user, self.db_password

    def connect(self):
        """
        A method for creating a database session and instantiating a connection object. The connection object will
        have access to all public methods and attributes of the psycopg2 connection class, including connection.close().
        """
        conn = py.connect(
            dbname=self.db_name,
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.db_password
        )
        return conn

    def read_tbl(self, tbl, chunksize):
        """
        A method for reading a table into a Pandas DataFrame.

        Args:
            tbl (str): Table name.
            chunksize (int): Number of rows to load into memory in each chunk.
        """
        # Returns a generator 'SQLiteDatabase._query_iterator'
        gen_obj = pd.read_sql_query(
            sql='SELECT * FROM {};'.format(tbl),
            con=self.connect(),
            chunksize=chunksize
        )
        return pd.concat(list(gen_obj))

    def read_query(self, sql, chunksize=None):
        """
        A method for reading a single sql query output into a Pandas DataFrame.

        Args:
            sql (str): A sql query.
            chunksize (int): Number of rows to load into memory in each chunk.
        """
        if chunksize is None:
            return pd.read_sql_query(
                sql=sql,
                con=self.connect()
            )
        else:
            gen_obj = pd.read_sql_query(
                sql=sql,
                con=self.connect(),
                chunksize=chunksize
            )
            return pd.concat(list(gen_obj))


# ---------------------------------------------------------------------------- #
#                        Helper function to get AWS keys                       #
# ---------------------------------------------------------------------------- #


def get_creds(path):
    """
    A helper function to retrieve and return AWS credentials in a tuple string form. Another method for getting aws credentials is through creating an `AwsCreds` objects. See `?AwsCreds` for details.
    To run this function, it is recommended that the user store his/her AWS credentials in a hidden directory, e.g., '~/.aws/credentials/accessKeys.csv'. The csv file should contain two columns--- 'Access key ID'
    and 'Secret access key'--- and a single row that records the AWS credentials. To be precise, users may store their AWS credentials anywhere, but the file type must be csv and its structure must follow the
    above recommendation--- two columns and a single row.

    Args:
        path (str): File path to credentials.

    Returns:
        tuple: (access_key, secret_key)
    """
    creds = pd.read_csv(path)
    secret_key = creds['Secret access key'].iloc[0]
    access_key = creds['Access key ID'].iloc[0]

    return access_key, secret_key


# ---------------------------------------------------------------------------- #
#                     Function to create tables in RedShift                    #
# ---------------------------------------------------------------------------- #


def create_tables(commands, db_name, host, port, user, db_password):
    """
    When passed a tuple of SQL commands, this function executes the commands and commits the
    changes to Redshift. For single table creation, use `(command, )` to pass a single-element
    tuple. For database connection parameters, you may store all parameters in a sequence container 
    and unpack the sequence so that all elements are passed as different parameters.

    Args:
        commands (tuple): A tuple of SQL commands. E.g.

            "CREATE TABLE ma_wick_11_02_20 (
                lalvoterid VARCHAR(14) NOT NULL,
                method_gen_2020 VARCHAR(29),
                bm2_support_gen_2020 INTEGER,
                bm2_pro_exposure_gen_2020 INTEGER,
                PRIMARY KEY (lalvoterid)
                )"
        db_name (str): Database name.
        host (str): Database host address.
        port (str): Connection port number.
        user (str): User name used to authenticate.
        db_password (str): Database password.

    Raises:
        TypeError: The argument 'commands' must be a tuple.
        TypeError: All SQL commands must be string objects wrapped by triple quotes.
    """
    # Check inputs
    if not isinstance(commands, tuple):
        raise TypeError("'commands' must be a tuple")
    if not all((isinstance(statement, str) for statement in commands)):
        raise TypeError(
            "All CREATE TABLE statements in 'commands' must be string objects")

    try:
        # Connection object
        conn = py.connect(
            dbname=db_name,
            host=host,
            port=port,
            user=user,
            password=db_password
        )
        # Cursor object
        cur = conn.cursor()
        # Create tables iteratively
        for command in commands:
            cur.execute(command)
        # Close cursor
        cur.close()
        # Commit changes
        conn.commit()
    except (Exception, py.DatabaseError, py.DataError, py.OperationalError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


# ---------------------------------------------------------------------------- #
#              Function to copy tables from S3 bucket to database              #
# ---------------------------------------------------------------------------- #

def copy_tables(table_names, paths, access_key, secret_key, db_name, host, port, user, db_password):
    """
    This function accepts a sequence of tables names and data-source paths and loads data into a table in
    the database hosted on Amazon Redshift. For database connection parameters, you may store all parameters
    in a sequence container and unpack the sequence so that all elements are passed as different parameters.

    Args:
        table_names (seq): The name of the target table for the COPY command. The table must already
        exist in the database. The table can be temporary or persistent. The COPY command appends the
        new input data to any existing rows in the table.
        paths (seq): The location of the source data to be loaded into the target table. A manifest file
        can be specified with some data sources. The most commonly used data repository is an Amazon S3 bucket.
        access_key (str): Access key.
        secret_key (str): Secret key.
        db_name (str): Database name.
        host (str): Database host address.
        port (str): Connection port number.
        user (str): User name used to authenticate.
        db_password (str): Database password.

    Raises:
        TypeError: The arguments 'table_names' and 'paths' must be registered as Sequences.
        ValueError: The sequences 'table_names' and 'paths' must have equal lengths.
    """
    # Check input
    if not (is_sequence(table_names) and is_sequence(paths)):
        raise TypeError(
            "'table_names' and 'paths' must be sequences like lists or tuples")
    if not len(table_names) == len(paths):
        raise ValueError(
            "'table_names' and 'paths' must have equal lengths")

    try:
        # Connection object
        conn = py.connect(
            dbname=db_name,
            host=host,
            port=port,
            user=user,
            password=db_password
        )
        # Cursor object
        cur = conn.cursor()
        # Copy tables iteratively
        for table_name, path in zip(table_names, paths):
            cur.execute(
                f'''
                COPY {table_name}
                FROM '{path}'
                CSV
                IGNOREHEADER 1
                FILLRECORD
                EMPTYASNULL
                ACCESS_KEY_ID '{access_key}'
                SECRET_ACCESS_KEY '{secret_key}'
                '''
            )
        # Close cursor
        cur.close()
        # Commit changes
        conn.commit()
    except (Exception, py.DatabaseError, py.DataError, NoCredentialsError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


# ---------------------------------------------------------------------------- #
#               Function to rename table columns in the database               #
# ---------------------------------------------------------------------------- #


def rename_col(tbl_names, old_nms, new_nms, db_name, host, port, user, db_password):
    """
    This function accepts three sequences of equal lengths, executing the ALTER TABLE RENAME COLUMN statements in the database. The arguments must sequences like a `list` or
    `tuple`. Each element in the three sequences must match in order for the query to be executed successfully. For database connection
    parameters, you may store all parameters in a sequence container and unpack the sequence so that all elements are passed as different parameters.

    Args:
        tbl_names (seq): An sequence containing table names.
        old_nms (seq): An sequence containing original column names.
        new_nms (seq): An sequence containing new column names.
        db_name (str): Database name.
        host (str): Database host address.
        port (str): Connection port number.
        user (str): User name used to authenticate.
        db_password (str): Database password.

    Raises:
        TypeError: The arguments 'tbl_names', 'old_nms', and 'new_nms' must be registered as Sequences.
        ValueError: The sequences 'tbl_names', 'old_nms', and 'new_nms' must have equal lengths.
    """
    # Check input
    if not (is_sequence(tbl_names) and is_sequence(old_nms) and is_sequence(new_nms)):
        raise TypeError(
            "'tbl_names', 'old_nms', and 'new_nms' must be sequences like lists or tuples")
    if not len(tbl_names) == len(old_nms) == len(new_nms):
        raise ValueError(
            "'tbl_names', 'old_nms', and 'new_nms' must have equal lengths")

    try:
        # Connection object
        conn = py.connect(
            dbname=db_name,
            host=host,
            port=port,
            user=user,
            password=db_password
        )
        # Cursor object
        cur = conn.cursor()
        # Copy tables iteratively
        for tbl_name, old_nm, new_nm in zip(tbl_names, old_nms, new_nms):
            cur.execute(
                f'''
                ALTER TABLE {tbl_name}
                RENAME COLUMN {old_nm} TO {new_nm}
                '''
            )
        # Close cursor
        cur.close()
        # Commit changes
        conn.commit()
    # Catch InvalidColumnReference or UndefinedTable with py.ProgrammingError class
    except (Exception, py.DatabaseError, py.DataError, py.ProgrammingError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


# ---------------------------------------------------------------------------- #
#                   Function to rename tables in the database                  #
# ---------------------------------------------------------------------------- #


def rename_tbl(old_nms, new_nms, db_name, host, port, user, db_password):
    """
    This function accepts two sequences of equal lengths, executing the ALTER TABLE RENAME TO statements in the database. The arguments must sequences like a `list` or
    `tuple`. Each element in the two sequences must match in order for the query to be executed successfully. For database connection
    parameters, you may store all parameters in a sequence container and unpack the sequence so that all elements are passed as different parameters.

    Args:
        old_nms (seq): An sequence containing original table names.
        new_nms (seq): An sequence containing new table names.
        db_name (str): Database name.
        host (str): Database host address.
        port (str): Connection port number.
        user (str): User name used to authenticate.
        db_password (str): Database password.

    Raises:
        TypeError: The arguments 'old_nms' and 'new_nms' must be registered as Sequences.
        ValueError: The sequences 'old_nms' and 'new_nms' must have equal lengths.
    """
    # Check input
    if not (is_sequence(old_nms) and is_sequence(new_nms)):
        raise TypeError(
            "'old_nms' and 'new_nms' must be sequences like lists or tuples")
    if not len(old_nms) == len(new_nms):
        raise ValueError(
            "'old_nms' and 'new_nms' must have equal lengths")

    try:
        # Connection object
        conn = py.connect(
            dbname=db_name,
            host=host,
            port=port,
            user=user,
            password=db_password
        )
        # Cursor object
        cur = conn.cursor()
        # Copy tables iteratively
        for old_nm, new_nm in zip(old_nms, new_nms):
            cur.execute(
                f'''
                ALTER TABLE {old_nm} RENAME TO {new_nm}
                '''
            )
        # Close cursor
        cur.close()
        # Commit changes
        conn.commit()
    # Catch UndefinedTable with py.ProgrammingError class
    except (Exception, py.DatabaseError, py.DataError, py.ProgrammingError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


# ------------------------ Interacting with S3 storage ----------------------- #

# ---------------------------------------------------------------------------- #
#                Class ProcessPercentage for progress monitoring               #
# ---------------------------------------------------------------------------- #


class ProgressPercentage(object):
    """
    A progress tracker that prints a simple progress percentage to the user. For more
    information, please refer to the `Boto3` documentations.
    """

    # Class constructor
    def __init__(self, filename):
        """
        A parameterized class constructor.

        Args:
            filename (str): The path and file name of the object.
        """
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._bytes_seen_so_far = 0
        # Create new lock object
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._bytes_seen_so_far += bytes_amount
            percentage = (self._bytes_seen_so_far / self._size) * 100
            sys.stdout.write(
                # Use \r for carriage return (resets to begining of a line of text)
                # Use %s as string of characters
                # A % followed by another % character will write a single %
                # So %.2f%% prints, e.g., 23.43% and use 'f' for float
                # This will print e.g. '**/filename.csv 3452 / 24356 (0.1417....)' to the screen
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._bytes_seen_so_far, self._size,
                    percentage))
            # Calling sys.stdout.flush() forces sys.stdout.write() to “flush”
            # That is, write the output on the screen on each call and do not buffer it (which store all output and print it at once)
            sys.stdout.flush()


# ---------------------------------------------------------------------------- #
#                     Function to upload file to s3 bucket                     #
# ---------------------------------------------------------------------------- #


def upload_file(file_name, bucket, access_key, secret_key, object_name=None):
    """
    This function uploads a file to an S3 bucket. User must also provide the file path to his or her AWS credentials.
    Read `?get_creds` or `?AwsCreds` for details on storing AWS credentials.

    Args:
        file_name (str): The path and file name of the object to upload.
        bucket (str): The name of the bucket.
        access_key (str): Access key.
        secret_key (str): Secret key.
        object_name (str, optional): Name for the S3 object that will be created by uploading this file. Defaults to `None`, which uses the filename.
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Create a service client
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    # Upload the file
    try:
        s3_client.upload_file(
            file_name,
            bucket,
            object_name,
            Callback=ProgressPercentage(file_name)
        )
        # User-facing message
        print(f'\nObject {object_name} uploaded to bucket {bucket}')
    except (ClientError, FileNotFoundError, NoCredentialsError) as error:
        print(error)
