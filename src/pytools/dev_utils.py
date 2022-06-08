# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #

import psycopg2 as py
import pandas as pd
import numpy as np

# ----------------------------- Standard library ----------------------------- #

import os

# ------------------------------- Intra-package ------------------------------ #

import pytools.aws_utils as au
from pytools.exceptions import ColumnDtypeInferError

if __name__ == '__main__':

    # ---------------------------------------------------------------------------- #
    #                                Redshift class                                #
    # ---------------------------------------------------------------------------- #

    db = au.MyRedShift(
        os.environ.get('DB_NAME'),
        os.environ.get('DB_HOST'),
        os.environ.get('DB_PORT'),
        os.environ.get('DB_USER'),
        os.environ.get('DB_PASSWORD')
    )

    db_params = db.get_params()

    # ---------------------------------------------------------------------------- #
    #                             AWS credentials class                            #
    # ---------------------------------------------------------------------------- #

    creds = au.AwsCreds(
        os.environ.get('Aws_access'),
        os.environ.get('Aws_secret')
    ).get_params()

    # ---------------------------------------------------------------------------- #
    #                                   Test data                                  #
    # ---------------------------------------------------------------------------- #

    df = pd.read_csv(
        os.getcwd().replace('src/pycitizen', '') + 'tests/test_data_frame.csv',
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
