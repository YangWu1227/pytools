import pycitizen.aws_utils as au
import pandas as pd
from random import sample
import numpy as np
import os

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

    # ---------------------------------------------------------------------------- #
    #                                   Test data                                  #
    # ---------------------------------------------------------------------------- #

    df = pd.read_csv(
        '/Users/kenwu/Pypkg/pycitizen/tests/test_data_frame.csv',
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

    df['mix_numeric_none'].replace(to_replace={np.nan: None}, inplace=True)

au.create_statement(df, 'test', 'key')

au._datetime_cols(df)

au._float_cols(df)
