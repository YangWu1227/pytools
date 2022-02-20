import psycopg2 as py
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

    db_params = db.get_params()

    # ---------------------------------------------------------------------------- #
    #                             AWS credentials class                            #
    # ---------------------------------------------------------------------------- #

    creds = au.AwsCreds(
        os.environ.get('Aws_access'),
        os.environ.get('Aws_secret')
    ).get_params()
