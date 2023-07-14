"""Utility functions for preprocessing data"""
import sqlite3 as sql3
import pandas as pd


def load_sample(percentage: float = 0.1,
                        csv_file: str = "data/train_full.csv",
                        sql_db: str = None
                        ) -> pd.DataFrame:
    """Load a sample from a data store"""

    df: pd.DataFrame = None
    if sql_db is not None:
        with sql3.connect(f"sqlite://{sql_db}") as db:
            df = pd.read_sql("SELECT * FROM data", db)
    else:
        df = pd.read_csv(csv_file)

    return df.sample(frac=percentage, axis=0).reset_index(drop=True)
