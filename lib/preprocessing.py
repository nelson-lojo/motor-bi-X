"""Utility functions for preprocessing data"""
import sqlite3 as sql3
import pandas as pd
from typing import Optional


def load_sample(percentage: float = 0.1,
                csv_file: str = "data/train_full.csv",
                sql_db: Optional[str] = None
                ) -> pd.DataFrame:
    """Load a sample from a data store"""

    df: Optional[pd.DataFrame] = None
    if sql_db is not None:
        with sql3.connect(sql_db) as db:
            df = pd.read_sql("SELECT * FROM data", db).drop(columns="index")
    else:
        df = pd.read_csv(csv_file)

    return df.sample(frac=percentage, axis=0, random_state=42).reset_index(drop=True)

def save_data(df: pd.DataFrame,
              csv_file: str = "saved.csv", 
              sql_db: Optional[str] = None
              ) -> None:
    
    if sql_db is not None:
        with sql3.connect(sql_db) as conn:
            conn.execute("DROP TABLE IF EXISTS data;")
            df.to_sql("data", conn)
    else:
        df.to_csv(csv_file)
