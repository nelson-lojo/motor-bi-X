import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def drops(df: pd.DataFrame) -> pd.DataFrame:
    """Drop `Precipitation in millimeters` due to nullity 
       and `Arrival at Destination - Time` due to redundancy"""
    return df.drop(columns=[
        "Precipitation in millimeters",
        'Arrival at Destination - Time'
    ])

def combine_weekdays(df: pd.DataFrame) -> pd.DataFrame:
    """Combine `Arrival at Destination - Weekday (Mo = 1)`, 
       `Arrival at Destination - Day of Month`, `Pickup - Weekday (Mo = 1)`, 
       `Pickup - Day of Month`, `Arrival at Pickup - Weekday (Mo = 1)`, 
       `Arrival at Pickup - Day of Month`, `Confirmation - Weekday (Mo = 1)`, and
       `Confirmation - Day of Month'` into `Fulfillment - Weekday (Su = 0)` and 
       `Fulfillment - Day of Month`
       """

    df['Fulfillment - Weekday (Su = 0)'] = df['Arrival at Destination - Weekday (Mo = 1)'] % 7
    df['Fulfillment - Day of Month'] = df['Arrival at Destination - Day of Month']

    return df.drop(
        columns=[
            'Arrival at Destination - Weekday (Mo = 1)', 
            'Arrival at Destination - Day of Month',
            'Pickup - Weekday (Mo = 1)',
            'Pickup - Day of Month',
            'Arrival at Pickup - Weekday (Mo = 1)',
            'Arrival at Pickup - Day of Month',
            'Confirmation - Weekday (Mo = 1)',
            'Confirmation - Day of Month'  
        ]
    )

def impute_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """Use IterativeImputer to impute the `Temperature` column using the following columns:
        `Placement - Day of Month`, `Placement - Weekday (Mo = 1)`, `Placement - Time`,
        `Pickup Long`, `Pickup Lat`,
        `Destination Long`, `Destination Lat`
        `Distance (KM)`, `Time from Pickup to Arrival`.
    """

    def get_minute_from_dt_series(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series).dt.hour * 60 + pd.to_datetime(series).dt.minute

    other_series = get_minute_from_dt_series(df["Placement - Time"])
    temp_control = pd.concat([
        df[[
            "Placement - Day of Month", 
            "Placement - Weekday (Mo = 1)", 
            "Pickup Long", 
            "Pickup Lat",
            "Destination Long",
            "Destination Lat",
            "Distance (KM)",
            "Time from Pickup to Arrival",
            "Temperature"
        ]],
        pd.DataFrame(other_series)
    ])

    imputer = IterativeImputer(random_state=42)
    imputer.fit(temp_control.to_numpy())
    df['Temperature'] = imputer.transform(temp_control)[:, -1]
    del imputer

    return df

