import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def get_minute_from_dt_series(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series).dt.hour * 60 + pd.to_datetime(series).dt.minute

def get_seconds_from_dt_series(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series).dt.hour * 3600 + pd.to_datetime(series).dt.minute * 60 + pd.to_datetime(series).dt.second

def drops(df: pd.DataFrame) -> pd.DataFrame:
    """Drop `Precipitation in millimeters` due to nullity 
       and `Arrival at Destination - Time` due to redundancy"""
    return df.drop(columns=[
        "Precipitation in millimeters",
        'Arrival at Destination - Time',
        'Arrival at Destination - Day of Month',
        'Arrival at Destination - Weekday (Mo = 1)'
    ])

def combine_weekdays(df: pd.DataFrame) -> pd.DataFrame:
    """Combine `Arrival at Destination - Weekday (Mo = 1)`, 
       `Arrival at Destination - Day of Month`, `Pickup - Weekday (Mo = 1)`, 
       `Pickup - Day of Month`, `Arrival at Pickup - Weekday (Mo = 1)`, 
       `Arrival at Pickup - Day of Month`, `Confirmation - Weekday (Mo = 1)`, and
       `Confirmation - Day of Month'` into `Fulfillment - Weekday (Su = 0)` and 
       `Fulfillment - Day of Month`
       """

    df['Fulfillment - Weekday (Su = 0)'] = df['Pickup - Weekday (Mo = 1)'] % 7
    df['Fulfillment - Day of Month'] = df['Pickup - Day of Month']

    return df.drop(
        columns=[
            # 'Arrival at Destination - Weekday (Mo = 1)', 
            # 'Arrival at Destination - Day of Month',
            'Pickup - Weekday (Mo = 1)',
            'Pickup - Day of Month',
            'Arrival at Pickup - Weekday (Mo = 1)',
            'Arrival at Pickup - Day of Month',
            'Confirmation - Weekday (Mo = 1)',
            'Confirmation - Day of Month'  
        ]
    )

def impute_temperature(df: pd.DataFrame) -> Tuple[pd.DataFrame, IterativeImputer]:
    """Use IterativeImputer to impute the `Temperature` column using the following columns:
        `Placement - Day of Month`, `Placement - Weekday (Mo = 1)`, `Placement - Time`,
        `Pickup Long`, `Pickup Lat`,
        `Destination Long`, `Destination Lat`
        `Distance (KM)`, `Time from Pickup to Arrival`.
    """

    other_series = get_minute_from_dt_series(df["Placement - Time"])
    temp_control = pd.concat(
        [
            df[[
                "Placement - Day of Month", 
                "Placement - Weekday (Mo = 1)", 
                "Pickup Long", 
                "Pickup Lat",
                "Destination Long",
                "Destination Lat",
                "Distance (KM)",
                "Temperature"
            ]],
            other_series
        ],
        axis=1
    )

    imputer = IterativeImputer(random_state=42)
    imputer.fit(temp_control.to_numpy())
    df['Temperature'] = imputer.transform(temp_control)[:, -2]

    return df, imputer


dont_scale = ["Order_No", "User_Id", "Rider_Id", "Business", "platform_3", 'Time from Pickup to Arrival']

def prep_train():
    """ oof """
    df = pd.read_csv("data/train_full.csv")
    df = drops(df)
    df = combine_weekdays(df)
    df, imputer = impute_temperature(df)

    for i in ['Order No', 'User Id', "Rider Id"]:
        under = i.replace(' ', '_')
        df.rename(columns={i: under}, inplace=True)
        df[under] = df[under].str.replace(f'{under}_', '')
        df[under] = df[under].astype(int)    
    
    riders = pd.read_csv("data/riders.csv")

    riders.rename(columns= {
        "Rider Id": "id",
        "No_Of_Orders": "orders",
        "Age": "age",
        "Average_Rating": "average_rating",
        "No_of_Ratings": "number_rating" 
    }, inplace=True)

    riders.id.replace('Rider_Id_', ' ',regex=True,inplace=True)

    riders["id"] = riders["id"].astype(int)
    riders.set_index("id", inplace=True)

    df = df.join(riders, on="Rider_Id", rsuffix="_rider")


    avg_kms = df['Distance (KM)'] / df['Time from Pickup to Arrival']

    speeders = df[avg_kms > 150/3600]
    snails = df[avg_kms < 4/3600]
    speed_anomalies = pd.concat([speeders, snails])

    df = df.drop(speed_anomalies.index)


    df["Placement - Time"] = get_seconds_from_dt_series(df['Placement - Time'])
    df["Confirmation - Time"] = get_seconds_from_dt_series(df['Confirmation - Time'])
    df["Arrival at Pickup - Time"] = get_seconds_from_dt_series(df['Arrival at Pickup - Time'])
    df["Pickup - Time"] = get_seconds_from_dt_series(df['Pickup - Time'])
    df["Business"] = (df['Personal or Business'] == "Business").astype(float)
    df["place_to_confirm"] = df["Confirmation - Time"] - df["Placement - Time"]
    df["confirm_to_pick_arr"] = df["Arrival at Pickup - Time"] - df["Confirmation - Time"]
    df["pick_arr_to_pick"] = df['Pickup - Time'] - df["Arrival at Pickup - Time"]
    df['platform_3'] = (df['Platform Type'] == 3).astype(float)
    df.drop(columns=["Personal or Business", "Vehicle Type", "Platform Type"], inplace=True)

    assert df.columns.isin(["Arrival at Destination - Day of Month", "Arrival at Destination - Weekday (Mo = 1)"]).sum() == 0, df.columns


    scaler = StandardScaler()
    s = scaler.fit(df[df.columns[~df.columns.isin(dont_scale)]])

    df = pd.concat(
        [
            pd.DataFrame(
                scaler.transform(df[df.columns[~df.columns.isin(dont_scale)]]), 
                columns = df.columns[~df.columns.isin(dont_scale)]
            ),
            df[df.columns[df.columns.isin(dont_scale)]]
        ], 
        join="inner",
        axis=1
    )

    df.rename(columns={
        "Rider Id": "rider_id",
        "User_Id": "user_id",
        "Order_Id": "order_id"
    }, inplace=True)

    return df, imputer, s


def prep_test(imputer, scaler):
    df = pd.read_csv("data/test.csv")



    # df = drops(df)
    df = combine_weekdays(df)

    # Impute Temperature
    other_series = get_minute_from_dt_series(df["Placement - Time"])
    temp_control = pd.concat(
        [
            df[[
                "Placement - Day of Month", 
                "Placement - Weekday (Mo = 1)", 
                "Pickup Long", 
                "Pickup Lat",
                "Destination Long",
                "Destination Lat",
                "Distance (KM)",
                "Temperature"
            ]],
            other_series
        ],
        axis=1
    )
    df['Temperature'] = imputer.transform(temp_control)[:, -2]

    # Renames
    for i in ['Order No', 'User Id', "Rider Id"]:
        under = i.replace(' ', '_')
        df.rename(columns={i: under}, inplace=True)
        df[under] = df[under].str.replace(f'{under}_', '')
        df[under] = df[under].astype(int)    
    
    # Rider Join
    riders = pd.read_csv("data/riders.csv")
    riders.rename(columns= {
        "Rider Id": "id",
        "No_Of_Orders": "orders",
        "Age": "age",
        "Average_Rating": "average_rating",
        "No_of_Ratings": "number_rating" 
    }, inplace=True)
    riders.id.replace('Rider_Id_', ' ',regex=True,inplace=True)
    riders["id"] = riders["id"].astype(int)
    riders.set_index("id", inplace=True)
    df = df.join(riders, on="Rider_Id", rsuffix="_rider")

    # Speed culling
    # avg_kms = df['Distance (KM)'] / df['Time from Pickup to Arrival']
    # speeders = df[avg_kms > 150/3600]
    # snails = df[avg_kms < 4/3600]
    # speed_anomalies = pd.concat([speeders, snails])
    # df = df.drop(speed_anomalies.index)


    df["Placement - Time"] = get_seconds_from_dt_series(df['Placement - Time'])
    df["Confirmation - Time"] = get_seconds_from_dt_series(df['Confirmation - Time'])
    df["Arrival at Pickup - Time"] = get_seconds_from_dt_series(df['Arrival at Pickup - Time'])
    df["Pickup - Time"] = get_seconds_from_dt_series(df['Pickup - Time'])
    df["Business"] = (df['Personal or Business'] == "Business").astype(float)
    df["place_to_confirm"] = df["Confirmation - Time"] - df["Placement - Time"]
    df["confirm_to_pick_arr"] = df["Arrival at Pickup - Time"] - df["Confirmation - Time"]
    df["pick_arr_to_pick"] = df['Pickup - Time'] - df["Arrival at Pickup - Time"]
    df['platform_3'] = (df['Platform Type'] == 3).astype(float)
    df.drop(columns=["Personal or Business", "Vehicle Type", "Platform Type"], inplace=True)


    df = pd.concat(
        [
            pd.DataFrame(
                scaler.transform(df[df.columns[df.columns.isin(scaler.feature_names_in_)]]), 
                columns = df.columns[df.columns.isin(scaler.feature_names_in_)]
            ),
            df[df.columns[df.columns.isin(dont_scale)]]
        ], 
        join="inner",
        axis=1
    )

    df.rename(columns={
        "Rider Id": "rider_id",
        "User_Id": "user_id",
        # "Order_No": "order_no"
    }, inplace=True)

    return df