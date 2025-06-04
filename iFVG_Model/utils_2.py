import datetime as dt
import polars as pl
import os
import pandas as pd
# import numpy as np
import mplfinance as mpf
import pytz


def create_and_preprocess_trading_data(start_date, end_date, ET, folder_path = 'data/data'):
    date_range = pd.date_range(start=start_date, end=end_date)
    date_list = date_range.strftime('%Y%m%d').tolist()
    file_names = [f"trades-{date}.parquet" for date in date_list]
    existing_files = [file for file in file_names if os.path.exists(os.path.join(folder_path, file))]
    # print(existing_files)
    # for file in existing_files:
    #     df = pl.read_parquet(os.path.join(folder_path, file))
    #     print(f"Columns in {file}: {df.columns}")
    df = pl.concat([pl.read_parquet(os.path.join(folder_path, file)) for file in existing_files])


    if ET:
        # Convert the datetime columns from UTC to Eastern Time
        df = df.with_columns([
            pl.col('ts_recv').dt.convert_time_zone('US/Eastern').alias('ts_recv'),
            pl.col('ts_event').dt.convert_time_zone('US/Eastern').alias('ts_event')
        ])
        df = df.with_columns(
            pl.col('ts_event').dt.strftime('%H:%M:%S').alias('time')
        )
        df = df.with_columns(
            pl.col("time").cast(pl.Utf8)
        )
        df = df.with_columns(
        pl.col('ts_event').dt.date().alias('date')
        )
        df = df.with_columns([
            (((pl.col("ts_event").dt.hour() > 9) |
            ((pl.col("ts_event").dt.hour() == 9) & (pl.col("ts_event").dt.minute() >= 30))) &
            (pl.col("ts_event").dt.hour() < 16)).alias("is_cash_hour")
        ])
        

    else:
        df = df.with_columns(
            pl.col('ts_event').dt.strftime('%H:%M:%S').alias('time')
        )
        df = df.with_columns(
            pl.col("time").cast(pl.Utf8)
        )
        df = df.with_columns(
        pl.col('ts_event').dt.date().alias('date')
        )
        df = df.with_columns([
            (((pl.col("ts_event").dt.hour() > 9) |
            ((pl.col("ts_event").dt.hour() == 9) & (pl.col("ts_event").dt.minute() >= 30))) &
            (pl.col("ts_event").dt.hour() < 16)).alias("is_cash_hour")
        ])
    df = df.with_columns(pl.col('date').dt.weekday().alias('weekday'))  # Extract the day of the week (0=Monday, 6=Sunday)
    df = df.filter(pl.col('weekday') != 6)
    df = df.drop('weekday')

    return df

def get_contract_for_date(date, asset = 'ES'):
    """
    Returns the contract based on the provided date.

    Parameters:
    date (str): The date for which the contract is determined (in 'YYYY-MM-DD' format).

    Returns:
    str: The contract code ('ESH4', 'ESM4', 'ESU4', 'ESZ4').
    """
    if isinstance(date, dt.datetime):
        date = date.strftime('%Y-%m-%d')  # Convert datetime to string
    
    # Ensure the date is a string in 'YYYY-MM-DD' format
    if isinstance(date, str):
        date_obj = dt.datetime.strptime(date, "%Y-%m-%d")
    else:
        raise ValueError("The 'date' must be either a string or datetime object")

    # Convert the date string to a datetime object
    date_obj = dt.datetime.strptime(date, "%Y-%m-%d")
    year = date_obj.year

    if asset == 'ES' and year == 2024:
        # Define contract date ranges, adjusting for year changes
        if dt.datetime(date_obj.year, 12, 16) <= date_obj <= dt.datetime(date_obj.year + 1, 3, 15):
            return "ESH4"  # December 16th to March 15th
        elif dt.datetime(date_obj.year, 3, 16) <= date_obj <= dt.datetime(date_obj.year, 6, 15):
            return "ESM4"  # March 16th to June 15th
        elif dt.datetime(date_obj.year, 6, 16) <= date_obj <= dt.datetime(date_obj.year, 9, 15):
            return "ESU4"  # June 16th to September 15th
        elif dt.datetime(date_obj.year, 9, 16) <= date_obj <= dt.datetime(date_obj.year + 1, 12, 15):
            return "ESZ4"  # September 16th to December 15th
        else:
            return "Invalid Date"  # If the date does not fall within any of the ranges
    elif asset == 'NQ' and year == 2024:
        # Define contract date ranges, adjusting for year changes
        if dt.datetime(date_obj.year, 12, 16) <= date_obj <= dt.datetime(date_obj.year + 1, 3, 15):
            return "NQH4"  # December 16th to March 15th
        elif dt.datetime(date_obj.year, 3, 16) <= date_obj <= dt.datetime(date_obj.year, 6, 15):
            return "NQM4"  # March 16th to June 15th
        elif dt.datetime(date_obj.year, 6, 16) <= date_obj <= dt.datetime(date_obj.year, 9, 15):
            return "NQU4"  # June 16th to September 15th
        elif dt.datetime(date_obj.year, 9, 16) <= date_obj <= dt.datetime(date_obj.year + 1, 12, 15):
            return "NQZ4"  # September 16th to December 15th
        else:
            return "Invalid Date"  # If the date does not fall within any of the ranges
    elif asset == 'ES' and year == 2023:
        if dt.datetime(date_obj.year, 12, 16) <= date_obj <= dt.datetime(date_obj.year + 1, 3, 15):
            return "ESH3"  # December 16th to March 15th
        elif dt.datetime(date_obj.year, 3, 16) <= date_obj <= dt.datetime(date_obj.year, 6, 15):
            return "ESM3"  # March 16th to June 15th
        elif dt.datetime(date_obj.year, 6, 16) <= date_obj <= dt.datetime(date_obj.year, 9, 15):
            return "ESU3"  # June 16th to September 15th
        elif dt.datetime(date_obj.year, 9, 16) <= date_obj <= dt.datetime(date_obj.year + 1, 12, 15):
            return "ESZ3"  # September 16th to December 15th
        else:
            return "Invalid Date"  # If the date does not fall within any of the ranges
    elif asset == 'NQ' and year == 2023:
        # Define contract date ranges, adjusting for year changes
        if dt.datetime(date_obj.year, 12, 16) <= date_obj <= dt.datetime(date_obj.year + 1, 3, 15):
            return "NQH3"  # December 16th to March 15th
        elif dt.datetime(date_obj.year, 3, 16) <= date_obj <= dt.datetime(date_obj.year, 6, 15):
            return "NQM3"  # March 16th to June 15th
        elif dt.datetime(date_obj.year, 6, 16) <= date_obj <= dt.datetime(date_obj.year, 9, 15):
            return "NQU3"  # June 16th to September 15th
        elif dt.datetime(date_obj.year, 9, 16) <= date_obj <= dt.datetime(date_obj.year + 1, 12, 15):
            return "NQZ3"  # September 16th to December 15th
        else:
            return "Invalid Date"  # If the date does not fall within any of the ranges
    elif asset == 'ES' and year == 2025:
        if dt.datetime(date_obj.year, 12, 16) <= date_obj <= dt.datetime(date_obj.year + 1, 3, 15):
            return "ESH5"  # December 16th to March 15th
        elif dt.datetime(date_obj.year, 3, 16) <= date_obj <= dt.datetime(date_obj.year, 6, 15):
            return "ESM5"  # March 16th to June 15th
        elif dt.datetime(date_obj.year, 6, 16) <= date_obj <= dt.datetime(date_obj.year, 9, 15):
            return "ESU5"  # June 16th to September 15th
        elif dt.datetime(date_obj.year, 9, 16) <= date_obj <= dt.datetime(date_obj.year + 1, 12, 15):
            return "ESZ5"  # September 16th to December 15th
        else:
            return "Invalid Date"  # If the date does not fall within any of the ranges
    elif asset == 'NQ' and year == 2025:
        # Define contract date ranges, adjusting for year changes
        if dt.datetime(date_obj.year, 12, 16) <= date_obj <= dt.datetime(date_obj.year + 1, 3, 15):
            return "NQH5"  # December 16th to March 15th
        elif dt.datetime(date_obj.year, 3, 16) <= date_obj <= dt.datetime(date_obj.year, 6, 15):
            return "NQM5"  # March 16th to June 15th
        elif dt.datetime(date_obj.year, 6, 16) <= date_obj <= dt.datetime(date_obj.year, 9, 15):
            return "NQU5"  # June 16th to September 15th
        elif dt.datetime(date_obj.year, 9, 16) <= date_obj <= dt.datetime(date_obj.year + 1, 12, 15):
            return "NQZ5"  # September 16th to December 15th
        else:
            return "Invalid Date"  # If the date does not fall within any of the ranges


    

def olhcv_data(df, date, timeframe='5m', asset = 'ES'):
    # Filter data for the given date
    if isinstance(date, dt.datetime):  # Check if 'date' is a datetime object
        date = date.strftime("%Y-%m-%d")    
    # Now date is guaranteed to be a string in the format 'YYYY-MM-DD'
    date = dt.datetime.strptime(date, "%Y-%m-%d")
    subset_df = df.filter(df["date"] == pl.lit(date).cast(pl.Date))
    contract = get_contract_for_date(date, asset)

    subset_df = subset_df.filter(pl.col("symbol") == pl.lit(contract))
    ohlc_df = subset_df.group_by_dynamic(
        "ts_event", every=timeframe, closed="left", group_by="symbol", label='left'
    ).agg([
        pl.col("ts_event").dt.truncate(every=timeframe).first().cast(pl.Int64).alias("interval_start_ns"),
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("size").sum().alias("volume"),
        pl.when(pl.col("side") == "B").then(pl.col("size")).sum().alias("bid_volume"),
        pl.when(pl.col("side") == "A").then(pl.col("size")).sum().alias("ask_volume"),
        pl.when(pl.col("side") == "N").then(pl.col("size")).sum().alias("n_volume"),
        pl.len().alias("transactions"),
        # Create 'interval_time' from 'ts_event' and remove 'ts_event' and 'interval_start'
        pl.col("ts_event").dt.truncate(every=timeframe).first().alias("interval_time")
    ])
    # Create a new column 'time' that contains just the time (HH:MM:SS) with zero-padding
    ohlc_df = ohlc_df.with_columns(
        (
            pl.col("interval_time").dt.hour().cast(pl.String).str.zfill(2) + ":" +
            pl.col("interval_time").dt.minute().cast(pl.String).str.zfill(2) + ":" +
            pl.col("interval_time").dt.second().cast(pl.String).str.zfill(2)
        ).alias("time")
    )

    # Reorder columns so that 'time' comes first
    ohlc_df = ohlc_df.select(["time"] + [col for col in ohlc_df.columns if col != "time"])

    # Drop ts_event and interval_start
    ohlc_df = ohlc_df.drop(["ts_event", "interval_start_ns"])

    ohlc_df = ohlc_df.with_columns(
    pl.col("time").cast(pl.Utf8)
    )
   
    return ohlc_df

def contract_size(stop_loss_size, risk=1000, asset = 'ES', dynamic_position_sizing=False, risk_percent=0.02, total_capital=None):
    if dynamic_position_sizing:
        risk = risk_percent * total_capital
    if asset == 'ES':
        number_of_contracts = round(risk/(50 * stop_loss_size))
    elif asset == 'NQ':
        number_of_contracts = round(risk/(20 * stop_loss_size))
    return number_of_contracts





def visualize_candlesticks(start_datetime, end_datetime, df, timeframe='5m', asset = 'ES'):
    """
    Visualizes the candlestick chart for a given time range by fetching data for all days between the start and end date.

    Parameters:
    start_datetime (datetime): The start datetime for the chart.
    end_datetime (datetime): The end datetime for the chart.
    df (DataFrame): DataFrame containing historical trading data.
    timeframe (str): The timeframe for the candlestick data (e.g., '1m', '5m', '1d', etc.)

    Returns:
    None: Displays the candlestick chart for the given time range.
    """

    # Convert the start and end datetimes to timezone-aware datetimes (UTC in this case)
    tz = pytz.UTC
    start_datetime = tz.localize(start_datetime)
    end_datetime = tz.localize(end_datetime)

    # Generate a list of all days in the range from start_datetime to end_datetime
    date_range = pd.date_range(start=start_datetime, end=end_datetime, freq='D').date

    # Initialize an empty DataFrame to collect all the data for the days in the range
    combined_data = pd.DataFrame()

    # Fetch and concatenate data for each day in the date range
    for date in date_range:
        olhc_data = olhcv_data(df, str(date), timeframe=timeframe, asset=asset)
        olhc_data_to_pandas = olhc_data.to_pandas()
        combined_data = pd.concat([combined_data, olhc_data_to_pandas])

        if olhc_data_to_pandas.empty:
            print(f"No data found for {date}.")
            continue

    # Check if any data is present after fetching
    if combined_data.empty:
        return {"No valid data available for the selected time range."}

    if combined_data['interval_time'].dt.tz is None:
        # If 'interval_time' is not timezone-aware, localize it
        combined_data['interval_time'] = combined_data['interval_time'].dt.tz_localize(tz)
    else:
        # If 'interval_time' is already timezone-aware, convert to the same timezone as start_datetime
        combined_data['interval_time'] = combined_data['interval_time'].dt.tz_convert(tz)
    
    # Filter the data for the given time range
    filtered_data = combined_data[(combined_data['interval_time'] >= start_datetime) & 
                                  (combined_data['interval_time'] <= end_datetime)]

    # Check if the filtered data is empty after filtering
    if filtered_data.empty:
        return {"No data found for the specified time range."}

    # Prepare the data for mplfinance (only OHLC data)
    filtered_data = filtered_data.set_index('interval_time')
    filtered_data = filtered_data[['open', 'high', 'low', 'close']]  # Only need OHLC for mplfinance

    # Plot the candlestick chart
    mpf.plot(filtered_data, type='candle', style='charles', 
             title=f"Candlestick Chart from {start_datetime} to {end_datetime}",
             ylabel='Price', figsize=(12, 8))
