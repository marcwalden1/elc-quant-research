import polars as pl
import os
import datetime as dt
processed_dir = "/Users/marcwalden/Desktop/ELC_folder/NQ/"


def _fix_data_types(df: pl.DataFrame) -> pl.DataFrame:
    """Fixes data types for various UInt types in a Polars DataFrame.

    This function iterates through the columns of a Polars DataFrame and
    casts UInt64, UInt32, UInt16, and UInt8 columns to their respective Int
    counterparts (Int64, Int32, Int16, Int8).

    Args:
        df: The input Polars DataFrame.

    Returns:
        A new Polars DataFrame with the corrected data types.
    """
    type_mapping = {
        pl.UInt64: pl.Int64,
        pl.UInt32: pl.Int32,
        pl.UInt16: pl.Int16,
        pl.UInt8: pl.Int8
    }
    for col in df.columns:
        original_type = df[col].dtype
        if original_type in type_mapping:
            df = df.with_columns(pl.col(col).cast(type_mapping[original_type]))
    return df


def _add_date_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Adds date and timestamp columns to a Polars DataFrame.

    This function creates two new columns: 'date' (representing the date
    extracted from 'ts_event') and 'ts_event_ny' (representing the
    timestamp in New York time, converted from 'ts_event').

    Args:
        df: The input Polars DataFrame containing a 'ts_event' column.

    Returns:
        A new Polars DataFrame with the 'date' and 'ts_event_ny' columns.
    """
    return df.with_columns(
        pl.col("ts_event").dt.date().alias("date"),
        pl.col("ts_event").dt.convert_time_zone("America/New_York").alias("ts_event_ny")
    )


def _select_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Selects specified columns from a Polars DataFrame.

    This function filters the DataFrame to include only the essential
    columns needed for downstream processing.

    Args:
        df: The input Polars DataFrame.

    Returns:
        A new Polars DataFrame containing only the specified columns.
    """
    desired_columns = [
        'ts_recv', 'ts_event', 'rtype', 'publisher_id',
        'instrument_id', 'action', 'side', 'depth',
        'price', 'size', 'flags', 'ts_in_delta',
        'sequence', 'symbol', 'date', 'ts_event_ny'
    ]
    existing_columns = [col for col in desired_columns if col in df.columns]
    return df.select(existing_columns)


def read_parquet_file(date_str: str, file_dir: str = processed_dir,
                      add_date_cols: bool = True,
                      fix_dtypes: bool = True) -> pl.DataFrame | None:
    """
    Reads a parquet file from the specified directory.

    Args:
        date (str): The date of the file to read in YYYYMMDD format.
        file_dir (str, optional): The directory to read the file from.
            Defaults to '/Users/eric/data/databento/futures/es/trades/processed/'.
        add_date_cols (bool, optional): Whether to add date and timestamp columns. Defaults to True.

    Returns:
        pl.DataFrame: The contents of the parquet file as a Polars DataFrame.
    """
    date = date_str.replace('-', '')
    file_name = f'trades-{date}.parquet'
    file_path = os.path.join(file_dir, file_name)

    try:
        df = pl.read_parquet(file_path)

        if add_date_cols:
            df = _add_date_columns(df)

        if fix_dtypes:
            df = _fix_data_types(df)
        # print("[DEBUG] Columns in DataFrame:", df.columns)
        return _select_columns(df)
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except pl.exceptions.ComputeError as e:
        print(f"Error reading file {file_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occured {e}")
        return None
    

def get_nq_contract(date_obj: dt.datetime) -> str:
    y = date_obj.year
    year_code = str(y % 10)

    if date_obj.month == 12 and date_obj.day >= 16:
        year_code = str((y + 1) % 10)  # December 16+ is next year's March contract

    if dt.datetime(y - 1, 12, 16, tzinfo=date_obj.tzinfo) <= date_obj <= dt.datetime(y, 3, 15, tzinfo=date_obj.tzinfo):
        return f"NQH{year_code}"
    elif dt.datetime(y, 3, 16, tzinfo=date_obj.tzinfo) <= date_obj <= dt.datetime(y, 6, 15, tzinfo=date_obj.tzinfo):
        return f"NQM{year_code}"
    elif dt.datetime(y, 6, 16, tzinfo=date_obj.tzinfo) <= date_obj <= dt.datetime(y, 9, 15, tzinfo=date_obj.tzinfo):
        return f"NQU{year_code}"
    elif dt.datetime(y, 9, 16, tzinfo=date_obj.tzinfo) <= date_obj <= dt.datetime(y, 12, 15, tzinfo=date_obj.tzinfo):
        return f"NQZ{year_code}"
    else:
        return "Invalid Date"


def convert_side_to_numeric(side):
    if side == "A":
        return 1
    elif side == "B":
        return -1
    elif side == "N":
        return 0
    
