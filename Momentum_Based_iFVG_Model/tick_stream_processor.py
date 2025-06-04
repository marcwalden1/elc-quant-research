import logging
import numpy as np
from dotenv import load_dotenv

load_dotenv()

_TARGET_SYMBOL = "ESM5"  # Filter for this symbol

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def floor_to_minute_ns(ns_int):
    """
    Floors a nanosecond integer to the start of the minute, maintaining the nanosecond integer format.

    Args:
        ns_int: An integer representing nanoseconds since the Unix epoch (1970-01-01T00:00:00).

    Returns:
        An integer representing the nanoseconds at the start of the minute.
    """
    # Number of nanoseconds in a minute
    ns_per_minute = 60 * 1_000_000_000
    return (ns_int // ns_per_minute) * ns_per_minute

class TickStreamProcessor:
    """
    Processes and stores tick data in memory,
    with pre-allocated NumPy array for efficiency.
    """

    def __init__(self, symbol=_TARGET_SYMBOL, max_records=1_000_000): #default max size
        # self.symbol = symbol
        self.max_records = max_records
        self.tick_data = np.zeros(self.max_records, dtype=[
        ('ts_event', 'int64'),
        ('interval_start_1m', 'int64'),
        ('price', 'float64'),
        ('size', 'float64'),
        ('side', 'int8'),
        ('depth', 'int8'),
        ])
        # self.tick_data = np.empty((self.max_records, len(self.tick_data_cols)), dtype=np.float64)
        self.data_index = 0
        self.aggregations = {}
        self.last_ts_event = None
        self.last_ts_event_minute = 0

    def convert_side_to_numeric(self, side):
        if side == "A":
            return 1
        elif side == "B":
            return -1
        elif side == "N":
            return 0

    def process_records(self, record):
        """
        Processes records and stores it in memory.
        """
        try:
            if self.data_index < self.max_records:
                self.tick_data[self.data_index] = (
                    record.hd.ts_event,
                    floor_to_minute_ns(record.hd.ts_event),
                    record.price / 100,
                    record.size,
                    self.convert_side_to_numeric(record.side),
                    # record.side.encode(),
                    record.depth
                )

                # Variable Updates
                self.last_ts_event = record.hd.ts_event
                self.last_ts_event_minute = floor_to_minute_ns(record.hd.ts_event)
                self.data_index += 1

                # logging.info(record)
            else:
                logging.warning("Warning: Maximum record limit reached. Increase max_records if necessary.")

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

    def flush_raw_data(self):
        """
        "Flushes" the stored data by resetting the index.  No data is removed.
        """
        self.data_index = 0 #reset index. data is still in the array

    def get_raw_data(self):
        """
        Retrieves the raw tick data.
        """
        return self.tick_data[:self.data_index] #return only valid data