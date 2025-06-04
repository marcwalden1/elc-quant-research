# Standard library imports
import datetime as dt
import logging
import time
from collections import namedtuple

# Third-party library imports
import databento as db
import pandas as pd
import pytz
from dotenv import load_dotenv

# Local project imports
from iFVG_live import (
    check_if_in_poi,
    convert_nanoseconds_to_ny_datetime,
    find_HTF_POI,
    floor_to_15minute_ns,
    floor_to_hour_ns,
    floor_to_minute_ns,
    identify_inverse_momentum_gap,
    # nanoseconds_to_datetime64,
    # nanoseconds_to_new_york_datetime64,
    POI_invalidation_1,
    POI_invalidation_2,
    POI_invalidation_3,
    update_olhc_data,
    validate_inverse_momentum_gap,
)
from parquet_utils import get_nq_contract, read_parquet_file, convert_side_to_numeric
from tick_stream_processor import TickStreamProcessor
from trade_manager import TradeManager

# Load environment variables
load_dotenv()

# _TARGET_SYMBOL = "ESH5"  # Filter for this symbol
TimeRange = namedtuple("TimeRange", ["start", "end"])


# from algo import algo # IMPORT ALGORITHM ****

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.disable(logging.INFO)

# --- Global Vars ---
NANOSECONDS_1M = 60 * 1_000_000_000

# --- Helper Funcs ---
def convert_no_tz_timestamp_to_ny(timestamp: pd.Timestamp) -> pd.Timestamp:
    utc_time = pd.Timestamp(timestamp, tz='UTC')
    return utc_time.astimezone(pytz.timezone('America/New_York'))


class Pipeline:
    """
    A pipeline for processing tick data and executing algorithms.
    """

    def __init__(
        self,
        date,
        processor: TickStreamProcessor,
        trade_manager: TradeManager
        # algorithm: algorithm
    ):
        """
        Initializes the Pipeline.

        Args:
            processor: An instance of TickStreamProcessor.
            algorithm: An optional algorithm function to be executed on processed data.
        """
        self.processor = processor
        # self.algorithm = algorithm
        # self.last_interval_value_1m = None  # Initialize last interval value
        self.date = date

        self.ohlc_data_1h = pd.DataFrame(columns=['time(ns)', 'time', 'open', 'high', 'low', 'close', 'volume'])
        self.ohlc_data_15m = pd.DataFrame(columns=['time(ns)', 'time', 'open', 'high', 'low', 'close', 'volume'])
        self.ohlc_data_1m = pd.DataFrame(columns=['time(ns)', 'time', 'open', 'high', 'low', 'close', 'volume'])
        self.inactive_pois = []
        self.active_pois = []
        self.in_bullish_poi = False
        self.in_bearish_poi = False
        self.most_recent_candle_1h = None
        self.most_recent_candle_15m = None
        self.most_recent_candle_1m = None
        self.last_1h_candle_time = None
        self.last_15m_candle_time = None
        self.last_1m_candle_time = None
        self.potential_ifvgs = []
        self.recent_bullish_momentum_gap = False
        self.recent_bearish_momentum_gap = False
        self.trade_manager = trade_manager
        self.trade_manager.needs_data = False
        self.in_a_trade = False
        self.window = 400 # Hyperparameter
        self.threshold = 0.6 # Hyperparameter
        self.close_through_stop = False # Hyperparameter



    def run(self, mode):
        """
        Runs the data processing pipeline.
        """
        if mode == 'backtest':
            df = read_parquet_file(self.date) # UPDATE FUNCTION TO READ PARQUET FILE(S)
            
            # print(f"[DEBUG] read_parquet_file returned: {df}")
            for row in df.iter_rows(named=True):
                
                row['ts_event'] = int(row['ts_event'].timestamp() * 1e9)
                row['ts_recv'] = int(row['ts_recv'].timestamp() * 1e9)
                # import pdb;pdb.set_trace()
                row['price'] = int(row['price'] * 100)
                

                ts_event_dt = dt.datetime.fromtimestamp(row['ts_event'] / 1e9, tz=pytz.utc)

                # Determine if the tick falls in a valid front-month range
                contract_code = get_nq_contract(ts_event_dt)
                

                if contract_code != row['symbol']:
                    continue


                record = db.TradeMsg(
                    0,  # publisher_id (dummy) 
                    row['instrument_id'],
                    row['ts_event'],
                    row['price'],
                    row['size'],
                    "T",  # action (dummy)
                    row['side'],
                    0,  # depth (dummy)
                    row['ts_recv'])
                
                self.process_and_execute(record)

        if mode == 'live':
            # not complete
            for row in df.iter_rows(named=True):
                self.process_and_execute(record)


    def process_and_execute(self, record: db.TradeMsg):
        """
        Processes a single record, checks for interval changes, and executes the algorithms.

        Args:
            record: A Databento trade message.
        """

        self.processor.process_records(record)  # Process the record
        

        # Check for interval change and execute algorithms if necessary
        current_ts_event = record.hd.ts_event  # Get current interval value
        # current_ts_event_dt = nanoseconds_to_datetime64(current_ts_event)
        current_ts_event_ny = convert_nanoseconds_to_ny_datetime(current_ts_event)
        current_ts_event_1m = floor_to_minute_ns(current_ts_event)
        current_ts_event_15m = floor_to_15minute_ns(current_ts_event)
        current_ts_event_1h = floor_to_hour_ns(current_ts_event)

        trades = self.processor.get_raw_data()

        if self.last_1h_candle_time is None or current_ts_event_1h > self.last_1h_candle_time:
            logging.info(f"New 1-Hour Candle Detected | Current Time (ns): {current_ts_event_ny}")


            # Append new olhc row to the olhc dataframe!
            
            if self.last_1h_candle_time is not None:
                
                self.ohlc_data_1h, self.most_recent_candle_1h = update_olhc_data(self.ohlc_data_1h, trades, self.last_1h_candle_time, '1h')
                

            # This will check if the previous 3 candles form the 3-candle momentum gap formation and append any ones it finds to self.inactive_pois.
            
            # import pdb;pdb.set_trace()
            # find_HTF_POI(self.ohlc_data_1h, self.inactive_pois, '1h')
            
            # Now I will check if this candle closed below bullish POIs or above bearish POIs which invalidates them.
            # Notice that this is being checked before whether a POI will be activated is checked (think about edge case).
            if self.most_recent_candle_1h is not None:
                self.active_pois = POI_invalidation_1(self.active_pois, self.most_recent_candle_1h, '1h')

            # These lines will simply invalidate POIs that are too old. 
            
            
            self.inactive_pois, self.active_pois = POI_invalidation_2(self.inactive_pois, self.active_pois, current_ts_event_1h, '15m', candles = 15)

            # Check if there are any POIs with gap_direction == "bullish"
            has_bullish_pois = any(poi['gap_direction'] == 'bullish' for poi in self.active_pois)
            has_bearish_pois = any(poi['gap_direction'] == 'bearish' for poi in self.active_pois)

            # To check if the list of bullish POIs is empty
            if has_bullish_pois:
                logging.info("There are still active bullish POIs in the list.")
            else:
                self.in_bullish_poi = False
                # logging.info("There are not any active bullish POIs in the list.")
            if has_bearish_pois:
                logging.info("There are still active bearish POIs in the list.")
            else:
                self.in_bearish_poi = False
                # logging.info("There are not any active bearish POIs in the list.")
            
            self.last_1h_candle_time = current_ts_event_1h  # Update last 1-hour candle close time

        
        if self.last_15m_candle_time is None or current_ts_event_15m > self.last_15m_candle_time:
            logging.info(f"New 15-Minute Candle Detected | Current Time (ns): {current_ts_event_ny}")


            # Append new olhc row to the olhc dataframe!
            
            if self.last_15m_candle_time is not None:
                # import pdb;pdb.set_trace()
                self.ohlc_data_15m, self.most_recent_candle_15m = update_olhc_data(self.ohlc_data_15m, trades, self.last_15m_candle_time, '15m')
                

            # This will check if the previous 3 candles form the 3-candle momentum gap formation and append any ones it finds to self.inactive_pois.
            
            find_HTF_POI(self.ohlc_data_15m, self.inactive_pois, '15m')
            
            # Now I will check if this candle closed below bullish POIs or above bearish POIs which invalidates them.
            # Notice that this is being checked before whether a POI will be activated is checked (think about edge case).
            if self.most_recent_candle_15m is not None:
                self.active_pois = POI_invalidation_1(self.active_pois, self.most_recent_candle_15m, '15m')

            # These lines will simply invalidate POIs that are too old. 
            self.inactive_pois, self.active_pois = POI_invalidation_2(self.inactive_pois, self.active_pois, current_ts_event_15m, '15m', candles = 15)

            # Check if there are any POIs with gap_direction == "bullish"
            has_bullish_pois = any(poi['gap_direction'] == 'bullish' for poi in self.active_pois)
            has_bearish_pois = any(poi['gap_direction'] == 'bearish' for poi in self.active_pois)

            # To check if the list of bullish POIs is empty
            if has_bullish_pois:
                logging.info("There are still active bullish POIs.")
            else:
                self.in_bullish_poi = False
                # logging.info("There are not any active bullish POIs in the list.")
            if has_bearish_pois:
                logging.info("There are still active bearish POIs.")
            else:
                self.in_bearish_poi = False
                # logging.info("There are not any active bearish POIs in the list.")
            
            self.last_15m_candle_time = current_ts_event_15m  # Update last 15-minute candle close time


        # This will run at interval change. Assumes that first tick - minute start is negligible
        if self.last_1m_candle_time is None or (current_ts_event_1m > self.last_1m_candle_time):
            # logging.info(f"Minute Changed | Current Time: {current_ts_event_ny}")
        
            # Append new olhc row to the olhc dataframe!
            if self.last_1m_candle_time is not None:
                self.ohlc_data_1m, self.most_recent_candle_1m = update_olhc_data(self.ohlc_data_1m, trades, self.last_1m_candle_time, '1m')
                most_recent_candle_1m_close = self.most_recent_candle_1m['close']

            # Deal with exiting an open trade if your stop is a close through the ifvg
            if self.trade_manager.needs_data and self.close_through_stop:
                if self.trade_manager.open_trade['Direction'] == 'bullish' and most_recent_candle_1m_close <= self.trade_manager.open_trade['Stoploss_price']:
                    closed_time = pd.to_datetime(record.ts_recv).tz_localize('UTC').tz_convert('America/New_York')
                    self.trade_manager._close_trade(closed_time, exit_price=most_recent_candle_1m_close)
                    self.in_a_trade = self.trade_manager.has_open_trade()
                    if not self.in_a_trade:
                        recorded_trades.append(self.trade_manager.trades[-1])
                    
                elif self.trade_manager.open_trade['Direction'] == 'bearish' and most_recent_candle_1m_close <= self.trade_manager.open_trade['Stoploss_price']:
                    closed_time = pd.to_datetime(record.ts_recv).tz_localize('UTC').tz_convert('America/New_York')
                    self.trade_manager._close_trade(closed_time, exit_price=most_recent_candle_1m_close)
                    self.in_a_trade = self.trade_manager.has_open_trade()
                    if not self.in_a_trade:
                        recorded_trades.append(self.trade_manager.trades[-1])


            # Check if we entered a valid HTF POI
            new_poi_entered = check_if_in_poi(self.ohlc_data_1m, self.inactive_pois)
            
            if new_poi_entered is not None:
                # import pdb;pdb.set_trace()
                self.active_pois.append(new_poi_entered)
                
                if new_poi_entered['gap_direction'] == 'bullish':
                    self.in_bullish_poi = True
                elif new_poi_entered['gap_direction'] == 'bearish':
                    self.in_bearish_poi = True

                logging.info(f"Market entered a {new_poi_entered['gap_direction']} POI from time {new_poi_entered['time']}!")
            
            # Loop through the active POIs and check whether price has already ran ERL to invalidate them.
            if self.most_recent_candle_1m is not None:
                self.active_pois = POI_invalidation_3(self.active_pois, self.most_recent_candle_1m)
            

            # Check if there are any POIs with gap_direction == "bullish"
            has_bullish_pois = any(poi['gap_direction'] == 'bullish' for poi in self.active_pois)
            has_bearish_pois = any(poi['gap_direction'] == 'bearish' for poi in self.active_pois)

            # To check if the list of POIs is empty
            if not has_bullish_pois:
                self.in_bullish_poi = False
            if not has_bearish_pois:
                self.in_bearish_poi = False

            
            # Now we will start the lower timeframe search for a inverse momentum gap.

            # This must first begin here by designing a way to find each momentum gap. 
            # They will be stored as a dictionary in a list called potential_ifvgs.
            updated_ifvgs = identify_inverse_momentum_gap(self.ohlc_data_1m, self.potential_ifvgs, self.recent_bullish_momentum_gap, self.recent_bearish_momentum_gap)
            self.recent_bullish_momentum_gap = updated_ifvgs[1]
            self.recent_bearish_momentum_gap = updated_ifvgs[2]
            
            # if self.most_recent_candle_1m and self.most_recent_candle_1m['time'] == pd.Timestamp('2024-09-03 11:18:00-0400', tz='America/New_York'):
            #     import pdb;pdb.set_trace()
            # if self.most_recent_candle_1m and self.most_recent_candle_1m['time'] == pd.Timestamp('2024-09-03 11:20:00-0400', tz='America/New_York'):
            #     import pdb;pdb.set_trace()

            if any(ifvg['inverted_now'] for ifvg in self.potential_ifvgs):
                inverse_momentum_gap = [gap for gap in self.potential_ifvgs if gap.get('inverted_now')][0]

                validate_inverse_momentum_gap(inverse_momentum_gap, self.ohlc_data_1m)

                inverse_momentum_gap['inverted_now'] = False

                if (inverse_momentum_gap['gap_direction'] == 'bearish') and (inverse_momentum_gap['valid']) and (self.in_bullish_poi) and (not inverse_momentum_gap['already_used']):
                    
                    for poi in [poi for poi in self.active_pois if poi['gap_direction'] == 'bullish']:
                        if len(self.active_pois) == 0:
                            raise ValueError("Something is wrong with the logic. self.in_bullish_poi should only be true if the length is >0.")
                        mid_point = poi['gap_low'] + (poi['gap_high'] - poi['gap_low']) / 2

                        if (inverse_momentum_gap['inverted_at_price'] >= mid_point) and (not inverse_momentum_gap['already_used']):
                            inverse_momentum_gap['already_used'] =  True

                            # Now just only consider trades between acceptable_start_time and acceptable_end_time
                            trade_time = inverse_momentum_gap['inverted_at_time']
                            acceptable_start_time = dt.time(9, 30)
                            acceptable_end_time = dt.time(13, 0)

                            if (acceptable_start_time <= trade_time.time() <= acceptable_end_time):
                                logging.info(f"Executing a bullish trade at price {inverse_momentum_gap['inverted_at_price']} and time {inverse_momentum_gap['inverted_at_time']}")
                                trade_info = {"Entry_price": inverse_momentum_gap['inverted_at_price'],
                                            "Stoploss_price": inverse_momentum_gap['proposed_sl'],
                                            "Direction": 'bullish',
                                            "Trade_time": pd.to_datetime(trade_time),
                                            "Date": inverse_momentum_gap['date'],
                                            "Closed_x_far": inverse_momentum_gap['close_x_far']}
                                print(f"Trade_info: {trade_info}❗❗❗")
                                all_trades.append(trade_info)
                                # import pdb;pdb.set_trace()
                            
                                if not self.in_a_trade:
                                    self.in_a_trade = True
                                    self.trade_manager.start_trade(trade_info, self.window)
                                    

                
                elif (inverse_momentum_gap['gap_direction'] == 'bullish') and (inverse_momentum_gap['valid']) and (self.in_bearish_poi) and (not inverse_momentum_gap['already_used']):
                    for poi in [poi for poi in self.active_pois if poi['gap_direction'] == 'bearish']:
                        if len(self.active_pois) == 0:
                            raise ValueError("Something is wrong with the logic. self.in_bearish_poi should only be true if the length is >0.")
                        mid_point = poi['gap_low'] + (poi['gap_high'] - poi['gap_low']) / 2

                        if (inverse_momentum_gap['inverted_at_price'] <= mid_point) and (not inverse_momentum_gap['already_used']):
                            inverse_momentum_gap['already_used'] =  True
                            
                            # Now just only consider trades between acceptable_start_time and acceptable_end_time
                            trade_time = inverse_momentum_gap['inverted_at_time']
                            acceptable_start_time = dt.time(8, 0) 
                            acceptable_end_time = dt.time(15, 30)

                            if (acceptable_start_time <= trade_time.time() <= acceptable_end_time):
                                logging.info(f"Executing a bearish trade at price {inverse_momentum_gap['inverted_at_price']} and time {inverse_momentum_gap['inverted_at_time']}")
                                trade_info = {"Entry_price": inverse_momentum_gap['inverted_at_price'],
                                            "Stoploss_price": inverse_momentum_gap['proposed_sl'],
                                            "Direction": 'bearish',
                                            "Trade_time": pd.to_datetime(trade_time),
                                            "Date": inverse_momentum_gap['date'],
                                            "Closed_x_far": inverse_momentum_gap['close_x_far']}
                                print(f"Trade_info: {trade_info}❗❗❗")
                                all_trades.append(trade_info)
                                # import pdb;pdb.set_trace()
                            
                                if not self.in_a_trade:
                                    self.in_a_trade = True
                                    self.trade_manager.start_trade(trade_info, self.window)
                                    
                     
            
            self.last_1m_candle_time = current_ts_event_1m # Overwrite last interval with latest value

        if self.trade_manager.needs_data:
            volume = convert_side_to_numeric(record.side) * record.size
            trade_direction = self.trade_manager.open_trade['Direction']
            tick = {'price': record.price/100, 'volume': volume, 'side': record.side, 'Direction': trade_direction, 'time_ns': record.ts_recv}
            self.trade_manager.process_tick(tick, window=self.window, threshold=self.threshold)
            # import pdb; pdb.set_trace()
            self.in_a_trade = self.trade_manager.has_open_trade()
            if not self.in_a_trade:
                recorded_trades.append(self.trade_manager.trades[-1])

def valid_time_range(current_dt_ny) -> bool:
    """
    Checks if the current time is within specific intervals.

    Args:
        current_time: The current time as np.datetime64.

    Returns:
        bool: True if current_time is within the specified intervals, False otherwise.
    """
    trade_time_ranges = [
        TimeRange(
            start=current_dt_ny.replace(hour=9, minute=35, second=0, microsecond=0),
            end=current_dt_ny.replace(hour=11, minute=30, second=0, microsecond=0)
        ),
        TimeRange(
            start=current_dt_ny.replace(hour=13, minute=30, second=0, microsecond=0),
            end=current_dt_ny.replace(hour=15, minute=30, second=0, microsecond=0)
        ),
    ]

    for time_range in trade_time_ranges:
        if time_range.start <= current_dt_ny <= time_range.end:
            return True
    return False


def main(date):
    """
    Main function to run the data processing pipeline.
    """
    processor = TickStreamProcessor(max_records=2_000_000)
    trade_manager = TradeManager(window=400) 

    logging.info("Pipeline Started...")
    pipeline = Pipeline(
        processor=processor,
        trade_manager=trade_manager,
        date=date
    )
    

    pipeline.run(mode='backtest')



if __name__ == "__main__":
    all_trades = []
    recorded_trades = []

    start_date = dt.datetime.strptime("2024-03-17", "%Y-%m-%d")
    end_date = dt.datetime.strptime("2024-12-17", "%Y-%m-%d")

    current_date = start_date
    while current_date <= end_date:
        start_time = time.time()
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"🔁 Running pipeline for {date_str}...")
        try:
            main(date_str)
        except Exception as e:
            print(f"❌❌❌ Error on {date_str}: {e} ❌❌❌")
        current_date += dt.timedelta(days=1)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Completed pipeline for {date_str} in {elapsed_time:.2f} seconds.")
    # Save to CSV
    recorded_trades = pd.DataFrame(recorded_trades)
    # recorded_trades.to_csv("model_2_live_backtest_dtp_2025.csv", index=False)
    # print("📁 All trades saved to model_2_live_backtest_dtp_2025.csv")