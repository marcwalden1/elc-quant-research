import logging
import pandas as pd
import datetime
from iFVG_live import contract_size, update_mean_delta

class TradeManager:
    
    def __init__(self, window: int):
        self.trades = []  # List to store all executed trades
        self.open_trade = None  # Currently active trade (if any)
        self.needs_data = False  # Set to True if a trade is open and we need tick data
        self.running_sum = 0
        self.recent_deltas = [0] * window

    def start_trade(self, trade_info: dict, window: int, close_through_stop = False):
        """
        Starts a new trade if none is currently active.

        Args:
            trade_info (dict): A dictionary containing trade details like:
                {
                    "Entry_price": float,
                    "Stoploss_price": float,
                    "Direction": 'bullish' | 'bearish',
                    "Trade_time": datetime object,
                    "Date": datetime object,
                    "close_x_far": float
                }
        """

        trade_info['Exit_price'] = None
        trade_info['Exit_time'] = None
        trade_info['Win'] = None
        trade_info['Status'] = 'OPEN'

        entry_price = trade_info['Entry_price']
        stop_loss_price = trade_info['Stoploss_price']

        if close_through_stop and trade_info['Direction'] == 'bullish':
            trade_info['Hard_stop'] = stop_loss_price - 12
        elif close_through_stop and trade_info['Direction'] == 'bearish':
            trade_info['Hard_stop'] = stop_loss_price + 12

        self.open_trade = trade_info
        self.trades.append(trade_info)
        self.needs_data = True
        
        logging.info(f"[TRADE ENTERED] {trade_info}")

        

        

        stop_loss_size = abs(entry_price - stop_loss_price)

        self.open_trade['Contracts'] = contract_size(stop_loss_size, 1000, asset='NQ')

        self.running_sum = 0
        self.recent_deltas = [0] * window



    def process_tick(self, tick: dict, window: int, threshold: float, close_through_stop = False):
        """
        Process incoming tick to check if trade should be closed.
        """

        if self.open_trade is None:
            self.needs_data = False
            return

        price = tick['price']
        trade_direction = self.open_trade['Direction']
        volume = tick['volume']
        time_ns = tick['time_ns']


        # Close trade if SL is hit
        if not close_through_stop:
            if trade_direction == 'bullish' and price <= self.open_trade['Stoploss_price']:
                closed_time = pd.to_datetime(time_ns).tz_localize('UTC').tz_convert('America/New_York')
                self.open_trade['Win'] = False
                self._close_trade(closed_time, price)

            elif trade_direction == 'bearish' and price >= self.open_trade['Stoploss_price']:
                closed_time = pd.to_datetime(time_ns).tz_localize('UTC').tz_convert('America/New_York')
                self.open_trade['Win'] = False
                self._close_trade(closed_time, price)
        else:
            if trade_direction == 'bullish' and price <= self.open_trade['Hard_stop']:
                closed_time = pd.to_datetime(time_ns).tz_localize('UTC').tz_convert('America/New_York')
                self.open_trade['Win'] = False
                self._close_trade(closed_time, price)

            elif trade_direction == 'bearish' and price >= self.open_trade['Hard_stop']:
                closed_time = pd.to_datetime(time_ns).tz_localize('UTC').tz_convert('America/New_York')
                self.open_trade['Win'] = False
                self._close_trade(closed_time, price)

        self.running_sum, self.recent_deltas, recent_mean_delta = update_mean_delta(self.running_sum, self.recent_deltas, volume, window)

        if tick['Direction'] == 'bullish' and recent_mean_delta < -threshold:
            closed_time = pd.to_datetime(time_ns).tz_localize('UTC').tz_convert('America/New_York')
            self._close_trade(closed_time, price)
        if tick['Direction'] == 'bearish' and recent_mean_delta > threshold:
            closed_time = pd.to_datetime(time_ns).tz_localize('UTC').tz_convert('America/New_York')
            self._close_trade(closed_time, price)

        # if the current time is later than 15:59, the trade will be exited at current market price. 
        current_dt = datetime.datetime.fromtimestamp(time_ns/1e9)
        cutoff = current_dt.replace(hour=15, minute=59, second=0, microsecond=0)
        if current_dt > cutoff:
            closed_time = pd.to_datetime(time_ns).tz_localize('UTC').tz_convert('America/New_York')
            self._close_trade(closed_time, price)
            

    def _close_trade(self, time, exit_price: float):
        """
        Closes the currently open trade.

        Args:
            exit_price (float): The price at which the trade was closed.
        """
        self.open_trade['Exit_price'] = exit_price
        self.open_trade['Exit_time'] = time
        self.open_trade['Status'] = 'CLOSED'
        
        if self.open_trade['Direction'] == 'bullish':
            self.open_trade['PnL'] = (exit_price - self.open_trade['Entry_price']) * 20 * self.open_trade['Contracts']
            self.open_trade['R'] = (exit_price - self.open_trade['Entry_price']) / abs(self.open_trade['Entry_price'] - self.open_trade['Stoploss_price']) if abs(self.open_trade['Entry_price'] - self.open_trade['Stoploss_price']) > 0 else 0
        elif self.open_trade['Direction'] == 'bearish':
            self.open_trade['PnL'] = (self.open_trade['Entry_price'] - exit_price) * 20 * self.open_trade['Contracts']
            self.open_trade['R'] = (self.open_trade['Entry_price'] - exit_price) / abs(self.open_trade['Entry_price'] - self.open_trade['Stoploss_price']) if abs(self.open_trade['Entry_price'] - self.open_trade['Stoploss_price']) > 0 else 0
        self.open_trade['Win'] = True if self.open_trade['R'] > 0 else False
        

        logging.info(f"[TRADE CLOSED] {self.open_trade}")

        self.trades[-1] = self.open_trade
        self.open_trade = None
        self.needs_data = False
        self.in_a_trade = False


    def get_all_trades(self):
        """Returns a list of all trades, both open and closed."""
        return self.trades
    
    def has_open_trade(self) -> bool:
        return self.open_trade is not None

