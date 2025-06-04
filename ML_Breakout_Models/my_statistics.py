import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from iFVG_variations import *
import scipy.stats
import mplfinance as mpf
import model_variations
import itertools
import concurrent.futures
import os



def important_statistics(stats):
    # Calculate statistics

    # Ensure 'Date' column is in datetime format
    stats['Date'] = pd.to_datetime(stats['Date'], errors='coerce')

    total_pnl = stats['PnL'].sum()
    total_r = stats['R'].sum()
    wins = (stats['Win']).sum()
    losses = (not stats['Win']).sum()
    winrate = wins / (wins + losses) * 100
    expected_value_per_trade = stats[stats["Direction"].notna()]["PnL"].sum() / len(stats[stats["Direction"].notna()])
    
    # Separate winning and losing trades
    winning_trades = stats[stats["Direction"].notna()][stats[stats["Direction"].notna()]["PnL"] > 0]
    losing_trades = stats[stats["Direction"].notna()][stats[stats["Direction"].notna()]["PnL"] < 0]
    
    total_gross_profit = winning_trades["PnL"].sum()
    total_gross_loss = losing_trades["PnL"].sum()
    profit_factor = total_gross_profit / abs(total_gross_loss)
    
    # Max Drawdown (assuming max_drawdown function exists)
    max_dd, max_dd_duration = max_drawdown(stats['PnL'].fillna(0).cumsum())
    
    total_days_recorded = stats['Date'].nunique()
    total_trades = stats['Direction'].notna().sum()
    total_days_not_traded = stats[stats['Direction'].isna()]['Date'].nunique() - (stats['Date'].dt.day_name() == "Sunday").sum()
    
    bullish_count = stats[stats['Direction'] == 'bullish'].shape[0]
    bearish_count = stats[stats['Direction'] == 'bearish'].shape[0]
    
    # Create a DataFrame with the statistics
    stats_df = pd.DataFrame({
        "Total PnL ($)": [total_pnl],
        "Total R": [total_r],
        "Win Rate (%)": [winrate],
        "Max Drawdown ($)": [max_dd],
        "Max Drawdown Duration (days)": [max_dd_duration],
        "Expected Value per Trade ($)": [expected_value_per_trade],
        "Profit Factor": [profit_factor],
        "Total Gross Profit ($)": [total_gross_profit],
        "Total Gross Loss ($)": [total_gross_loss],
        "Wins": [wins],
        "Losses": [losses],
        "Total Days Recorded": [total_days_recorded],
        "Total Trades": [total_trades],
        "Total Days Not Traded": [total_days_not_traded],
        "Bullish Trades": [bullish_count],
        "Bearish Trades": [bearish_count]
    })
    
    return stats_df

def cumulative_PnL(stats):
    PnL_processed = stats['PnL'].fillna(0)

    stats['Cumulative_PnL'] = PnL_processed.cumsum()

    plt.figure(figsize=(10, 6))
    plt.plot(stats['Date'], stats['Cumulative_PnL'], color='blue', label='Cumulative PnL')
    plt.title('Cumulative PnL Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL')
    plt.grid(True)
    plt.legend()
    plt.show()

def cumulative_R(stats):
    R_processed = stats['R'].fillna(0)

    stats['Cumulative_R'] = R_processed.cumsum()

    plt.figure(figsize=(10, 6))
    plt.plot(stats['Date'], stats['Cumulative_R'], color='blue', label='Cumulative R')
    plt.title('Cumulative R Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative R')
    plt.grid(True)
    plt.legend()
    plt.show()

def performance_by_time_of_entry(stats, bin_size=15):
    """
    This function calculates and plots the average PnL by entry time, using a specified time bin size.

    Parameters:
    - stats: DataFrame containing trading data with 'Entry_Time' and 'PnL' columns.
    - bin_size: The size of the time bins in minutes (default is 15).
    """
    # Ensure 'Entry_Time' is in datetime format
    stats['Entry_Time'] = pd.to_datetime(stats['Entry_Time'], errors='coerce')

    # Round 'Entry_Time' to the nearest time bin interval (in minutes)
    time_bin_label = f'{bin_size}T'  # Construct the bin size string (e.g., '15T' for 15 minutes)
    stats['Entry_Time_Bin'] = stats['Entry_Time'].dt.floor(time_bin_label)

    # Group by the time bin and calculate the average PnL
    pnl_by_entry_time_bin = stats.groupby('Entry_Time_Bin')['PnL'].mean()

    # Plot the average PnL by entry time bin
    plt.figure(figsize=(10, 6))
    pnl_by_entry_time_bin.plot(kind='bar', color='lightblue')
    plt.title(f'Average PnL by {bin_size}-Minute Entry Time Interval')
    plt.xlabel(f'Entry Time ({bin_size}-minute intervals)')
    plt.ylabel('Average PnL')
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate labels to fit better
    plt.show()



def count_entries_by_time_bin(stats, bin_size=15):
    """
    This function counts how many entries (trades) were taken within each time bin and plots the result.
    
    Parameters:
    - stats: DataFrame containing trading data with 'Entry_Time' column
    - bin_size: Size of the time bin in minutes (default is 15)
    
    Returns:
    - A DataFrame with the counts of entries per time bin.
    """
    # Ensure 'Entry_Time' is in datetime format
    stats['Entry_Time'] = pd.to_datetime(stats['Entry_Time'], errors='coerce')

    # Round 'Entry_Time' to the nearest time bin interval (in minutes)
    time_bin_label = f'{bin_size}T'  # Construct the bin size string (e.g., '15T' for 15 minutes)
    stats['Entry_Time_Bin'] = stats['Entry_Time'].dt.floor(time_bin_label)

    # Group by the time bin and count the number of entries in each bin
    entries_by_time_bin = stats.groupby('Entry_Time_Bin').size()

    # Convert the result to a DataFrame and reset the index for easier access
    entries_by_time_bin_df = entries_by_time_bin.reset_index(name='Entry_Count')

    # Plot the result
    plt.figure(figsize=(10, 6))
    plt.bar(entries_by_time_bin_df['Entry_Time_Bin'].dt.strftime('%H:%M'), 
            entries_by_time_bin_df['Entry_Count'], color='black')
    plt.title(f'Number of Entries per {bin_size}-Minute Time Bin')
    plt.xlabel(f'Time Bin ({bin_size}-minute intervals)')
    plt.ylabel('Number of Entries')
    plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.grid(True)
    plt.show()


def EV_vs_trade_duration(stats):
    # Group trades by duration bins (e.g., <30 mins, 30-60 mins, >60 mins)
    duration_bins = [0, 30, 60, 120, 240, 480]  # Example bins for trade duration (minutes)
    duration_labels = ['<30 mins', '30-60 mins', '1-2 hours', '2-4 hours', '>4 hours']

    stats['Duration_Bin'] = pd.cut(stats['Trade_Duration (mins)'], bins=duration_bins, labels=duration_labels)

    # Calculate expected value per duration bin (mean PnL)
    ev_by_duration = stats.groupby('Duration_Bin')['PnL'].mean()

    # Plot expected value per trade vs duration
    plt.figure(figsize=(10, 6))
    ev_by_duration.plot(kind='bar', color='lightgreen')
    plt.title('Expected Value per Trade vs Trade Duration')
    plt.xlabel('Trade Duration')
    plt.ylabel('Expected Value per Trade (PnL)')
    plt.grid(True)
    plt.show()


def days_of_the_week(stats):
    # Extract day of week from Date
    stats['Date'] = pd.to_datetime(stats['Date'], errors='coerce')

    stats['Day_of_Week'] = stats['Date'].dt.day_name()

    # Group by day of week and calculate mean PnL
    pnl_by_day = stats.groupby('Day_of_Week')['PnL'].mean().sort_values()

    plt.figure(figsize=(10, 6))
    pnl_by_day.plot(kind='bar', color='lightgreen')
    plt.title('Average PnL by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average PnL')
    plt.grid(True)
    plt.show()


def EV_CI(trade_results, alpha=0.95):
    """
    This function calculates the confidence interval for the expected value per trade 
    based on the list of individual trade results in R.

    Inputs:
    - trade_results: list or array of R values for each trade (win or loss)
    - alpha: confidence level (default is 95%)
    
    Output:
    - Confidence Interval for Expected Value per Trade
    """

    total_trades = len(trade_results)

    # Calculate the mean expected value per trade
    mean_ev_per_trade = np.mean(trade_results)

    # Calculate the standard deviation of the R values
    std_dev = np.std(trade_results)

    # Standard Error
    SE = std_dev / np.sqrt(total_trades)

    # Degrees of freedom
    df = total_trades - 1

    # Find the t critical value
    t_critical = scipy.stats.t.ppf(1 - (1 - alpha) / 2, df)

    # Margin of error
    margin_of_error = t_critical * SE

    # Confidence Interval
    CI_lower = mean_ev_per_trade - margin_of_error
    CI_upper = mean_ev_per_trade + margin_of_error

    print(f"Expected Value per Trade: {mean_ev_per_trade}")
    print(f"{alpha * 100}% Confidence Interval for Expected Value per Trade: ({round(CI_lower, 3)}, {round(CI_upper, 3)})")






def visualize_trade_with_candlesticks(trade_info, df, start_time, end_time, timeframe='5m', asset='ES', profit_target = True):
    """
    Visualizes the trade with candlestick chart using mplfinance for a given timeframe.
    
    Parameters:
    trade_info (dict): The trade information returned from model_1_day.
    df (DataFrame): DataFrame containing historical trading data.
    start_time (str): The start time for the chart in 'YYYY-MM-DD HH:MM:SS' format.
    end_time (str): The end time for the chart in 'YYYY-MM-DD HH:MM:SS' format.
    timeframe (str): The timeframe for the candlestick data, e.g., '1m' for 1-minute, '5m' for 5-minute (default is '5m').
    
    Returns:
    None: Displays the candlestick chart with the trade's entry, exit, stop loss, and target price.
    """

    
    # Convert start_time and end_time to datetime
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    entry_time = pd.to_datetime(trade_info['Entry_Time'])
    entry_price = trade_info['Entry_Price']
    exit_time = pd.to_datetime(trade_info['Exit_Time'])

    start_time = start_time.time()
    end_time = end_time.time()

    exit_price = trade_info['Exit_Price']
    stop_loss = trade_info['Stop_Loss_Placement']
    if profit_target:
        target_price = trade_info['Profit_Target_Placement']
    date = trade_info['Date']


    # print(f"Entry Time: {entry_time}, Exit Time: {exit_time}")
    
    # Fetch the OHLC data for the given date and timeframe
    olhc_data = olhcv_data(df, date, timeframe=timeframe, asset=asset)
    olhc_data_to_pandas = olhc_data.to_pandas()

    # print(olhc_data_to_pandas.head()) this is working

    # Check if it's a Saturday or if the first candle starts at Sunday 23:00
    if olhc_data_to_pandas.empty or (pd.to_datetime(olhc_data_to_pandas['interval_time'].iloc[0]) == pd.to_datetime("2024-11-17 23:00:00+00:00")):
        return {"It's a Saturday or Sunday"}

    # Convert 'time' column to datetime for comparison
    olhc_data_to_pandas['time'] = pd.to_datetime(olhc_data_to_pandas['time'])

    olhc_data_to_pandas['time_only'] = olhc_data_to_pandas['time'].dt.time



    # Filter the data for the given time range
    trade_data = olhc_data_to_pandas[(olhc_data_to_pandas['time_only'] >= start_time) & 
                                      (olhc_data_to_pandas['time_only'] <= end_time)]
    
    # Prepare the data for mplfinance (only OHLC data)
    trade_data = trade_data.set_index('time')
    trade_data = trade_data[['open', 'high', 'low', 'close']]  # Only need OHLC for mplfinance
    
    # Create an Entry Line (instead of scatter, we use a line)
    entry_line = np.full(len(trade_data), np.nan)


    if entry_time is not None and exit_time is not None:
        # Fill the entry line for the duration of the trade
        entry_idx = trade_data.index.get_loc(trade_data.index.asof(entry_time))
        exit_idx = trade_data.index.get_loc(trade_data.index.asof(exit_time))
        entry_line[entry_idx:exit_idx+1] = entry_price
        # Create the addplot for the entry line
        entry_line_plot = mpf.make_addplot(entry_line, panel=0, color='black', linestyle='-', width=2)

        stop_loss_line = np.full(len(trade_data), np.nan)
        target_line = np.full(len(trade_data), np.nan)

        stop_loss_line[entry_idx:exit_idx+1] = stop_loss
        if profit_target:
            target_line[entry_idx:exit_idx+1] = target_price
            target_plot = mpf.make_addplot(target_line, panel=0, color='green', linestyle='-', width=2)

        stop_loss_plot = mpf.make_addplot(stop_loss_line, panel=0, color='red', linestyle='-', width=2)

        if profit_target:
            mpf.plot(trade_data, type='candle', style='charles', title=f"Trade on {date} at {entry_time}. (Entry: {entry_price} / Exit: {exit_price})",
                 ylabel='Price', 
                 addplot=[entry_line_plot, stop_loss_plot, target_plot], 
                 figsize=(12, 8))
        else:
            mpf.plot(trade_data, type='candle', style='charles', title=f"Trade on {date} at {entry_time}. (Entry: {entry_price} / Exit: {exit_price})",
                 ylabel='Price', 
                 addplot=[entry_line_plot, stop_loss_plot], 
                 figsize=(12, 8))

    else:
        mpf.plot(trade_data, type='candle', style='charles', title=f"No Trade",
                 ylabel='Price', figsize=(12, 8))  # No entry line if no trade
        


def grid_search(pandas_data, df, model_1_day_9, thresholds, windows, min_fvg_sizes, NYSEOpen=False):
    results = []
    
    # Generate all combinations of hyperparameters
    param_combinations = itertools.product(thresholds, windows, min_fvg_sizes)
    
    for threshold, window, min_fvg_size in param_combinations:
        # Run the model with the current combination of hyperparameters
        output_df = model_variations.model_1_6(pandas_data, df, model_1_day_9, threshold=threshold, window=window, minimum_fvg_size=min_fvg_size, NYSEOpen=NYSEOpen)
        
        # Get the important statistics for this run
        stats = important_statistics(output_df)
        
        # Flatten the 1x16 DataFrame into a dictionary
        stats_dict = stats.squeeze().to_dict()  # Converts the DataFrame into a dictionary

        # Add the hyperparameters to the stats dictionary
        stats_dict['threshold'] = threshold
        stats_dict['window'] = window
        stats_dict['minimum_fvg_size'] = min_fvg_size

        print(stats_dict)
        
        # Store the result
        results.append(stats_dict)
    
    # Convert results to a DataFrame for easy analysis and storage
    results_df = pd.DataFrame(results)
    
    return results_df


def grid_search_parallel(pandas_data, df, model_1_day_9, thresholds, windows, min_fvg_sizes):
    results = []

    # Generate all combinations of hyperparameters
    param_combinations = itertools.product(thresholds, windows, min_fvg_sizes)

    # Worker function to process each combination of hyperparameters
    def worker(params):
        threshold, window, min_fvg_size = params
        
        # Run the model with the current combination of hyperparameters
        output_df = model_variations.model_1_6(pandas_data, df, model_1_day_9, threshold=threshold, window=window, minimum_fvg_size=min_fvg_size)
        
        # Get the important statistics for this run
        stats = important_statistics(output_df)
        
        # Flatten the 1x16 DataFrame into a dictionary
        stats_dict = stats.squeeze().to_dict()  # Converts the DataFrame into a dictionary

        # Add the hyperparameters to the stats dictionary
        stats_dict['threshold'] = threshold
        stats_dict['window'] = window
        stats_dict['minimum_fvg_size'] = min_fvg_size

        return stats_dict

    # Use ProcessPoolExecutor to parallelize the computation
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        # Submit all tasks to the executor
        futures = [executor.submit(worker, params) for params in param_combinations]

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result)
            results.append(result)

    # Convert results to a DataFrame for easy analysis and storage
    results_df = pd.DataFrame(results)
    
    return results_df


def grid_search_2(pandas_data, df, model_2_day, close_too_far=20, minimum_size=4, asset='ES', maximum_age=15, can_run=3, max_candles_btwn=15, threshold=1.5, window=400, extra_sl_margin=0.25, stop_modification = False):
    results = []
    
    param_combinations = itertools.product(threshold, window) #, max_candles_btwn, extra_sl_margin)
    
    for threshold, window in param_combinations: #, max_candles_btwn, extra_sl_margin
        # Run the model with the current combination of hyperparameters
        output_df = model_2_2(
            pandas_data, df, model_2_day, close_too_far=close_too_far, 
            minimum_size=minimum_size, asset=asset, maximum_age=maximum_age, 
            can_run=can_run, max_candles_btwn=max_candles_btwn, threshold=threshold, window=window, stop_modification=stop_modification
        )
        # Get the important statistics for this run
        stats = important_statistics(output_df)
        
        # Flatten the 1x16 DataFrame into a dictionary
        stats_dict = stats.squeeze().to_dict()  # Converts the DataFrame into a dictionary

        # Add the hyperparameters to the stats dictionary
        stats_dict['threshold'] = threshold
        stats_dict['window'] = window
        # stats_dict['max_candles_btwn'] = max_candles_btwn
        # stats_dict['extra_sl_margin'] = extra_sl_margin
        # stats_dict['min_size'] = min_size

        print(stats_dict)
        
        # Store the result
        results.append(stats_dict)

        
    
    # Convert results to a DataFrame for easy analysis and storage
    results_df = pd.DataFrame(results)
    
    return results_df


def worker(params, pandas_data, df, model_2_day, close_too_far, minimum_size, asset, maximum_age, can_run):
    threshold, window, max_candles_btwn, extra_sl_margin = params
    
    output_df = model_2.model_2(
        pandas_data, df, model_2_day, close_too_far=close_too_far, 
        minimum_size=minimum_size, asset=asset, maximum_age=maximum_age, 
        can_run=can_run, max_candles_btwn=max_candles_btwn, threshold=threshold, 
        window=window, extra_sl_margin=extra_sl_margin
    )
    
    stats = important_statistics(output_df)
    stats_dict = stats.squeeze().to_dict()
    stats_dict['threshold'] = threshold
    stats_dict['window'] = window
    stats_dict['max_candles_btwn'] = max_candles_btwn
    stats_dict['extra_sl_margin'] = extra_sl_margin

    return stats_dict

# Main grid search function with parallelization
def grid_search_2_parallel(pandas_data, df, model_2_day, close_too_far=20, minimum_size=4, asset='ES', maximum_age=15, can_run=3, max_candles_btwn=15, threshold=[1.5], window=[400], extra_sl_margin=[0.25]):
    results = []
    
    param_combinations = itertools.product(threshold, window, max_candles_btwn, extra_sl_margin)
    
    # Use ProcessPoolExecutor to parallelize the computation
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        # Submit all tasks to the executor
        futures = [executor.submit(worker, params, pandas_data, df, model_2_day, close_too_far, minimum_size, asset, maximum_age, can_run) for params in param_combinations]

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result)
            results.append(result)
    
    # Convert results to a DataFrame for easy analysis and storage
    results_df = pd.DataFrame(results)
    
    return results_df


def heatmap(search_results, string_var='Total R', fix_bounds = True):
    # Get the R values and reshape them into a matrix based on the size of thresholds and windows
    R_values = search_results[string_var].values
    R_matrix = R_values.reshape(6, 7)  # 6 thresholds and 7 windows
    R_matrix = R_matrix[::-1]  # Reverse to match the desired order

    if string_var=='Total R' and fix_bounds:
        vmin = -3
        vmax = 34
    elif string_var == 'Expected Value per Trade ($)' and fix_bounds:
        vmin = -106
        vmax = 1006
    else:
        vmin = None
        vmax = None

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(R_matrix, cmap='coolwarm_r', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Total R')  # Add a color bar to show the scale

    # Set the labels for the axes
    plt.yticks(np.arange(6), [val for val in [4, 3.5, 3, 2.5, 2, 1.5]])
    plt.xticks(np.arange(7), [val for val in [200, 400, 800, 1600, 3200, 6400, 12800]])

    # Title and labels
    plt.title('Heatmap of R Values for Different Hyperparameter Combinations')
    plt.xlabel('Window')
    plt.ylabel('Threshold')

    # Display the heatmap
    plt.show()


def heatmap_2(search_results, string_var = 'Total R'):

    # Get the R values and reshape them into a 6x7 matrix
    R_values = search_results[string_var].values
    R_matrix = R_values.reshape(7, 6)
    R_matrix = R_matrix[::-1]


    # print(pd.DataFrame(R_matrix))

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(R_matrix, cmap='coolwarm_r', interpolation='nearest')
    plt.colorbar(label='Total R')  # Add a color bar to show the scale

    # Set the labels for the axes
    plt.yticks(np.arange(7), [val for val in [1.4, 1.2, 1, 0.8, 0.6, 0.4, 0.2]])
    plt.xticks(np.arange(6), [val for val in [100, 200, 400, 800, 1600, 3200]])

    # Title and labels
    plt.title('Heatmap of R Values for Different Hyperparameter Combinations')
    plt.xlabel('Window')
    plt.ylabel('Threshold')

    # Display the heatmap
    plt.show()



