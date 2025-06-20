import pandas as pd
import numpy as np

def moving_average_crossover(stock_data, short_term_period=20, long_term_period=50, initial_cash=10000, commission_rate=0.001):
    data = stock_data.copy()
    
    # Initialize values array w/ closing prices and total_num_days variable
    values = data['Close'].values
    total_num_days = len(values)

    # Initialize portfolio as fully cash, w/ default of $10K
    cash = initial_cash
    shares = 0
    portfolio_value_history = []

    # Initialize trade history
    trade_history = []

    # Calculate initial moving averages using Pandas built-in rolling mean
    data['SMA_Short'] = data['Close'].rolling(window=short_term_period).mean()
    # print('sma short arr', data['SMA_Short'])
    data['SMA_Long'] = data['Close'].rolling(window=long_term_period).mean()
    # print('sma long arr', data['SMA_Long'])

    # Strategy is only applicable on the long_term_period+1-st day, when short avg, long avg, and short_term_greater_prev can all be defined
    start_index_for_signals = long_term_period

    # Determine initial state for crossover detection
    # short_term_greater_prev = True if SMA_Short was > SMA_Long yesterday
    if data['SMA_Short'].iloc[start_index_for_signals - 1] > data['SMA_Long'].iloc[start_index_for_signals - 1]:
        short_term_greater_prev = True
    else:
        short_term_greater_prev = False

    # print('starting short_term_greater_prev', short_term_greater_prev)
    

    for day_index in range(start_index_for_signals, total_num_days):
        current_price = values[day_index]
        current_sma_short = data['SMA_Short'].iloc[day_index]
        current_sma_long = data['SMA_Long'].iloc[day_index]
        # print('current price, sht, lng', current_price, current_sma_short, current_sma_long)

        # Skip if invalid values
        if pd.isna(current_sma_short) or pd.isna(current_sma_long):
            # print('not enough data for day', day_index)
            portfolio_value_history.append(cash + shares * current_price)
            continue # Move to the next day

        signal = 0 # 0: no trade, 1: buy, -1: sell

        # Check for crossover conditions
        # Buy signal: short MA crosses above long MA
        if current_sma_short > current_sma_long and not short_term_greater_prev:
            signal = 1
        # Sell signal: short MA crosses below long MA
        elif current_sma_short < current_sma_long and short_term_greater_prev:
            signal = -1

        # print('todays signal:', signal)

        # Update short_term_greater_prev in preparation for next day
        short_term_greater_prev = (current_sma_short > current_sma_long)
        # print('new short term greater prev:', short_term_greater_prev)

        # Execute trades
        if signal == 1: # Buy
            if cash > 0: # Only buy if we have cash to do so
                # Calculate maximum shares we can buy
                buyable_shares = int(cash // current_price)
                if buyable_shares > 0:
                    cost = buyable_shares * current_price
                    transaction_cost = cost * commission_rate
                    
                    if cash >= (cost + transaction_cost):
                        shares += buyable_shares
                        old_cash = cash
                        cash -= (cost + transaction_cost)
                        trade_history.append({'Event' : 'BUY', 'Day' : day_index, 'Price' : float(np.round(current_price, 2)), 'Shares' : buyable_shares, 'Old Cash Balance' : float(np.round(old_cash, 2)), 'New Cash Balance' : float(np.round(cash, 2))})
                        # print(trade_history[-1])

        elif signal == -1: # Sell
            if shares > 0: # Only sell if we own shares
                proceeds = shares * current_price
                transaction_cost = proceeds * commission_rate
                old_cash = float(cash)
                old_shares = shares
                cash += (proceeds - transaction_cost)
                shares = 0 # Sell all shares
                trade_history.append({'Event' : 'SELL', 'Day' : day_index, 'Price' : float(np.round(current_price, 2)), 'Shares' : old_shares, 'Old Cash Balance' : float(np.round(old_cash, 2)), 'New Cash Balance' : float(np.round(cash, 2))})

                # print(trade_history[-1])

        # Record portfolio value at the end of each day
        portfolio_value_history.append(float(cash + shares * current_price))

    # Final portfolio value at the end of the backtest
    final_portfolio_value = float(cash + shares * values[-1])
    
    # Return final value, value history, trade history, and new dataframe
    return final_portfolio_value, portfolio_value_history, trade_history, data


# Backtest result display mechanism (for formatting)

def display_backtest_results(data, final_value, portfolio_history, trade_history, initial_cash=10000):
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)

    # 1. Overall Performance
    print(f"Initial Capital: ${initial_cash:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    total_return = ((final_value - initial_cash) / initial_cash) * 100
    print(f"Strategy Total Return: {total_return:,.2f}%")

    # Calculate and display Buy-and-Hold Performance
    # Use .item() to extract scalar Python float values from Series
    start_price = data['Close'].iloc[0].item()
    end_price = data['Close'].iloc[-1].item()
    
    # Handle division by zero if start_price is 0
    if start_price == 0:
        buy_and_hold_return = float('inf') if end_price > 0 else 0.0
    else:
        # Perform calculation
        buy_and_hold_return = float(((end_price - start_price) / start_price) * 100)


    start_date_str = data.index[0].strftime('%Y-%m-%d')
    end_date_str = data.index[-1].strftime('%Y-%m-%d')
    print(f"Buy-and-Hold Return ({start_date_str} to {end_date_str}): {buy_and_hold_return:,.2f}%")


    # May add annualized return later
    # num_years = (data.index[-1] - data.index[0]).days / 365.25
    # if num_years > 0:
    #     annualized_return = ((final_value / initial_cash)**(1/num_years) - 1) * 100
    #     print(f"Annualized Return: {annualized_return:,.2f}%")
    
    print("\n" + "="*50)
    print("TRADE LOG")
    print("="*50)

    # 2. Trade Log (Formatted)
    # Print a header for the trade log
    print(f"{'Event':<7} {'Day':<5} {'Price ($)':<12} {'Shares':<8} {'Old Cash ($)':<16} {'New Cash ($)':<16}")
    print("-" * 80)

    for trade in trade_history:
        event = trade['Event']
        day = trade['Day']
        price = trade['Price'] 
        shares = trade['Shares']
        old_cash = trade['Old Cash Balance']
        new_cash = trade['New Cash Balance']
        
        print(f"{event:<7} {day:<5} {price:<12.2f} {shares:<8} {old_cash:<16.2f} {new_cash:<16.2f}")
    
    print("-" * 80)
    print(f"Total Trades: {len(trade_history)}")
    print("="*50)
    return