import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

class Portfolio:

    def __init__(self, ticker_allocation, start_date, end_date, start_cash=10000, commission_rate=0.001):

        self.shares = {}
        self.data = {}
        self.ticker_allocation = ticker_allocation
        for ticker in ticker_allocation:
            self.data[ticker] = self.get_yfinance_data(ticker, start_date, end_date)
            self.shares[ticker] = 0
        self.commission_rate = float(commission_rate)
        self.cash = float(start_cash)
        self.initial_cash = float(start_cash)
        self.portfolio_value_history = []
        self.trade_history = []

    def get_yfinance_data(self, ticker, start_date, end_date):
        return yf.download(ticker, start = start_date, end = end_date, auto_adjust = False)

    def moving_average_crossover(self, short_term_period=20, long_term_period=50):
        
        # Initialize values array w/ closing prices and total_num_days variable
        self.portfolio_value_history = []
        self.trade_history = []

        # Initialize strategy data as dictionary, taking data only for common dates
        self.strategy_data = {}
        common_dates = self.data[next(iter(self.data))].index
        for df in self.data.values():
            common_dates = common_dates.intersection(df.index)

        # For each ticker, compute short and long avgs to construct strategy_data dictionary of DFs
        for ticker in self.data:
            ticker_df = self.data[ticker].loc[common_dates].copy()
            ticker_df['SMA_Short'] = ticker_df['Close'].rolling(window=short_term_period).mean()
            ticker_df['SMA_Long'] = ticker_df['Close'].rolling(window=long_term_period).mean()
            self.strategy_data[ticker] = ticker_df

        # Loop through dates, executing the strategy for each stock
        for day_idx, current_date in enumerate(common_dates):
            for ticker, weight in self.ticker_allocation.items():
                ticker_df = self.strategy_data[ticker]
                if day_idx >= long_term_period:
                    row = ticker_df.iloc[day_idx]
                    prev_row = ticker_df.iloc[day_idx - 1]
                    price = row['Close']
                    short_ma = row['SMA_Short']
                    long_ma = row['SMA_Long']
                    prev_short_ma = prev_row['SMA_Short']
                    prev_long_ma = prev_row['SMA_Long']

                    # Skip if any of the SMAs are NaN
                    if pd.isna(short_ma) or pd.isna(long_ma) or pd.isna(prev_short_ma) or pd.isna(prev_long_ma):
                        continue

                    
                    # Generate trading signal
                    signal = 0
                    if prev_short_ma <= prev_long_ma and short_ma > long_ma:
                        signal = 1
                    elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
                        signal = -1

                    # Save stock price for reference
                    price = row['Close']

                    # Execute trade if non-zero signal
                    if signal == 1 and self.cash > 0:
                        # Calculate maximum shares we can buy
                        allocatable_cash = self.initial_cash * weight
                        available_cash = min(allocatable_cash, self.cash)
                        buyable_shares = int(available_cash // (price * (1 + self.commission_rate)))
                        if buyable_shares > 0:
                            stock_cost = buyable_shares * price
                            commission_cost = stock_cost * self.commission_rate
                            self.shares[ticker] += buyable_shares
                            old_cash = float(self.cash)
                            self.cash -= float(stock_cost + commission_cost)
                            self.trade_history.append({
                                'Event': 'BUY',
                                'Ticker': ticker,
                                'Day': day_idx,
                                'Price': round(price, 2),
                                'Shares': buyable_shares,
                                'Old Cash Balance': round(old_cash, 2),
                                'New Cash Balance': round(self.cash, 2)
                            })

                    elif signal == -1 and self.shares[ticker] > 0:
                        proceeds = self.shares[ticker] * price
                        transaction_cost = proceeds * self.commission_rate
                        old_cash = float(self.cash)
                        self.cash += float(proceeds - transaction_cost)
                        self.trade_history.append({
                            'Event': 'SELL',
                            'Ticker': ticker,
                            'Day': day_idx,
                            'Price': round(price, 2),
                            'Shares': sell_shares,
                            'Old Cash Balance': round(old_cash, 2),
                            'New Cash Balance': round(self.cash, 2)
                        })                        
                        self.shares[ticker] = 0 # Sell all shares

            # Record portfolio value at the end of each day
            total_value = self.cash
            for ticker in self.data:
                price = self.strategy_data[ticker].loc[current_date, 'Close']
                total_value += self.shares[ticker] * price
            self.portfolio_value_history.append(total_value)

        # Final portfolio value at the end of the backtest
        total_portfolio_value = self.portfolio_value_history[-1]
        self.total_portfolio_value = total_portfolio_value


    def display_backtest_results(self):
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        # 1. Overall Portfolio Performance
        print(f"Initial Capital: ${self.initial_cash:,.2f}")
        print(f"Final Portfolio Value: ${self.total_portfolio_value:,.2f}")
        total_return = ((self.total_portfolio_value - self.initial_cash) / self.initial_cash) * 100
        print(f"Strategy Total Return: {total_return:,.2f}%")

        print("\n" + "=" * 60)
        print("BUY-AND-HOLD BENCHMARK (PER STOCK)")
        print("=" * 60)

        blended_final_value = 0.0

        for ticker, weight in self.ticker_allocation.items():
            df = self.data[ticker]
            start_price = df['Close'].iloc[0]
            end_price = df['Close'].iloc[-1]

            if start_price == 0:
                bh_return = float('inf') if end_price > 0 else 0.0
                final_value = self.initial_cash * weight  # assume 100% gain or loss
            else:
                bh_return = ((end_price - start_price) / start_price) * 100
                alloc_cash = self.initial_cash * weight
                final_value = alloc_cash * (end_price / start_price)

            blended_final_value += final_value

            start_date_str = df.index[0].strftime('%Y-%m-%d')
            end_date_str = df.index[-1].strftime('%Y-%m-%d')

            print(f"{ticker:<8} | Weight: {weight:.2%} | "
                f"Return: {bh_return:6.2f}% | "
                f"({start_date_str} → {end_date_str}) | "
                f"Final Value: ${final_value:,.2f}")

        blended_return = ((blended_final_value - self.initial_cash) / self.initial_cash) * 100

        print("-" * 60)
        print(f"Blended Buy-and-Hold Final Value: ${blended_final_value:,.2f}")
        print(f"Blended Buy-and-Hold Return:     {blended_return:,.2f}%")

        print("\n" + "=" * 60)
        print("TRADE LOG")
        print("=" * 60)

        # 3. Trade Log Header
        print(f"{'Event':<7} {'Ticker':<8} {'Day':<5} {'Price ($)':<12} {'Shares':<8} {'Old Cash ($)':<16} {'New Cash ($)':<16}")
        print("-" * 90)

        for trade in self.trade_history:
            event = trade['Event']
            ticker = trade['Ticker']
            day = trade['Day']
            price = trade['Price']
            shares = trade['Shares']
            old_cash = trade['Old Cash Balance']
            new_cash = trade['New Cash Balance']

            print(f"{event:<7} {ticker:<8} {day:<5} {price:<12.2f} {shares:<8} {old_cash:<16.2f} {new_cash:<16.2f}")

        print("-" * 90)
        print(f"Total Trades: {len(self.trade_history)}")
        print("=" * 60)

    
    def plot_portfolio_value(self):
        if not self.portfolio_value_history:
            print("Portfolio value history is empty. Run a strategy first.")
            return

        # Step 1: Use common dates as the timeline (already aligned in strategy_data)
        common_dates = self.strategy_data[next(iter(self.strategy_data))].index
        if len(common_dates) != len(self.portfolio_value_history):
            print("Mismatch in common dates and portfolio value history length.")
            return

        # Step 2: Compute the blended buy-and-hold value for each day
        bh_values = []
        for date in common_dates:
            total_value = 0.0
            for ticker, weight in self.ticker_allocation.items():
                df = self.data[ticker]
                start_price = df['Close'].loc[common_dates[0]]
                if start_price == 0:
                    continue
                alloc_cash = self.initial_cash * weight
                shares_held = alloc_cash / start_price
                current_price = df['Close'].loc[date]
                total_value += shares_held * current_price
            bh_values.append(total_value)

        # Step 3: Plot both curves
        plt.figure(figsize=(12, 6))
        plt.plot(common_dates, self.portfolio_value_history, label='Portfolio Value', color='blue', linewidth=2)
        plt.plot(common_dates, bh_values, label='Buy-and-Hold Benchmark', color='gray', linestyle='--')

        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
            
