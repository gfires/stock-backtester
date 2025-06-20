import yfinance as yf
import pandas as pd
import strategies

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start = start_date, end = end_date)
    return stock_data

aapl_data = get_stock_data('AAPL', '2022-01-01', '2023-01-01')
simulation = strategies.moving_average_crossover(aapl_data)
strategies.display_backtest_results(aapl_data, simulation[0], simulation[1], simulation[2])
