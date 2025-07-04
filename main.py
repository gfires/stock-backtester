import portfolio

allocation = {'AAPL' : 0.25, 'IBM' : 0.25, 'MSFT' : 0.4, 'UBER' : 0.1}
test_portfolio = portfolio.Portfolio(allocation, '2022-01-01', '2023-01-01')
test_portfolio.moving_average_crossover()
test_portfolio.display_backtest_results()
test_portfolio.plot_portfolio_value()
