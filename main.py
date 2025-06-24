import portfolio

test_portfolio = portfolio.Portfolio('AAPL', '2022-01-01', '2023-01-01')
test_portfolio.moving_average_crossover()
test_portfolio.display_backtest_results()
test_portfolio.plot_portfolio_value()
