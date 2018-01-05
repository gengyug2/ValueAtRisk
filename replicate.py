# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:11:08 2018

@author: gengy
"""

import pandas as pd
import pandas_datareader.data as web
import numpy as np
import datetime as dt
from scipy.special import ndtri


# Data Preparation
def fetch_data(ticker_list, start_date, end_date):
    # Read Tickers and Positions
    tickers = pd.read_csv(ticker_list)
    positions = np.array(tickers.Position.tolist())
    tickers = tickers.Symbol.tolist()
    # Request Adjusted and Closing Prices
    close = pd.DataFrame()
    adj = pd.DataFrame()
    for ticker in tickers:
        price_pad = web.DataReader(ticker, 'yahoo', start_date, end_date)
        close[ticker] = price_pad.Close
        adj[ticker] = price_pad['Adj Close']
        max_record = min(len(price_pad), len(close))
    close.fillna(method='ffill', inplace=True)
    adj.fillna(method='ffill', inplace=True)
    # Calculate Returns
    max_record -= 1
    return_pad = adj / adj.shift(1) - 1
    return_pad = return_pad.tail(max_record)
    return return_pad, positions, max_record


# Weighted Historical Simulation Approach. Defualt Significance Level is 95%
def hist_simul(return_pad, positions, max_record, eta=.99, signi=95):
    # Calculate Historical Daily Profits of Given Positions
    hist_profit = return_pad * positions
    hist_profit['total'] = hist_profit.apply(lambda x: x.sum(), axis=1)
    # Calculate Weights of Different Dates
    hist_weights = list(map(lambda x: eta ** (max_record - x - 1) * (1 - eta) / (1 - eta ** max_record),
                            [x for x in range(max_record)]))
    weighted_hist = np.column_stack((hist_profit.total.tolist(), hist_weights))
    # Sort Profits (Low to High)
    arg = np.argsort(weighted_hist[:, 0])
    weighted_hist = weighted_hist[arg]
    # Calculate Cumulative Weights
    cum_weights = np.cumsum(weighted_hist[:, 1], axis=0)
    weighted_hist = np.column_stack((weighted_hist, cum_weights))
    # Find Profits Around Significance Level
    arg = cum_weights <= (1 - signi / 100)
    arg2 = cum_weights >= (1 - signi / 100)
    # Interpolation
    ip_mat = np.row_stack((weighted_hist[arg][-1, [0, 2]], weighted_hist[arg2][0, [0, 2]]))
    hs_var = (ip_mat[1, 0] - ip_mat[0, 0]) / (ip_mat[1, 1] - ip_mat[0, 1]) * ((1 - signi / 100) - ip_mat[0, 1]) + ip_mat[0, 0]
    return hs_var


# Naive Historical Simulation Approach. Defualt Significance Level is 95%
def naive_hist(return_pad, positions, signi=95):
    hist_profit = return_pad * positions
    hist_total = hist_profit.sum(axis=1)
    nh_var = np.percentile(hist_total, 100 - signi)
    return nh_var


# Weighted Delta-Normal Simulation W or W/O RiskMetrics. Defualt Significance Level is 95%
def delta_norm(return_pad, positions, max_record, lmbd=.94, signi=95, rm_sign=1):
    # Employ RiskMetrics Weights
    dn_weights = list(map(lambda x: (1 - lmbd) * lmbd ** (max_record - x - 1), [x for x in range(max_record)]))
    if not rm_sign:
        dn_weights = 1 / max_record * np.ones(max_record)
    # Calculate Weighted/Unweighted COV, Assuming Zero-Mean
    weighted_cov = np.dot(return_pad.T * dn_weights, return_pad)
    delta = np.sqrt(np.dot(np.dot(positions, weighted_cov), positions))
    dn_var = delta * ndtri(1 - signi / 100)
    return dn_var


# Monte Carlo Approach. Defualt Significance Level is 95%
def monte_carlo(return_pad, positions, max_record, signi=95, iters=100000, lmbd=.94, rm_sign=1):
    # Calculate COV
    cov = return_pad.cov()
    if rm_sign:
        # Employ RiskMetrics Weights
        mc_weights = list(map(lambda x: (1 - lmbd) * lmbd ** (max_record - x - 1), [x for x in range(max_record)]))
        cov = np.dot(return_pad.T * mc_weights, return_pad)
    # Generate Simulated returns
    mc_simul = np.random.multivariate_normal(np.zeros(return_pad.shape[1]), cov, iters)
    simulated_profit = mc_simul * positions
    simulated_total = simulated_profit.sum(axis=1)
    mc_var = np.percentile(simulated_total, 100 - signi)
    return mc_var


def main():
    # Yahoo! Finance API is somehow unstable. Try for several times
    i = 0
    while True:
        try:
            return_pad, positions, max_record = fetch_data('./tickers.csv', '2015/9/1', dt.date.today())
            break
        except:
            i += 1
            if i >= 50:
                print('Yahoo! Finance API failed. Please try again later.')
    print('There are ', max_record, ' records available.')
    print()
    hs_var = -hist_simul(return_pad, positions, max_record)
    dn_var = -delta_norm(return_pad, positions, max_record)
    mc_var = -monte_carlo(return_pad, positions, max_record)
    print('Tomorrow VaR for the portfolio is $', round(hs_var, 2), 'using historical simulation method.')
    print('Tomorrow VaR for the portfolio is $', round(dn_var, 2), 'using delta-normal method.')
    print('Tomorrow VaR for the portfolio is $', round(mc_var, 2), 'using Monte Carlo method.')


if __name__ == '__main__':
    main()
