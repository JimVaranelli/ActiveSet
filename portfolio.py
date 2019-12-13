import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from activeset import PortfolioOpt
from math import sqrt


# plot a bar chart for portfolio composition
def plot_portfolio_composition(df, wts, dir, ex, varx, port):
    wts *= 100
    df[df.index] = wts.reshape(wts.shape[0],).tolist()
    df = df.sort_values(ascending=True)
    # remove zero-weight entries
    mask = np.isclose(df.values, 0)
    df = df[~mask]
    pos = np.arange(df.shape[0])
    colors = []
    for val in df.values:
        if val < 0:
            colors.append('r')
        else:
            colors.append('g')
    plt.bar(pos, df.values, color=colors)
    plt.xticks(pos, df.index, rotation='vertical')
    plt.ylabel('Weight (%)')
    # annualize monthly variance
    if port == 'minrisk':
        str = 'Minimum-Risk Portfolio: E[x] = {0:.2f}% '.format(100*ex)
        str += 'vol={0:.2f}%'.format(100*sqrt(12*varx))
        plt.title(str)
    else:
        str = 'Maximum-Return Portfolio: E[x] = {0:.2f}% '.format(100*ex)
        str += 'vol={0:.2f}%'.format(100*sqrt(12*varx))
        plt.title(str)
    plt.savefig(os.path.join(dir, "./results/{}.png".format(port)))
    plt.show()


# Example portfolio optimization:
#   input:
#     60 monthly returns (10/2014-10/2019) for 50 S&P 500 stocks
#   calculated input:
#     covariance matrix
#     next-month forecasts
#   constraints:
#     5% liquidity buffer
#     dollar neutrality
#     short book bounds: -10% <= x <= 0%
#     long book bounds: 0 <= x <= 10%
#   output:
#     efficient frontier
#     minimum-risk portfolio weights
#     maximum-return portfolio weights
def main():
    print("Generating portfolio efficient frontier...")
    np.set_printoptions(precision=3)
    # read the stock price data set
    t0 = time.time()
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    file = os.path.join(cur_dir, "./results/stocks.csv")
    prices = pd.read_csv(file)
    # convert to returns
    returns = prices.select_dtypes(include=['number']).pct_change()[1:]
    # sort returns columns according to decreasing final
    # 6-period moving average window
    maprds = 6
    erdf = returns[-maprds:].mean().sort_values(ascending=False)
    R = np.asarray(returns[erdf.index])
    er = np.asarray(erdf)
    # get sample covariance matrix of returns
    Cov = np.cov(R, rowvar=False)
    # equality constraints:
    #   1) expected portfolio return
    Ce = er
    de = [[0]]
    #   2) long book portfolio weights with 5% liquidity buffer
    liqbuf = 0.05
    mid = int(Cov.shape[1] / 2)
    c = np.zeros(shape=(1, Cov.shape[1]))
    c[:, :mid] = 1
    Ce = np.vstack((Ce, c))
    de = np.vstack((de, [[1 - liqbuf]]))
    #   3) short book portfolio weights with 5% liquidity buffer
    c = np.zeros(shape=(1, Cov.shape[1]))
    c[:, mid:] = 1
    Ce = np.vstack((Ce, c))
    de = np.vstack((de, [[-(1 - liqbuf)]]))
    # bound constraints:
    #   long :    0 <= x <= 10%
    #  short : -10% <= x <= 0
    lim = 0.10
    lb = np.zeros(shape=(Cov.shape[1], 1))
    lb[mid:] = -lim
    ub = np.zeros(shape=(Cov.shape[1], 1))
    ub[:mid] = lim
    # calculate 100 efficient frontier steps.
    # trim infeasible programs at range extremes.
    numsteps = 100
    maxer = 0.5 * np.max(er[:mid]) - np.min(er[mid:])
    miner = 1.5 * np.min(er[:mid]) - np.max(er[mid:])
    step = (maxer - miner) / (numsteps - 1)
    cs = np.zeros(shape=(numsteps,))
    cs[0] = miner
    cs[1:] = step
    cs = np.cumsum(cs)
    # generate efficient frontier
    opt = PortfolioOpt()
    ef = []; sc = []; minx = []; maxx = []
    minvar = np.Inf; maxvar = -np.Inf
    miner = 0; maxer = 0
    for er in cs:
        # update expected return constraint
        # which is the first equality constraint
        de[0] = er
        print("  E[x] = {0:.5f}".format(er))
        # solve. first and last few iterations will
        # be infeasible so catch the exceptions.
        try:
            x, var, nit = opt(Cov, Ce, de, cu=ub, cl=lb)
        except ValueError:
            continue
        ef.append(er)
        sc.append(var)
        # minimum-risk portfolio
        if var < minvar:
            minx = x
            minvar = var
            miner = er
        # maximum-return portfolio
        maxx = x
        maxvar = var
        maxer = er
    print("elapsed time = {0:.3f}s".format(time.time() - t0))
    print("min: E[x] = {0:.5f} ".format(miner) + \
        "var = {0:.5f}".format(minvar))
    print("max: E[x] = {0:.5f} ".format(maxer) + \
        "var = {0:.5f}".format(maxvar))
    # plot the efficient frontier
    mask = ef >= miner
    sc = np.asarray(sc)[mask]
    ef = np.asarray(ef)[mask]
    plt.scatter(sc, ef, marker='+', color='r')
    plt.xlabel('var')
    plt.ylabel('E[x]')
    plt.xlim(0.9 * np.min(sc), 1.1 * np.max(sc))
    plt.title('Risk-Optimized Efficient Frontier')
    plt.savefig(os.path.join(cur_dir, "./results/efront.png"))
    plt.show()
    # plot the minimum-risk portfolio composition
    plot_portfolio_composition(erdf, minx, cur_dir, miner, minvar, 'minrisk')
    # plot the maximum-return portfolio composition
    plot_portfolio_composition(erdf, maxx, cur_dir, maxer, maxvar, 'maxret')


if __name__ == "__main__":
    sys.exit(int(main() or 0))