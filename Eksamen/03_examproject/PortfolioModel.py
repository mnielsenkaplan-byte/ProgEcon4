""" 
given portfolio code
"""

from types import SimpleNamespace
import numpy as np

class PortfolioModelClass:
    """ a portfolio of a risky and a safe asset with a rebalancing rule """

    def __init__(self,**kwargs):
        """ set the default parameters, then overwrite with any keyword arguments """

        par = self.par = SimpleNamespace()

        # a. returns
        par.mu = 0.05 # mean log return on the risky asset
        par.sigma = 0.20 # standard deviation of the log return on the risky asset
        par.r = 0.01 # log return on the safe asset

        # b. the rebalancing rule
        par.theta_star = 0.50 # target share of wealth in the risky asset
        par.Delta = 0.10 # width of the no-trade band
        par.tau = 0.01 # proportional transaction cost

        # c. preferences
        par.gamma = 3.0 # relative risk aversion

        # d. simulation settings
        par.W0 = 1.0 # initial wealth
        par.T = 40 # number of periods
        par.N = 50_000 # number of simulated portfolios
        par.seed = 2026 # seed for the random number generator

        # e. overwrite with keyword arguments, e.g. PortfolioModelClass(Delta=0.0)
        for key,value in kwargs.items(): setattr(par,key,value)

        # f. empty container for simulation results
        self.sim = SimpleNamespace()

    def __str__(self):
        """ called when using print """

        par = self.par

        text = 'Portfolio model with:\n'
        text += f'  mu    = {par.mu:.4f}, sigma = {par.sigma:.4f}, r = {par.r:.4f}\n'
        text += f'  theta_star = {par.theta_star:.4f}, Delta = {par.Delta:.4f}, tau = {par.tau:.4f}\n'
        text += f'  gamma = {par.gamma:.4f} (relative risk aversion)\n'
        text += f'  W0 = {par.W0:.2f}, T = {par.T}, N = {par.N:,}, seed = {par.seed}'

        return text

    def draw_returns(self):
        """ draw the gross return on the risky asset in all periods and all portfolios

        Returns:

            (ndarray): gross returns with shape (N,T)

        """

        par = self.par

        rng = np.random.default_rng(par.seed)
        eps = rng.normal(size=(par.N,par.T))

        return np.exp(par.mu + par.sigma*eps)

    def u(self,W):
        """ CRRA utility of wealth """

        par = self.par

        return W**(1-par.gamma)/(1-par.gamma)
    # NotImplementedError (student) section, using given portfolio code for form and implementation
    # the share of wealth in the risky asset after trading, and the amount traded
    def trade(self,theta):
        """ apply the no-trade-band rule to the share held before trading

        Returns:
            (tuple): (theta_post, amount_traded)
                theta_post (ndarray): share of wealth in the risky asset, after trading
                amount_traded (ndarray): |theta_post - theta|, the share of wealth traded
        """

        par = self.par

        # checks portfolios that are out of the no-trade band
        out_of_band = np.abs(theta-par.theta_star) > par.Delta

        # trade only with those out of band
        theta_post = np.where(out_of_band,par.theta_star,theta)

        # how much was tradeds
        amount_traded = np.abs(theta_post-theta)

        return theta_post,amount_traded

    # simulate all N portfolios forward T periods
    def simulate(self,R=None):
        """ simulate N portfolios forward T periods under the rebalancing rule

        Returns:
            Results are stored in self.sim:
                W (ndarray): wealth, shape (N,T+1), W[:,0] = par.W0
                theta (ndarray): share in the risky asset, shape (N,T+1), before trading
                trades (ndarray): 1 if traded that period else 0, shape (N,T)
                dist (ndarray): |theta-theta_star| before trading, shape (N,T)
        """

        par = self.par
        sim = self.sim

        # the returns to use
        if R is None: R = self.draw_returns()
        Rf = np.exp(par.r)

        # containers for W and theta with columns for period 0 through period T
        W = np.empty((par.N,par.T+1))
        theta = np.empty((par.N,par.T+1))
        trades = np.empty((par.N,par.T))
        dist = np.empty((par.N,par.T))

        # conditions
        W[:,0] = par.W0
        theta[:,0] = par.theta_star

        # vectorized portfolio loop
        for t in range(par.T):

            dist[:,t] = np.abs(theta[:,t]-par.theta_star)

            theta_post,amount_traded = self.trade(theta[:,t])
            trades[:,t] = amount_traded > 0

            W_post = W[:,t]*(1-par.tau*amount_traded)

            W[:,t+1] = theta_post*W_post*R[:,t] + (1-theta_post)*W_post*Rf
            theta[:,t+1] = theta_post*W_post*R[:,t]/W[:,t+1]

        # store
        sim.R = R
        sim.W = W
        sim.theta = theta
        sim.trades = trades
        sim.dist = dist

    # the numbers to report for a rule, including expected utility
    def summary(self):
        """ the six numbers to report for the rule, from the latest .simulate()

        Returns:
            (SimpleNamespace): avg_trades, avg_dist, mean_WT, median_WT, p10_WT, EU
        """

        sim = self.sim
        rep = SimpleNamespace()

        WT = sim.W[:,-1]
        # six numbers to report
        rep.avg_trades = sim.trades.sum(axis=1).mean()   # average number of trades
        rep.avg_dist = sim.dist.mean()                   # average distance to target befor trade
        rep.mean_WT = WT.mean()                          # mean Wt
        rep.median_WT = np.median(WT)                    # median wt
        rep.p10_WT = np.percentile(WT,10)                # 10th percentile Wt
        rep.EU = self.u(WT).mean()                       # expected utility

        return rep