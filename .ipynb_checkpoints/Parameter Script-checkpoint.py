import pandas as pd
import numpy as np
import datetime as dt
import math

import OeconToolbox as ott
import LoanPortfolioTool as pft

# Corporate Issuer Parameters
V0 = 100      # Initial Asset Value
rf = .035     # Risk-Free Rate
rm = .105     # Market Return
beta = .8     # Firm CAPM Coefficient
sigma_i = .25 # Idiosyncratic Risk
sigma_m = .14 # Market Risk
ttm = 5       # Time to Maturity of Debt
penalty = 0   # Penalty on prepayment
loan_rating = 'B'
SPV_rating = 'B'

# SPV Parameters
J = 125    # number of loans
N = 50000 # Number of simulated portfolios

# Auxiliary
DefTable = pd.read_excel("CumulativeDefaultTable.xlsx",header=[0],index_col=[0],skipfooter=5,usecols="A:K")
mu = rf + beta * (rm - rf)
sigma = ott.sigma_beta_adj(beta, sigma_m, sigma_i)
face_value = ott.facevalue_from_probability(DefTable.loc[loan_rating, ttm] / 100, V0, ttm, mu, sigma)
market_value = ott.mv_bond(V0, face_value, ttm, rf, sigma)
b_loan_yield = ott.zero_yield(market_value, face_value, ttm)
seeds = 1234

###
### Estimating MV on callable bond
###

models = ott.Stochastic_Model(T = ttm, freq = 1, seed = seeds)
asset_paths_Q = models.GBM(n = N, v_0 = V0, mu = rf, sigma = sigma)
bond_paths_Q = ott.nc_loan_paths(asset_paths_Q, ttm, face_value, rf, sigma)
mv_call = models.callable_loan_value(v_0=V0, nc_yield=b_loan_yield,penalty=penalty,face_val=face_value,mu=rf,sigma=sigma,n=N, matrices=False)
b_callable_yield = ott.zero_yield(mv_call,face_value, ttm)

print("The market value of a callable loan is: {0:0.2f} with yield: {1:0.2f}%".format(mv_call, b_callable_yield*100))

# Results from Collateral Dynamics Notebook

###
### Loan Portfolios
###

loan_portfolios = pft.Loan_Portfolios(V = V0, B = face_value, T = ttm, rf = rf, rm = rm,
                                      beta = beta, sigmaM = sigma_m, sigmaI = sigma_i,
                                      j=J, n=N, seed=1234)

max_spv_cash_flows = face_value * J
initial_market_value = market_value * J

SPV_Q, _ = loan_portfolios.no_prepayments(risk_neutral = True, paths = False)
SPV_P, M = loan_portfolios.no_prepayments(risk_neutral = False, paths = False)

print("Mean payoffs under Q and P: {0:0.2f} and {1:0.2f} (without prepayments)".format(SPV_Q.mean(), SPV_P.mean()))

SPV_Q_pp, _, equity_Q_pp = loan_portfolios.with_prepayments(default_table = DefTable,
                                                            rating = loan_rating,
                                                            mv_callable = mv_call,
                                                            risk_neutral = True,
                                                            penalty = penalty)

SPV_P_pp, M_pp, equity_P_pp = loan_portfolios.with_prepayments(default_table = DefTable,
                                                               rating=loan_rating,
                                                               mv_callable=mv_call,
                                                               risk_neutral = False,
                                                               penalty=penalty)

print("Mean payoffs under Q and P: {0:0.2f} and {1:0.2f} (with prepayments)".format(SPV_Q_pp.mean(), SPV_P_pp.mean()))

SPV_MV = np.minimum(SPV_Q, np.quantile(SPV_P,DefTable.loc[SPV_rating,ttm]/100)).mean() * np.exp(-rf*ttm)
SPV_MV_pp = np.minimum(SPV_Q_pp, np.quantile(SPV_P_pp,DefTable.loc[SPV_rating,ttm]/100)).mean() * np.exp(-rf*ttm)

wacd = ott.zero_yield(SPV_MV,
                      np.quantile(SPV_P,
                                  DefTable.loc[SPV_rating,ttm]/100),
                      ttm)
adj_wacd = ott.zero_yield(SPV_MV_pp,
                          np.quantile(SPV_P,
                                      DefTable.loc[SPV_rating,ttm]/100),
                          ttm)

print("Results:\n---------\n 1) Yield: {0:0.2f}%,\n 2) Adj. Yield: {1:0.2f}% and\n 3) Mispricing: {2:0.2f}%".format(wacd*100,adj_wacd*100,(adj_wacd-wacd)*100))

