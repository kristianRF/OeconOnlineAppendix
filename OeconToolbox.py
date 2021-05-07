import pandas as pd
import numpy as np
from scipy.stats import norm

def sigma_beta_adj(beta: float, sigmaM: float, sigmaI: float):
    """
    Returns sigma adjusted for beta
    ______________________________________________________________
    Parameters:
        beta (float): CAPM Beta
        sigmaM (float): Market Risk
        sigmaI (float): Idiosyncratic Risk
    Returns:
        Sigma adjusted for Beta
    """
    return ((beta * sigmaM) ** 2 + sigmaI ** 2) ** 0.5

def facevalue_from_probability(DefProb: float, V: float, T: float, mu: float, sigma: float):
    """
    Returns the fair face value of debt from a given default probability
    ______________________________________________________________
    Parameters:
        DefProb (float): The default probability of the firm
        V (float): Value of the firm
        T (float): Number of periods
        mu (float): Drift of the firm-value
        sigma (float): Volatility of the firm-value
    Returns:
        The estimated face value
    """
    return V / np.exp(-norm.ppf(DefProb) * sigma * (T ** 0.5) - (mu - 0.5 * sigma ** 2) * T)

def mv_bond(V: float, D: float, T: float, r: float, sigma: float):
    """
    The market value of the rating k reference bond, following from Merton (1974)
    ______________________________________________________________
    Parameters:
        V (float): Initial firm value
        D (float): Face value of debt
        T (float): Time to maturity
        r (float): The risk-free interest rate
        sigma (float): The volatility of the underlying asset
    Returns:
        Market value of debt under the risk-neutral Q-measure
    """
    
    d1 = (np.log(V / D) + (r + 0.5 * sigma ** 2) * T) / (sigma * (T ** 0.5))
    d2 = (np.log(V / D) + (r - 0.5 * sigma ** 2) * T) / (sigma * (T ** 0.5))

    def phi(x): return norm.cdf(x)
    
    return D * np.exp(-r * T) * phi(d2) + V * phi(-d1)

def zero_yield(V0: float, V: float, T: float = 1):
    """
    Returns the yield of a zero-coupon bond
    ______________________________________________________________
    Parameters:
        V0 (float): Initial Value
        VT (float): Terminal Value
        T (float) = number of periods
    Returns:
        Annual yield of a zero-coupon bond
    """
    return np.log(V / V0) / T

class Stochastic_Model:
    """
    A class containing models of stochastic processes
    ______________________________________________________________
    
    Attributes:
        T (int):  The number of time periods to generate data for
        freq (int): Granularity of the process, per time period
        seed (int): Random seed
    """
    def __init__(self, T: int = 1, freq: int = 12, seed: int = 1234):
        self.T = T
        self.freq = freq
        self.dt = (1/freq)
        self.T_array = np.linspace(self.dt, T, freq * T).reshape(freq * T, 1)
        self.seed = seed
        
    def noise(self, n: int = 10):
        """
        Simulation of standard normal random distributed noise with mean = 0 and standard devation sqrt(dt)
        ______________________________________________________________
        Parameters:
            n (int): The number of scenarios
        Returns: 
            A numpy array of (1 + freq * T) x n rows
        """
        rnd = np.random.RandomState(self.seed)
        return rnd.normal(loc = 0, scale = np.sqrt(self.dt), size=(1 + self.T_array.shape[0], n))

    def GBM(self, n: int = 10, v_0: float = 1, mu: float = 0.0, sigma: float = 0.10):
        """
        Monte Carlo simulation of a Geometric Brownian Motion
        ______________________________________________________________
        Parameters:
            n (int): The number of scenarios
            v_0 (float): Initial Value
            mu (float): Drift
            sigma (float): Volatility
        Returns: 
            A numpy array of (1 + freq * T) columns and n rows
        """
        mean = (mu-0.5*(sigma**2)) * self.dt
        increments = mean + self.noise(n = n) * sigma
        increments[0] = 0
        return v_0 * np.exp(increments.cumsum(axis=0))
    
    def CIR(self, n: int = 10, r_0: float = 0.03, alpha: float = 0.1, beta: float = 0.03, sigma: float = 0.05):
        """
        Monte Carlo simulation of a Cox-Ingersoll-Ross process (1985)
        ______________________________________________________________
        Parameters:
            n (int): The number of scenarios
            r_0 (float): Initial level (annualized rate)
            alpha (float): Mean-reversion coefficient
            beta (float): Long-run mean
            sigma (float): Volatility
        Returns: 
                rates, prices
            Where:
                rates (np.ndarray): An (1 + freq * T) x n np.ndarray of the rates
                prices (np.ndarray): An (1 + freq * T) x n np.ndarray of the bond prices
        """
        def price_from_rate(ttm, r):
            h = np.sqrt(alpha**2 + 2*sigma**2)
            A = ((2*h*np.exp((h+alpha)*ttm/2)) / (2*h+(h+alpha)*(np.exp(h*ttm)-1)))**(2*alpha*beta/sigma**2)
            B = (2*(np.exp(h*ttm)-1))/(2*h + (h+alpha)*(np.exp(h*ttm)-1))
            return A * np.exp(-B * r)
        
        noise = self.noise(n=n)
        
        r_0 = np.log1p(r_0) # Annualized Rate to short rate, is reversed at the end
        rates, prices = np.empty_like(noise), np.empty_like(noise)
        rates[0], prices[0] = r_0, price_from_rate(self.T, r_0)
        
        for t in range(1, rates.shape[0]):
            current_rate = rates[t - 1]
            rates[t] = current_rate + alpha * (beta - current_rate) * self.dt + sigma * np.sqrt(current_rate) * noise[t]
            prices[t] = price_from_rate(self.T - (t * self.dt),rates[t])
            
        return np.expm1(rates), prices
    
    def callable_loan_value(self, v_0: float,  nc_yield: float, penalty: float, face_val: float, mu: float, sigma: float, n: int = 250000, matrices: (bool) = False):
        """
        Returns the market values of a callable loan
        ______________________________________________________________
        Parameters:
            v_0 (float): Initial Value
            nc_yield (float): The yield on a non-callable bond
            penalty (float): Penalty cost of refinancing
            face_val (float): The principal of the loan
            mu (float): Drift of asset values
            sigma (float): The total asset volatility
            n (int): Number of simulations
            matrices (bool): If True then return call and cash flow matrices
        Returns:
            if matrices == False:
                Returns a numpy array of the loan market paths
            if matrices == True:
                Returns two numpy arrays of the optimal call matrix and the bond paths
        """
        ttm = self.T
            
        Vt = self.GBM(n, v_0, mu, sigma)
        Bt = nc_loan_paths(Vt, ttm, face_val, mu, sigma)
            
        call_mat = call_matrix(Bt, face_val, nc_yield, ttm, penalty)
        bond_cash_flows = cash_flow_matrix(Bt, call_mat, face_val, nc_yield, ttm)
            
        callable_bond_mv = np.exp(-mu * np.linspace(0, ttm, ttm+1)).reshape((ttm+1,1)) * bond_cash_flows
        
        if matrices == False:
            return callable_bond_mv.sum(axis=0).mean()
        else:
            return call_mat, Bt
        
def nc_loan_paths(Vt: np.ndarray, ttm: int, face_val: float, rf: float, sigma: float):
    """
    Returns the market value paths of a non-callable loan
    ______________________________________________________________
    Parameters:
        Vt (np.ndarray): A numpy array of the asset values
        ttm (int): Time to maturity
        face_val (float): The principal of the loan
        rf (float): The risk-free rate
        sigma (float): The total asset volatility
    Returns:
        Returns a numpy array of the loan market paths
    """
    Bt = np.zeros_like(Vt)
    
    for t in range(0, ttm + 1):
        if t == ttm: Bt[t] = np.minimum(Vt[ttm], face_val)
        else: Bt[t] = mv_bond(Vt[t], face_val, (ttm - t), rf, sigma)
            
    return Bt

def call_matrix(Bt: np.ndarray, face_val: float, yld: float, ttm: int, penalty: float):
    """
    Returns the call matrix for optimal redemption of a call
    ______________________________________________________________
    Parameters:
        Bt (np.ndarray): A numpy array of the bond values
        face_val (float): The principal of the loan
        yld (float): The per-period continuously compounded bond yield
        ttm (int): Time to maturity
        penalty (float): Penalty cost of refinancing
    Returns:
        Returns a numpy array of the call or repayment dates of each path
    """
    Ct = np.zeros_like(Bt)
    
    for t in range(1, ttm + 1):
        cond1 = (Ct[:t].sum(axis = 0) == 0)
        cond2 = (Bt[t] - (face_val * np.exp(-yld * (ttm - t))) - penalty > 0)
        if t == ttm: Ct[t] = np.where(cond1, 1, 0)
        else: Ct[t] = np.where(cond1 & cond2, 1, 0)
            
    return Ct

def cash_flow_matrix(Bt: np.ndarray, Ct: np.ndarray, face_val: float, yld: float, ttm: int):
    """
    Returns a numpy array of bond cash flows for a given principal and yield
    ______________________________________________________________
    Parameters:
        Bt (np.ndarray): A numpy array of the bond values
        Ct (np.ndarray): A numpy array of the call dates
        face_val (float): The principal of the loan
        yld (float): The per-period continuously compounded bond yield
        ttm (int): Time to maturity
    Returns:
        Returns a numpy array of the callable bond cash flows
    """
    cf_t = np.zeros_like(Ct)
    
    for t in range(0, ttm + 1):
        if t == ttm: cf_t[t] = Bt[t] * Ct[t]
        else: cf_t[t] = face_val * np.exp(-yld * (ttm - t)) * Ct[t]
            
    return cf_t

def loan_portfolio(j: int = 125, n: float = 1000, V: float = 100, B: float = 50, T: float = 1, rf: float = 0, rm: float = .01, beta: float = 1, sigmaI: float = 0, sigmaM: float = 0, risk_neutral: bool = False, paths: bool = False, seed: int = 1234):
    """
    Generates n random outcomes of cash flows in a SPV loan portfolio
    ______________________________________________________________
    Parameters:
        j (int): Number of loans
        n (int): number of simulations
        V (float): The total asset value
        B (float): The face value of debt
        T (float): Number of periods
        rf (float): The risk free rate
        rm (float): The market return
        beta (float): CAPM Beta
        sigmaM (float): Market Risk
        sigmaI (float): Idiosyncratic Risk
        risk_neutral (bool): True if using the risk-neutral Q-measure
        paths (bool): if True then return portfolio paths, if False return sorted cash flow matrix
        seed (int): Random seed
    Returns:
        if paths == True:
            n-simulations of the SPV portfolio value in all periods
        if paths == False:
            The final payoffs from the SPV as well as the market factor
    """
    # Auxiliary equations:
    sigma_star = sigma_beta_adj(beta, sigmaM, sigmaI)
    mu = rf + beta * (rm - rf)
    
    random_state = np.random.RandomState(seed)
    random = random_state.standard_normal(size=(T + 1, n, j + 1))
    
    def asset_gbm(x,ttm):
        drift = (x - 0.5 * sigma_star ** 2)
        vol_M = (beta * sigmaM * random[:,:,0].reshape((T + 1,n,1)))
        vol_I = (sigmaI * random[:,:,1:])
        increments = drift + vol_M + vol_I
        increments[0] = 0
        return V * np.exp(increments.cumsum(axis=0))
    
    if risk_neutral == False: asset_paths = asset_gbm(mu,T)
    else: asset_paths = asset_gbm(rf,T)
            
    if paths == False: 
        terminal_value = asset_paths[-1,:,:]
        cash_flows = np.minimum(terminal_value, B).sum(axis=1)
        sort = cash_flows.argsort()
        return cash_flows[sort], random[1:,:,0].sum(axis=0)[sort]
    else:                                              
        bond_paths = np.zeros_like(asset_paths)
        for t in range(0, T + 1):
            if t == T: bond_paths[t,:,:] = np.minimum(asset_paths[t,:,:], B)
            else: bond_paths[t,:,:] = mv_bond(asset_paths[t,:,:], B, (T - t), rf, sigma_star)
        return bond_paths.sum(axis=2)