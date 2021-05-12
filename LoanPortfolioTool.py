import pandas as pd
import numpy as np
import datetime as dt
import math
# Import Custom Module
import OeconToolbox as ott

class Loan_Portfolios:
    """
    A class containing models of a loan portfolios, with and without collateral prepayments.
    ______________________________________________________________
    
    Attributes:
        Corporate Issuer Dynamics:
            V (float): The total asset value
            B (float): The face value of debt
            T (int): Number of periods
            rf (float): The risk free rate
            rm (float): The market return
            beta (float): CAPM Beta
            sigmaM (float): Market Risk
            sigmaI (float): Idiosyncratic Risk
        Simulation Parameters:
            j (int): Number of loans
            n (int): number of simulations
            seed (int): Random seed
    """
    def __init__(self, V: float = 100, B: float = 50, T: int = 5, rf: float = 0.01, rm: float = 0.02, beta: float = 1, sigmaM: float = 0.1, sigmaI: float = 0.1, j: int = 100, n: int = 1000, seed: int = 1234):
        # Corporate Issuer Dynamics
        self.__V0 = V
        self.__FV = B
        self.__ttm = T
        self.rf = rf
        self.rm = rm
        self.beta = beta
        self.sigma_m = sigmaM
        self.sigma_i = sigmaI
        # Auxiliary Equations
        self.__sigma = ott.sigma_beta_adj(beta, sigmaM, sigmaI)
        self.__mu = rf + beta * (rm - rf)
        self.__mv = ott.mv_bond(V, B, T, rf, self.__sigma)
        self.__ync = ott.zero_yield(self.__mv, B, T)
        # Simulation Parameters
        self.J = j
        self.N = n
        self.__random_state = np.random.RandomState(seed)
        self.random = self.__random_state.standard_normal(size=(T + 1, n, j + 1))
        
    def vanilla_gbm(self, risk_neutral: bool = False, ttm: int = 1):
        """
        Simulates a simple Geometric Brownian Motion
        ______________________________________________________________
        Parameters:
            risk_neutral (bool): If True drift is risk-free rate
            ttm (int): time to maturity
        """
        if risk_neutral == True: var = self.rf
        else: var = self.__mu
        drift = (var - 0.5 * self.__sigma ** 2)
        diffusion = (self.beta * self.sigma_m * self.random[:,:,0].reshape((self.__ttm + 1,self.N,1))) + (self.sigma_i * self.random[:,:,1:])
        increments = drift + diffusion
        increments[0] = 0
        return self.__V0 * np.exp(increments.cumsum(axis=0))
    
    def no_prepayments(self, risk_neutral: bool = False, paths: bool = False):
        """
        Generates n random outcomes of cash flows in a SPV loan portfolio without prepayments
        ______________________________________________________________
        Parameters:
            risk_neutral (bool): True if using the risk-neutral Q-measure
            paths (bool): if True then return portfolio paths, if False return sorted cash flow matrix
        Returns:
            if paths == True:
                n-simulations of the SPV portfolio value in all periods
            if paths == False:
                The final payoffs from the SPV as well as the market factor
        """
        asset_paths = self.vanilla_gbm(risk_neutral,self.__ttm)
                    
        if paths == False: 
            terminal_value = asset_paths[-1,:,:]
            cash_flows = np.minimum(terminal_value, self.__FV).sum(axis=1)
            sort = cash_flows.argsort()
            return cash_flows[sort], self.random[1:,:,0].sum(axis=0)[sort]
        else:                                              
            bond_paths = np.zeros_like(asset_paths)
            for t in range(0, self.__ttm + 1):
                if t == T: bond_paths[t,:,:] = np.minimum(asset_paths[t,:,:], self.__FV)
                else: bond_paths[t,:,:] = ott.mv_bond(asset_paths[t,:,:], self.__FV, (self.__ttm - t), self.rf, self.__sigma)
            return bond_paths.sum(axis=2)
        
    def call_barrier(self, penalty: float):
        """
        Estimate the prepayment barrier as a (ttm + 1 x 1) np.ndarray
        """
        call_barrier = (self.__FV * np.exp(-self.__ync* np.linspace(0,self.__ttm,self.__ttm+1)) - penalty).reshape((self.__ttm+1,1))
        return np.flip(call_barrier)
                                                                                                   
    def nc_loans(self, default_df, rating: str = 'B'):
        """
        Returns a pandas dataframe of face values, market values and yields.
        """
        PD, FV, MV, Y = [np.zeros(self.__ttm-1) for _ in range(4)]

        for t in range (1, self.__ttm):
            PD[t-1] = default_df.loc[rating, t] / 100
            FV[t-1] = ott.facevalue_from_probability(PD[t-1], self.__V0, t, self.__mu, self.__sigma)
            MV[t-1] = ott.mv_bond(self.__V0, FV[t-1], t, self.rf, self.__sigma)
            Y[t-1] = ott.zero_yield(MV[t-1],FV[t-1],t)
        return pd.DataFrame({'PD':PD,'FV':FV,'MV':MV,'Y':Y},list(reversed(range(1,self.__ttm))))
    
    def prepayment_gbm(self,call_mat, risk_neutral: bool = False, ttm: int = 1):
        """
        Simulates a simple Geometric Brownian Motion, with asset value resets at call dates
        ______________________________________________________________
        Parameters:
            call_mat (np.ndarray): A numpy matrix, indicating call periods
            risk_neutral (bool): If True drift is risk-free rate
            ttm (int): time to maturity
        """
        if risk_neutral == True: var = self.rf
        else: var = self.__mu
        drift = (var - 0.5 * self.__sigma ** 2)
        diffusion = (self.beta * self.sigma_m * self.random[:,:,0].reshape((self.__ttm + 1,self.N,1))) + (self.sigma_i * self.random[:,:,1:])
        increments = drift + diffusion
        increments[0] = 0
        __paths = np.zeros_like(self.random[:,:,1:])
        __paths[0,:,:] = self.__V0 
        
        for t in range(1,self.__ttm):
            __paths[t,:,:] = np.where(call_mat[t,:,:] == 1, self.__V0, __paths[t-1,:,:] * np.exp(increments[t,:,:]))
        
        __paths[self.__ttm,:,:] = __paths[self.__ttm-1,:,:] * np.exp(increments[self.__ttm,:,:])
        
        return __paths

    def with_prepayments(self, default_table, rating: str ='B', mv_callable: float = 50, risk_neutral: bool = False, penalty: float = 0):
        """
        Generates n random outcomes of cash flows in a SPV loan portfolio with prepayments
        ______________________________________________________________
        Parameters:
            default_table (pd.DataFrame): The cumulative default table, with ratings as index and time-to-maturity as columns.
            rating (str): The rating of the newly bought loan
            mv_callable (float): The equilibrium market value of a callable loan
            risk_neutral (bool): True if using the risk-neutral Q-measure
            penalty (float): Optional Penalty to refinancing
        Returns:
            The final payoffs from the SPV, the market factor and distributions to equity
        """
        asset_paths = self.vanilla_gbm(risk_neutral,self.__ttm)
        nc_loans = self.nc_loans(default_table, rating)
        call_barrier = self.call_barrier(penalty)
        __yc = ott.zero_yield(mv_callable, self.__FV, self.__ttm)
        
        nc_loan_paths, call_mat, nc_cfs_mat,  pf_cfs, call_cfs_mat = [np.zeros_like(asset_paths) for _ in range(5)]
        
        face_values_mat = np.ones((self.N,self.J)) * self.__FV
        cfs_distributions = np.zeros((self.__ttm,self.N))
        
        for t in range(0, self.__ttm + 1):
            cond1 = (call_mat[:t,:,:].sum(axis = 0) == 0)
            if t == 0:
                nc_loan_paths[t,:,:] = ott.mv_bond(asset_paths[t,:,:], self.__FV, (self.__ttm - t), self.rf, self.__sigma)
                call_cfs_mat[t,:,:] = self.__FV * np.exp(-__yc * (self.__ttm - t)) * call_mat[t,:,:]
            elif t == self.__ttm:
                nc_loan_paths[t,:,:] = np.minimum(asset_paths[t,:,:], face_values_mat[:,:])
                call_mat[t,:,:] = np.where(cond1, 1, 0)
                nc_cfs_mat[t,:,:] = nc_loan_paths[t,:,:]
                call_cfs_mat[t,:,:] = np.minimum(asset_paths[t,:,:], self.__FV) * call_mat[t,:,:]
            else:
                nc_loan_paths[t,:,:] = ott.mv_bond(asset_paths[t,:,:], self.__FV, (self.__ttm - t), self.rf, self.__sigma)
                call_mat[t,:,:] = np.where((nc_loan_paths[t,:,:] - call_barrier[t] > 0) & cond1, 1, 0)
                nc_cfs_mat[t,:,:] = self.__FV * np.exp(-self.__ync * (self.__ttm - t)) * call_mat[t,:,:]
                face_values_mat[:,:] = np.where(call_mat[t,:,:]==1,nc_loans.loc[t,'FV'],face_values_mat[:,:])
                call_cfs_mat[t,:,:] = self.__FV * np.exp(-__yc * (self.__ttm - t)) * call_mat[t,:,:]
                cfs_distributions[t,:] = np.where(call_mat[t,:,:]==1, call_cfs_mat[t,:,:] - nc_loans.loc[t,'MV'],0).sum(axis=1) * np.exp(self.rf * (self.__ttm - t))
        
        terminal_value = self.prepayment_gbm(call_mat,risk_neutral,self.__ttm)[-1,:,:]
        cash_flows = np.minimum(terminal_value, face_values_mat).sum(axis=1)
        sort = cash_flows.argsort()
        
        return cash_flows[sort], self.random[1:,:,0].sum(axis=0)[sort], cfs_distributions.sum(axis=0)[sort]