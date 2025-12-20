"""
Malmquist Productivity Index Models
Based on Chapter 4.8 of Hosseinzadeh Lotfi et al. (2020)
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple
import pandas as pd


class MalmquistModel:
    """
    Malmquist Productivity Index - CCR Model
    Based on Chapter 4.8
    """
    
    def __init__(self, inputs_t: np.ndarray, outputs_t: np.ndarray,
                 inputs_t1: np.ndarray, outputs_t1: np.ndarray):
        """
        Initialize Malmquist model with data from two time periods
        
        Parameters:
        -----------
        inputs_t : np.ndarray
            Input matrix at time t
        outputs_t : np.ndarray
            Output matrix at time t
        inputs_t1 : np.ndarray
            Input matrix at time t+1
        outputs_t1 : np.ndarray
            Output matrix at time t+1
        """
        self.inputs_t = np.array(inputs_t)
        self.outputs_t = np.array(outputs_t)
        self.inputs_t1 = np.array(inputs_t1)
        self.outputs_t1 = np.array(outputs_t1)
        self.n_dmus, self.n_inputs = self.inputs_t.shape
        self.n_outputs = self.outputs_t.shape[1]
    
    def solve_multiplier(self, dmu_index: int, time_period: str = 't') -> float:
        """
        Solve Malmquist Productivity Index - CCR Multiplier Model (4.8.1)
        
        Calculates efficiency scores for different time period combinations
        """
        from .ccr import CCRModel
        
        if time_period == 't':
            inputs_ref = self.inputs_t
            outputs_ref = self.outputs_t
            inputs_dmu = self.inputs_t[dmu_index, :]
            outputs_dmu = self.outputs_t[dmu_index, :]
        else:  # t+1
            inputs_ref = self.inputs_t1
            outputs_ref = self.outputs_t1
            inputs_dmu = self.inputs_t1[dmu_index, :]
            outputs_dmu = self.outputs_t1[dmu_index, :]
        
        # Solve CCR multiplier model
        ccr = CCRModel(inputs_ref, outputs_ref)
        eff, _, _ = ccr.solve_multiplier(dmu_index)
        
        return eff
    
    def calculate_malmquist_index(self, dmu_index: int) -> Tuple[float, float, float, float, float]:
        """
        Calculate Malmquist Productivity Index
        
        Returns:
        --------
        d_t_t : float
            Efficiency at time t relative to frontier at time t
        d_t1_t1 : float
            Efficiency at time t+1 relative to frontier at time t+1
        d_t_t1 : float
            Efficiency at time t relative to frontier at time t+1
        d_t1_t : float
            Efficiency at time t+1 relative to frontier at time t
        mi : float
            Malmquist Index
        """
        # Calculate four efficiency scores
        d_t_t = self._solve_efficiency(self.inputs_t, self.outputs_t,
                                       self.inputs_t[dmu_index, :],
                                       self.outputs_t[dmu_index, :],
                                       self.inputs_t, self.outputs_t)
        
        d_t1_t1 = self._solve_efficiency(self.inputs_t1, self.outputs_t1,
                                         self.inputs_t1[dmu_index, :],
                                         self.outputs_t1[dmu_index, :],
                                         self.inputs_t1, self.outputs_t1)
        
        d_t_t1 = self._solve_efficiency(self.inputs_t1, self.outputs_t1,
                                        self.inputs_t[dmu_index, :],
                                        self.outputs_t[dmu_index, :],
                                        self.inputs_t1, self.outputs_t1)
        
        d_t1_t = self._solve_efficiency(self.inputs_t, self.outputs_t,
                                        self.inputs_t1[dmu_index, :],
                                        self.outputs_t1[dmu_index, :],
                                        self.inputs_t, self.outputs_t)
        
        # Malmquist Index
        mi = np.sqrt((d_t_t1 / d_t_t) * (d_t1_t1 / d_t1_t))
        
        return d_t_t, d_t1_t1, d_t_t1, d_t1_t, mi
    
    def _solve_efficiency(self, inputs_ref, outputs_ref, inputs_dmu, outputs_dmu,
                         inputs_frontier, outputs_frontier) -> float:
        """Solve efficiency score for given reference set and DMU"""
        # Input-oriented CCR model
        n_dmus = inputs_ref.shape[0]
        c = np.zeros(n_dmus + 1)
        c[0] = 1.0
        
        n_constraints = len(inputs_dmu) + len(outputs_dmu)
        A = np.zeros((n_constraints, n_dmus + 1))
        
        for i in range(len(inputs_dmu)):
            A[i, 0] = -inputs_dmu[i]
            A[i, 1:] = inputs_frontier[:, i]
        
        for r in range(len(outputs_dmu)):
            A[len(inputs_dmu) + r, 1:] = -outputs_frontier[:, r]
        
        b = np.zeros(n_constraints)
        for r in range(len(outputs_dmu)):
            b[len(inputs_dmu) + r] = -outputs_dmu[r]
        
        bounds = [(0, None)] * (n_dmus + 1)
        
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        if not result.success:
            return 0.0
        
        return result.x[0]
    
    def evaluate_all(self) -> pd.DataFrame:
        results = []
        for j in range(self.n_dmus):
            d_t_t, d_t1_t1, d_t_t1, d_t1_t, mi = self.calculate_malmquist_index(j)
            results.append({
                'DMU': j + 1,
                'D_t_t': d_t_t,
                'D_t1_t1': d_t1_t1,
                'D_t_t1': d_t_t1,
                'D_t1_t': d_t1_t,
                'Malmquist_Index': mi
            })
        return pd.DataFrame(results)

