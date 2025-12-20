"""
Additive DEA Models
Based on Chapter 3.4 of Hosseinzadeh Lotfi et al. (2020)
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple
import pandas as pd


class AdditiveModel:
    """
    Additive DEA Models
    
    The additive model maximizes the sum of input and output slacks.
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        """
        Initialize Additive model
        
        Parameters:
        -----------
        inputs : np.ndarray
            Input matrix of shape (n_dmus, n_inputs)
        outputs : np.ndarray
            Output matrix of shape (n_dmus, n_outputs)
        """
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve_ccr(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Additive CCR Model (3.4.1)
        
        max sum(s_i^-) + sum(s_r^+)
        s.t. sum(lambda_j * x_ij) + s_i^- = x_ip, i=1,...,m
             sum(lambda_j * y_rj) - s_r^+ = y_rp, r=1,...,s
             lambda_j >= 0, s_i^- >= 0, s_r^+ >= 0
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation (0-based)
        
        Returns:
        --------
        sum_slacks : float
            Sum of optimal slacks
        lambdas : np.ndarray
            Optimal intensity variables
        input_slacks : np.ndarray
            Optimal input slacks (s^-)
        output_slacks : np.ndarray
            Optimal output slacks (s^+)
        """
        # Objective: maximize sum(s_i^-) + sum(s_r^+)
        # Variables: [lambda_1, ..., lambda_n, s_1^-, ..., s_m^-, s_1^+, ..., s_s^+]
        n_vars = self.n_dmus + self.n_inputs + self.n_outputs
        c = np.zeros(n_vars)
        c[self.n_dmus:self.n_dmus + self.n_inputs] = 1.0  # input slacks
        c[self.n_dmus + self.n_inputs:] = 1.0  # output slacks
        
        # Constraints: equality constraints
        n_constraints = self.n_inputs + self.n_outputs
        A_eq = np.zeros((n_constraints, n_vars))
        
        # Input constraints: sum(lambda_j * x_ij) + s_i^- = x_ip
        for i in range(self.n_inputs):
            A_eq[i, :self.n_dmus] = self.inputs[:, i]  # lambdas
            A_eq[i, self.n_dmus + i] = 1.0  # s_i^-
        
        # Output constraints: sum(lambda_j * y_rj) - s_r^+ = y_rp
        for r in range(self.n_outputs):
            A_eq[self.n_inputs + r, :self.n_dmus] = self.outputs[:, r]  # lambdas
            A_eq[self.n_inputs + r, self.n_dmus + self.n_inputs + r] = -1.0  # -s_r^+
        
        # Right-hand side
        b_eq = np.zeros(n_constraints)
        for i in range(self.n_inputs):
            b_eq[i] = self.inputs[dmu_index, i]
        for r in range(self.n_outputs):
            b_eq[self.n_inputs + r] = self.outputs[dmu_index, r]
        
        # Bounds: all variables >= 0
        bounds = [(0, None)] * n_vars
        
        # Solve (maximize, so negate objective)
        result = linprog(-c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        sum_slacks = -result.fun  # negate because we minimized negative
        lambdas = result.x[:self.n_dmus]
        input_slacks = result.x[self.n_dmus:self.n_dmus + self.n_inputs]
        output_slacks = result.x[self.n_dmus + self.n_inputs:]
        
        return sum_slacks, lambdas, input_slacks, output_slacks
    
    def solve_bcc(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Additive BCC Model (3.4.2)
        
        max sum(s_i^-) + sum(s_r^+)
        s.t. sum(lambda_j * x_ij) + s_i^- = x_ip, i=1,...,m
             sum(lambda_j * y_rj) - s_r^+ = y_rp, r=1,...,s
             sum(lambda_j) = 1
             lambda_j >= 0, s_i^- >= 0, s_r^+ >= 0
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation (0-based)
        
        Returns:
        --------
        sum_slacks : float
            Sum of optimal slacks
        lambdas : np.ndarray
            Optimal intensity variables
        input_slacks : np.ndarray
            Optimal input slacks (s^-)
        output_slacks : np.ndarray
            Optimal output slacks (s^+)
        """
        # Objective: maximize sum(s_i^-) + sum(s_r^+)
        n_vars = self.n_dmus + self.n_inputs + self.n_outputs
        c = np.zeros(n_vars)
        c[self.n_dmus:self.n_dmus + self.n_inputs] = 1.0
        c[self.n_dmus + self.n_inputs:] = 1.0
        
        # Constraints: equality constraints + convexity
        n_constraints = self.n_inputs + self.n_outputs + 1
        A_eq = np.zeros((n_constraints, n_vars))
        
        # Input constraints
        for i in range(self.n_inputs):
            A_eq[i, :self.n_dmus] = self.inputs[:, i]
            A_eq[i, self.n_dmus + i] = 1.0
        
        # Output constraints
        for r in range(self.n_outputs):
            A_eq[self.n_inputs + r, :self.n_dmus] = self.outputs[:, r]
            A_eq[self.n_inputs + r, self.n_dmus + self.n_inputs + r] = -1.0
        
        # Convexity constraint: sum(lambda_j) = 1
        A_eq[self.n_inputs + self.n_outputs, :self.n_dmus] = 1.0
        
        # Right-hand side
        b_eq = np.zeros(n_constraints)
        for i in range(self.n_inputs):
            b_eq[i] = self.inputs[dmu_index, i]
        for r in range(self.n_outputs):
            b_eq[self.n_inputs + r] = self.outputs[dmu_index, r]
        b_eq[self.n_inputs + self.n_outputs] = 1.0
        
        # Bounds
        bounds = [(0, None)] * n_vars
        
        # Solve
        result = linprog(-c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        sum_slacks = -result.fun
        lambdas = result.x[:self.n_dmus]
        input_slacks = result.x[self.n_dmus:self.n_dmus + self.n_inputs]
        output_slacks = result.x[self.n_dmus + self.n_inputs:]
        
        return sum_slacks, lambdas, input_slacks, output_slacks
    
    def evaluate_all(self, model_type: str = 'ccr') -> pd.DataFrame:
        """
        Evaluate all DMUs
        
        Parameters:
        -----------
        model_type : str
            'ccr' or 'bcc'
        
        Returns:
        --------
        results : pd.DataFrame
            DataFrame with results
        """
        results = []
        
        for j in range(self.n_dmus):
            if model_type == 'ccr':
                sum_slacks, lambdas, input_slacks, output_slacks = self.solve_ccr(j)
            else:
                sum_slacks, lambdas, input_slacks, output_slacks = self.solve_bcc(j)
            
            result_dict = {
                'DMU': j + 1,
                'Sum_Slacks': sum_slacks
            }
            for i, lam in enumerate(lambdas):
                result_dict[f'Lambda_{i+1}'] = lam
            for i in range(self.n_inputs):
                result_dict[f'S-_{i+1}'] = input_slacks[i]
            for r in range(self.n_outputs):
                result_dict[f'S+_{r+1}'] = output_slacks[r]
            
            results.append(result_dict)
        
        return pd.DataFrame(results)

