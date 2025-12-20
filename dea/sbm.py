"""
Slacks-Based Measure (SBM) DEA Models
Based on Chapter 4.9 of Hosseinzadeh Lotfi et al. (2020)
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple
import pandas as pd


class SBMModel:
    """
    Slacks-Based Measure (SBM) DEA Model
    Based on Chapter 4.9
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve_model1(self, dmu_index: int, rts: str = 'vrs') -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve First Model of SBM (4.9.1)
        
        Linearized SBM model where numerator is minimized
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation (0-based)
        rts : str
            Returns to scale: 'crs', 'vrs', 'drs', 'irs' (default: 'vrs')
        """
        # Variables: [t, lambda_1, ..., lambda_n, s_1^-, ..., s_m^-, s_1^+, ..., s_s^+]
        n_vars = 1 + self.n_dmus + self.n_inputs + self.n_outputs
        c = np.zeros(n_vars)
        c[0] = 1.0  # minimize t
        
        # Constraints: 1 (normalization) + m (inputs) + s (outputs) + 1 (RTS if VRS)
        n_eq_constraints = 1 + self.n_inputs + self.n_outputs
        if rts == 'vrs':
            n_eq_constraints += 1  # Add convexity constraint
        
        A_eq = np.zeros((n_eq_constraints, n_vars))
        A_ub = np.zeros((self.n_inputs + self.n_outputs, n_vars))
        
        row = 0
        # Constraint: t - (1/m) * sum(s_i^- / x_ip) = 1
        A_eq[row, 0] = 1.0
        for i in range(self.n_inputs):
            A_eq[row, 1 + self.n_dmus + i] = -1.0 / (self.n_inputs * self.inputs[dmu_index, i])
        row += 1
        
        # Input constraints: sum(lambda_j * x_ij) + s_i^- = t * x_ip
        for i in range(self.n_inputs):
            A_eq[row, 0] = -self.inputs[dmu_index, i]
            A_eq[row, 1:1 + self.n_dmus] = self.inputs[:, i]
            A_eq[row, 1 + self.n_dmus + i] = 1.0
            row += 1
        
        # Output constraints: sum(lambda_j * y_rj) - s_r^+ = t * y_rp
        for r in range(self.n_outputs):
            A_eq[row, 1:1 + self.n_dmus] = self.outputs[:, r]
            A_eq[row, 1 + self.n_dmus + self.n_inputs + r] = -1.0
            A_eq[row, 0] = -self.outputs[dmu_index, r]
            row += 1
        
        # RTS constraint
        if rts == 'vrs':
            # sum(lambda_j) = 1
            A_eq[row, 1:1 + self.n_dmus] = 1.0
            row += 1
        elif rts == 'drs':
            # sum(lambda_j) <= 1
            A_ub = np.vstack([A_ub, np.zeros((1, n_vars))])
            A_ub[-1, 1:1 + self.n_dmus] = 1.0
        elif rts == 'irs':
            # sum(lambda_j) >= 1
            A_ub = np.vstack([A_ub, np.zeros((1, n_vars))])
            A_ub[-1, 1:1 + self.n_dmus] = -1.0
        # crs: no constraint on sum of lambdas
        
        b_eq = np.zeros(n_eq_constraints)
        b_eq[0] = 1.0
        if rts == 'vrs':
            b_eq[-1] = 1.0
        
        # Non-negativity constraints
        for i in range(self.n_inputs):
            A_ub[i, 1 + self.n_dmus + i] = -1.0
        for r in range(self.n_outputs):
            A_ub[self.n_inputs + r, 1 + self.n_dmus + self.n_inputs + r] = -1.0
        
        b_ub = np.zeros(A_ub.shape[0])
        
        bounds = [(0, None)] * n_vars
        
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                        bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        t = result.x[0]
        lambdas = result.x[1:1 + self.n_dmus]
        input_slacks = result.x[1 + self.n_dmus:1 + self.n_dmus + self.n_inputs]
        output_slacks = result.x[1 + self.n_dmus + self.n_inputs:]
        
        # SBM efficiency = t
        sbm_eff = t
        
        return sbm_eff, lambdas, input_slacks, output_slacks
    
    def solve_model2(self, dmu_index: int, rts: str = 'vrs') -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Second Model of SBM (4.9.2)
        
        Linearized SBM model where denominator is maximized
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation (0-based)
        rts : str
            Returns to scale: 'crs', 'vrs', 'drs', 'irs' (default: 'vrs')
        """
        # Variables: [t, lambda_1, ..., lambda_n, s_1^-, ..., s_m^-, s_1^+, ..., s_s^+]
        n_vars = 1 + self.n_dmus + self.n_inputs + self.n_outputs
        c = np.zeros(n_vars)
        c[0] = -1.0  # maximize t (minimize -t)
        
        # Constraints: 1 (normalization) + m (inputs) + s (outputs) + 1 (RTS if VRS)
        n_eq_constraints = 1 + self.n_inputs + self.n_outputs
        if rts == 'vrs':
            n_eq_constraints += 1  # Add convexity constraint
        
        A_eq = np.zeros((n_eq_constraints, n_vars))
        A_ub = np.zeros((self.n_inputs + self.n_outputs, n_vars))
        
        row = 0
        # Constraint: t + (1/s) * sum(s_r^+ / y_rp) = 1
        A_eq[row, 0] = 1.0
        for r in range(self.n_outputs):
            A_eq[row, 1 + self.n_dmus + self.n_inputs + r] = 1.0 / (self.n_outputs * self.outputs[dmu_index, r])
        row += 1
        
        # Input constraints: sum(lambda_j * x_ij) + s_i^- = t * x_ip
        for i in range(self.n_inputs):
            A_eq[row, 0] = -self.inputs[dmu_index, i]
            A_eq[row, 1:1 + self.n_dmus] = self.inputs[:, i]
            A_eq[row, 1 + self.n_dmus + i] = 1.0
            row += 1
        
        # Output constraints: sum(lambda_j * y_rj) - s_r^+ = t * y_rp
        for r in range(self.n_outputs):
            A_eq[row, 1:1 + self.n_dmus] = self.outputs[:, r]
            A_eq[row, 1 + self.n_dmus + self.n_inputs + r] = -1.0
            A_eq[row, 0] = -self.outputs[dmu_index, r]
            row += 1
        
        # RTS constraint
        if rts == 'vrs':
            # sum(lambda_j) = 1
            A_eq[row, 1:1 + self.n_dmus] = 1.0
            row += 1
        elif rts == 'drs':
            # sum(lambda_j) <= 1
            A_ub = np.vstack([A_ub, np.zeros((1, n_vars))])
            A_ub[-1, 1:1 + self.n_dmus] = 1.0
        elif rts == 'irs':
            # sum(lambda_j) >= 1
            A_ub = np.vstack([A_ub, np.zeros((1, n_vars))])
            A_ub[-1, 1:1 + self.n_dmus] = -1.0
        # crs: no constraint on sum of lambdas
        
        b_eq = np.zeros(n_eq_constraints)
        b_eq[0] = 1.0
        if rts == 'vrs':
            b_eq[-1] = 1.0
        
        # Non-negativity
        for i in range(self.n_inputs):
            A_ub[i, 1 + self.n_dmus + i] = -1.0
        for r in range(self.n_outputs):
            A_ub[self.n_inputs + r, 1 + self.n_dmus + self.n_inputs + r] = -1.0
        
        b_ub = np.zeros(A_ub.shape[0])
        
        bounds = [(0, None)] * n_vars
        
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                        bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        t = result.x[0]
        lambdas = result.x[1:1 + self.n_dmus]
        input_slacks = result.x[1 + self.n_dmus:1 + self.n_dmus + self.n_inputs]
        output_slacks = result.x[1 + self.n_dmus + self.n_inputs:]
        
        # SBM efficiency = t (since we minimized -t, t = -result.fun)
        sbm_eff = t
        
        return sbm_eff, lambdas, input_slacks, output_slacks
    
    def evaluate_all(self, model_type: int = 1, rts: str = 'vrs') -> pd.DataFrame:
        """
        Evaluate all DMUs
        
        Parameters:
        -----------
        model_type : int
            1 for first model, 2 for second model
        rts : str
            Returns to scale: 'crs', 'vrs', 'drs', 'irs' (default: 'vrs')
        """
        results = []
        for j in range(self.n_dmus):
            if model_type == 1:
                eff, lambdas, input_slacks, output_slacks = self.solve_model1(j, rts=rts)
            else:
                eff, lambdas, input_slacks, output_slacks = self.solve_model2(j, rts=rts)
            
            result_dict = {'DMU': j + 1, 'SBM_Efficiency': eff}
            for i, lam in enumerate(lambdas):
                result_dict[f'Lambda_{i+1}'] = lam
            for i in range(self.n_inputs):
                result_dict[f'S-_{i+1}'] = input_slacks[i]
            for r in range(self.n_outputs):
                result_dict[f'S+_{r+1}'] = output_slacks[r]
            
            results.append(result_dict)
        
        return pd.DataFrame(results)

