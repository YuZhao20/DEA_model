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
        
        Linearized SBM model using Charnes-Cooper transformation
        Variables: [t, Lambda_1, ..., Lambda_n, S_1^-, ..., S_m^-, S_1^+, ..., S_s^+]
        where Lambda = t * lambda, S^- = t * s^-, S^+ = t * s^+
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation (0-based)
        rts : str
            Returns to scale: 'crs', 'vrs', 'drs', 'irs' (default: 'vrs')
        """
        # Variables: [t, Lambda_1, ..., Lambda_n, S_1^-, ..., S_m^-, S_1^+, ..., S_s^+]
        n_vars = 1 + self.n_dmus + self.n_inputs + self.n_outputs
        c = np.zeros(n_vars)
        c[0] = 1.0  # minimize t (which represents the numerator)
        
        # Constraints: 1 (normalization) + m (inputs) + s (outputs) + 1 (RTS if VRS)
        n_eq_constraints = 1 + self.n_inputs + self.n_outputs
        if rts == 'vrs':
            n_eq_constraints += 1  # Add convexity constraint
        
        A_eq = np.zeros((n_eq_constraints, n_vars))
        A_ub = np.zeros((self.n_inputs + self.n_outputs, n_vars))
        
        row = 0
        # Normalization constraint: t + (1/s) * sum(S_r^+ / y_rp) = 1
        # This ensures the denominator is normalized to 1
        A_eq[row, 0] = 1.0
        for r in range(self.n_outputs):
            if self.outputs[dmu_index, r] > 0:
                A_eq[row, 1 + self.n_dmus + self.n_inputs + r] = 1.0 / (self.n_outputs * self.outputs[dmu_index, r])
        row += 1
        
        # Input constraints: sum(Lambda_j * x_ij) + S_i^- = t * x_ip
        for i in range(self.n_inputs):
            A_eq[row, 0] = -self.inputs[dmu_index, i]
            A_eq[row, 1:1 + self.n_dmus] = self.inputs[:, i]
            A_eq[row, 1 + self.n_dmus + i] = 1.0
            row += 1
        
        # Output constraints: sum(Lambda_j * y_rj) - S_r^+ = t * y_rp
        for r in range(self.n_outputs):
            A_eq[row, 1:1 + self.n_dmus] = self.outputs[:, r]
            A_eq[row, 1 + self.n_dmus + self.n_inputs + r] = -1.0
            A_eq[row, 0] = -self.outputs[dmu_index, r]
            row += 1
        
        # RTS constraint
        if rts == 'vrs':
            # sum(Lambda_j) = t
            A_eq[row, 1:1 + self.n_dmus] = 1.0
            A_eq[row, 0] = -1.0
            row += 1
        elif rts == 'drs':
            # sum(Lambda_j) <= t
            A_ub = np.vstack([A_ub, np.zeros((1, n_vars))])
            A_ub[-1, 1:1 + self.n_dmus] = 1.0
            A_ub[-1, 0] = -1.0
        elif rts == 'irs':
            # sum(Lambda_j) >= t
            A_ub = np.vstack([A_ub, np.zeros((1, n_vars))])
            A_ub[-1, 1:1 + self.n_dmus] = -1.0
            A_ub[-1, 0] = 1.0
        # crs: no constraint on sum of lambdas
        
        b_eq = np.zeros(n_eq_constraints)
        b_eq[0] = 1.0  # normalization constraint
        if rts == 'vrs':
            b_eq[-1] = 0.0  # sum(Lambda_j) = t
        
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
        Lambda = result.x[1:1 + self.n_dmus]
        S_minus = result.x[1 + self.n_dmus:1 + self.n_dmus + self.n_inputs]
        S_plus = result.x[1 + self.n_dmus + self.n_inputs:]
        
        # Convert back to original variables
        if t > 1e-10:
            lambdas = Lambda / t
            input_slacks = S_minus / t
            output_slacks = S_plus / t
        else:
            lambdas = Lambda
            input_slacks = S_minus
            output_slacks = S_plus
        
        # SBM efficiency = t (since denominator is normalized to 1)
        sbm_eff = t
        
        return sbm_eff, lambdas, input_slacks, output_slacks
    
    def solve_model2(self, dmu_index: int, rts: str = 'vrs') -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Second Model of SBM (4.9.2)
        
        Linearized SBM model using Charnes-Cooper transformation
        Variables: [t, Lambda_1, ..., Lambda_n, S_1^-, ..., S_m^-, S_1^+, ..., S_s^+]
        where Lambda = t * lambda, S^- = t * s^-, S^+ = t * s^+
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation (0-based)
        rts : str
            Returns to scale: 'crs', 'vrs', 'drs', 'irs' (default: 'vrs')
        """
        # Variables: [t, Lambda_1, ..., Lambda_n, S_1^-, ..., S_m^-, S_1^+, ..., S_s^+]
        n_vars = 1 + self.n_dmus + self.n_inputs + self.n_outputs
        c = np.zeros(n_vars)
        c[0] = -1.0  # maximize t (minimize -t), which represents the denominator
        
        # Constraints: 1 (normalization) + m (inputs) + s (outputs) + 1 (RTS if VRS)
        n_eq_constraints = 1 + self.n_inputs + self.n_outputs
        if rts == 'vrs':
            n_eq_constraints += 1  # Add convexity constraint
        
        A_eq = np.zeros((n_eq_constraints, n_vars))
        A_ub = np.zeros((self.n_inputs + self.n_outputs, n_vars))
        
        row = 0
        # Normalization constraint: t - (1/m) * sum(S_i^- / x_ip) = 1
        # This ensures the numerator is normalized to 1
        A_eq[row, 0] = 1.0
        for i in range(self.n_inputs):
            if self.inputs[dmu_index, i] > 0:
                A_eq[row, 1 + self.n_dmus + i] = -1.0 / (self.n_inputs * self.inputs[dmu_index, i])
        row += 1
        
        # Input constraints: sum(Lambda_j * x_ij) + S_i^- = t * x_ip
        for i in range(self.n_inputs):
            A_eq[row, 0] = -self.inputs[dmu_index, i]
            A_eq[row, 1:1 + self.n_dmus] = self.inputs[:, i]
            A_eq[row, 1 + self.n_dmus + i] = 1.0
            row += 1
        
        # Output constraints: sum(Lambda_j * y_rj) - S_r^+ = t * y_rp
        for r in range(self.n_outputs):
            A_eq[row, 1:1 + self.n_dmus] = self.outputs[:, r]
            A_eq[row, 1 + self.n_dmus + self.n_inputs + r] = -1.0
            A_eq[row, 0] = -self.outputs[dmu_index, r]
            row += 1
        
        # RTS constraint
        if rts == 'vrs':
            # sum(Lambda_j) = t
            A_eq[row, 1:1 + self.n_dmus] = 1.0
            A_eq[row, 0] = -1.0
            row += 1
        elif rts == 'drs':
            # sum(Lambda_j) <= t
            A_ub = np.vstack([A_ub, np.zeros((1, n_vars))])
            A_ub[-1, 1:1 + self.n_dmus] = 1.0
            A_ub[-1, 0] = -1.0
        elif rts == 'irs':
            # sum(Lambda_j) >= t
            A_ub = np.vstack([A_ub, np.zeros((1, n_vars))])
            A_ub[-1, 1:1 + self.n_dmus] = -1.0
            A_ub[-1, 0] = 1.0
        # crs: no constraint on sum of lambdas
        
        b_eq = np.zeros(n_eq_constraints)
        b_eq[0] = 1.0  # normalization constraint
        if rts == 'vrs':
            b_eq[-1] = 0.0  # sum(Lambda_j) = t
        
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
        Lambda = result.x[1:1 + self.n_dmus]
        S_minus = result.x[1 + self.n_dmus:1 + self.n_dmus + self.n_inputs]
        S_plus = result.x[1 + self.n_dmus + self.n_inputs:]
        
        # Convert back to original variables
        if t > 1e-10:
            lambdas = Lambda / t
            input_slacks = S_minus / t
            output_slacks = S_plus / t
        else:
            lambdas = Lambda
            input_slacks = S_minus
            output_slacks = S_plus
        
        # SBM efficiency = 1 / t (since numerator is normalized to 1)
        if t > 1e-10:
            sbm_eff = 1.0 / t
        else:
            sbm_eff = 0.0
        
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

