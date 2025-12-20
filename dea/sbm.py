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
    
    def solve_model1(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve First Model of SBM (4.9.1)
        
        Linearized SBM model where numerator is minimized
        """
        # Variables: [t, lambda_1, ..., lambda_n, s_1^-, ..., s_m^-, s_1^+, ..., s_s^+]
        n_vars = 1 + self.n_dmus + self.n_inputs + self.n_outputs
        c = np.zeros(n_vars)
        c[0] = 1.0  # minimize t
        
        # Constraints
        n_constraints = 1 + self.n_inputs + self.n_outputs + self.n_inputs + self.n_outputs
        A_eq = np.zeros((1 + self.n_inputs + self.n_outputs, n_vars))
        A_ub = np.zeros((self.n_inputs + self.n_outputs, n_vars))
        
        # Constraint: t - (1/m) * sum(s_i^- / x_ip) = 1
        A_eq[0, 0] = 1.0
        for i in range(self.n_inputs):
            A_eq[0, 1 + self.n_dmus + i] = -1.0 / (self.n_inputs * self.inputs[dmu_index, i])
        
        # Input constraints: sum(lambda_j * x_ij) + s_i^- = t * x_ip
        for i in range(self.n_inputs):
            A_eq[1 + i, 0] = -self.inputs[dmu_index, i]
            A_eq[1 + i, 1:1 + self.n_dmus] = self.inputs[:, i]
            A_eq[1 + i, 1 + self.n_dmus + i] = 1.0
        
        # Output constraints: sum(lambda_j * y_rj) - s_r^+ = t * y_rp
        for r in range(self.n_outputs):
            A_eq[1 + self.n_inputs + r, 1:1 + self.n_dmus] = self.outputs[:, r]
            A_eq[1 + self.n_inputs + r, 1 + self.n_dmus + self.n_inputs + r] = -1.0
            A_eq[1 + self.n_inputs + r, 0] = -self.outputs[dmu_index, r]
        
        b_eq = np.zeros(1 + self.n_inputs + self.n_outputs)
        b_eq[0] = 1.0
        
        # Non-negativity constraints
        for i in range(self.n_inputs):
            A_ub[i, 1 + self.n_dmus + i] = -1.0
        for r in range(self.n_outputs):
            A_ub[self.n_inputs + r, 1 + self.n_dmus + self.n_inputs + r] = -1.0
        
        b_ub = np.zeros(self.n_inputs + self.n_outputs)
        
        bounds = [(0, None)] * n_vars
        
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                        bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        t = result.x[0]
        lambdas = result.x[1:1 + self.n_dmus]
        input_slacks = result.x[1 + self.n_dmus:1 + self.n_dmus + self.n_inputs]
        output_slacks = result.x[1 + self.n_dmus + self.n_inputs:]
        
        # SBM efficiency
        sbm_eff = t
        
        return sbm_eff, lambdas, input_slacks, output_slacks
    
    def solve_model2(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Second Model of SBM (4.9.2)
        
        Linearized SBM model where denominator is maximized
        """
        # Similar structure but with different objective
        # Variables: [t, lambda_1, ..., lambda_n, s_1^-, ..., s_m^-, s_1^+, ..., s_s^+]
        n_vars = 1 + self.n_dmus + self.n_inputs + self.n_outputs
        c = np.zeros(n_vars)
        c[0] = -1.0  # maximize t (minimize -t)
        
        # Constraints similar to model 1
        n_constraints = 1 + self.n_inputs + self.n_outputs + self.n_inputs + self.n_outputs
        A_eq = np.zeros((1 + self.n_inputs + self.n_outputs, n_vars))
        A_ub = np.zeros((self.n_inputs + self.n_outputs, n_vars))
        
        # Constraint: t + (1/s) * sum(s_r^+ / y_rp) = 1
        A_eq[0, 0] = 1.0
        for r in range(self.n_outputs):
            A_eq[0, 1 + self.n_dmus + self.n_inputs + r] = 1.0 / (self.n_outputs * self.outputs[dmu_index, r])
        
        # Input constraints: sum(lambda_j * x_ij) + s_i^- = t * x_ip
        for i in range(self.n_inputs):
            A_eq[1 + i, 0] = -self.inputs[dmu_index, i]
            A_eq[1 + i, 1:1 + self.n_dmus] = self.inputs[:, i]
            A_eq[1 + i, 1 + self.n_dmus + i] = 1.0
        
        # Output constraints: sum(lambda_j * y_rj) - s_r^+ = t * y_rp
        for r in range(self.n_outputs):
            A_eq[1 + self.n_inputs + r, 1:1 + self.n_dmus] = self.outputs[:, r]
            A_eq[1 + self.n_inputs + r, 1 + self.n_dmus + self.n_inputs + r] = -1.0
            A_eq[1 + self.n_inputs + r, 0] = -self.outputs[dmu_index, r]
        
        b_eq = np.zeros(1 + self.n_inputs + self.n_outputs)
        b_eq[0] = 1.0
        
        # Non-negativity
        for i in range(self.n_inputs):
            A_ub[i, 1 + self.n_dmus + i] = -1.0
        for r in range(self.n_outputs):
            A_ub[self.n_inputs + r, 1 + self.n_dmus + self.n_inputs + r] = -1.0
        
        b_ub = np.zeros(self.n_inputs + self.n_outputs)
        
        bounds = [(0, None)] * n_vars
        
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                        bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        t = result.x[0]
        lambdas = result.x[1:1 + self.n_dmus]
        input_slacks = result.x[1 + self.n_dmus:1 + self.n_dmus + self.n_inputs]
        output_slacks = result.x[1 + self.n_dmus + self.n_inputs:]
        
        sbm_eff = -result.fun  # negate because we minimized negative
        
        return sbm_eff, lambdas, input_slacks, output_slacks
    
    def evaluate_all(self, model_type: int = 1) -> pd.DataFrame:
        """
        Evaluate all DMUs
        
        Parameters:
        -----------
        model_type : int
            1 for first model, 2 for second model
        """
        results = []
        for j in range(self.n_dmus):
            if model_type == 1:
                eff, lambdas, input_slacks, output_slacks = self.solve_model1(j)
            else:
                eff, lambdas, input_slacks, output_slacks = self.solve_model2(j)
            
            result_dict = {'DMU': j + 1, 'SBM_Efficiency': eff}
            for i, lam in enumerate(lambdas):
                result_dict[f'Lambda_{i+1}'] = lam
            for i in range(self.n_inputs):
                result_dict[f'S-_{i+1}'] = input_slacks[i]
            for r in range(self.n_outputs):
                result_dict[f'S+_{r+1}'] = output_slacks[r]
            
            results.append(result_dict)
        
        return pd.DataFrame(results)

