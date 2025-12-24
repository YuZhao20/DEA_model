"""
Advanced DEA Models from Chapter 4
Including: Directional Efficiency
Based on Chapter 4 of Hosseinzadeh Lotfi et al. (2020)
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple
import pandas as pd


class DirectionalEfficiencyModel:
    """
    Directional Efficiency DEA Model
    Based on Chapter 4.15
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve(self, dmu_index: int, gx: np.ndarray = None, gy: np.ndarray = None, rts: str = 'vrs') -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Directional Efficiency Model (4.15)
        
        max beta
        s.t. beta*gx_i + sum(lambda_j * x_ij) <= x_ip, i=1,...,m
             -beta*gy_r + sum(lambda_j * y_rj) >= y_rp, r=1,...,s
             lambda_j >= 0
             sum(lambda_j) = 1 (if VRS)
        
        Default direction: gx = -x_p, gy = y_p
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation
        gx : np.ndarray, optional
            Input direction vector
        gy : np.ndarray, optional
            Output direction vector
        rts : str
            Returns to scale: 'crs', 'vrs', 'drs', 'irs' (default: 'vrs')
        
        Returns:
        --------
        efficiency : float
            Directional efficiency score
        lambdas : np.ndarray
            Optimal intensity variables
        input_slacks : np.ndarray
            Input slacks
        output_slacks : np.ndarray
            Output slacks
        """
        if gx is None:
            gx = -self.inputs[dmu_index, :]
        if gy is None:
            gy = self.outputs[dmu_index, :]
        
        # Variables: [beta, lambda_1, ..., lambda_n, s_1^-, ..., s_m^-, s_1^+, ..., s_s^+]
        n_vars = 1 + self.n_dmus + self.n_inputs + self.n_outputs
        c = np.zeros(n_vars)
        c[0] = -1.0  # maximize beta (minimize -beta)
        
        # Constraints: inputs + outputs + RTS (if VRS)
        n_constraints = self.n_inputs + self.n_outputs
        if rts == 'vrs':
            n_constraints += 1
        
        A = np.zeros((n_constraints, n_vars))
        
        # Input constraints: beta*gx_i + sum(lambda_j * x_ij) + s_i^- <= x_ip
        for i in range(self.n_inputs):
            A[i, 0] = gx[i]  # beta * gx_i
            A[i, 1:1 + self.n_dmus] = self.inputs[:, i]  # sum(lambda_j * x_ij)
            A[i, 1 + self.n_dmus + i] = 1.0  # s_i^-
        
        # Output constraints: -beta*gy_r + sum(lambda_j * y_rj) - s_r^+ >= y_rp
        # For linprog: beta*gy_r - sum(lambda_j * y_rj) + s_r^+ <= -y_rp
        for r in range(self.n_outputs):
            A[self.n_inputs + r, 0] = -gy[r]  # -beta * gy_r (negative for >= constraint)
            A[self.n_inputs + r, 1:1 + self.n_dmus] = -self.outputs[:, r]  # -sum(lambda_j * y_rj)
            A[self.n_inputs + r, 1 + self.n_dmus + self.n_inputs + r] = 1.0  # s_r^+
        
        # RTS constraint
        if rts == 'vrs':
            A[self.n_inputs + self.n_outputs, 1:1 + self.n_dmus] = 1.0  # sum(lambda_j) = 1
        elif rts == 'drs':
            A[self.n_inputs + self.n_outputs, 1:1 + self.n_dmus] = 1.0  # sum(lambda_j) <= 1
        elif rts == 'irs':
            A[self.n_inputs + self.n_outputs, 1:1 + self.n_dmus] = -1.0  # sum(lambda_j) >= 1
        
        # Right-hand side
        b = np.zeros(n_constraints)
        for i in range(self.n_inputs):
            b[i] = self.inputs[dmu_index, i]
        for r in range(self.n_outputs):
            b[self.n_inputs + r] = -self.outputs[dmu_index, r]
        
        if rts == 'vrs':
            b[self.n_inputs + self.n_outputs] = 1.0
        elif rts == 'drs':
            b[self.n_inputs + self.n_outputs] = 1.0
        elif rts == 'irs':
            b[self.n_inputs + self.n_outputs] = -1.0
        
        # Constraint types
        n_eq = 0
        if rts == 'vrs':
            n_eq = 1
        
        A_eq = A[self.n_inputs + self.n_outputs:self.n_inputs + self.n_outputs + n_eq, :] if n_eq > 0 else None
        b_eq = b[self.n_inputs + self.n_outputs:self.n_inputs + self.n_outputs + n_eq] if n_eq > 0 else None
        A_ub = A[:self.n_inputs + self.n_outputs + (0 if rts == 'vrs' else 1), :]
        b_ub = b[:self.n_inputs + self.n_outputs + (0 if rts == 'vrs' else 1)]
        
        if rts != 'vrs':
            if rts == 'drs':
                A_ub = np.vstack([A_ub, A[self.n_inputs + self.n_outputs:self.n_inputs + self.n_outputs + 1, :]])
                b_ub = np.append(b_ub, b[self.n_inputs + self.n_outputs:self.n_inputs + self.n_outputs + 1])
            elif rts == 'irs':
                A_ub = np.vstack([A_ub, -A[self.n_inputs + self.n_outputs:self.n_inputs + self.n_outputs + 1, :]])
                b_ub = np.append(b_ub, -b[self.n_inputs + self.n_outputs:self.n_inputs + self.n_outputs + 1])
        
        # Bounds
        bounds = [(0, None)] * n_vars
        bounds[0] = (None, None)  # beta can be negative
        
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                         bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        efficiency = -result.fun
        lambdas = result.x[1:1 + self.n_dmus]
        input_slacks = result.x[1 + self.n_dmus:1 + self.n_dmus + self.n_inputs]
        output_slacks = result.x[1 + self.n_dmus + self.n_inputs:]
        
        return efficiency, lambdas, input_slacks, output_slacks
    
    def evaluate_all(self, gx: np.ndarray = None, gy: np.ndarray = None, rts: str = 'vrs') -> pd.DataFrame:
        results = []
        for j in range(self.n_dmus):
            # solve returns 4 values: eff, lambdas, input_slacks, output_slacks
            eff, lambdas, input_slacks, output_slacks = self.solve(j, gx, gy, rts)
            result_dict = {'DMU': j + 1, 'Efficiency': eff}
            for i, lam in enumerate(lambdas):
                result_dict[f'Lambda_{i+1}'] = lam
            for i in range(self.n_inputs):
                result_dict[f'Input_Slack_{i+1}'] = input_slacks[i]
            for r in range(self.n_outputs):
                result_dict[f'Output_Slack_{r+1}'] = output_slacks[r]
            results.append(result_dict)
        return pd.DataFrame(results)
