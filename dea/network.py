"""
Series Network DEA Model
Based on Chapter 4.10 of Hosseinzadeh Lotfi et al. (2020)
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple
import pandas as pd


class SeriesNetworkModel:
    """
    Series Network DEA Model
    Based on Chapter 4.10
    
    This model considers intermediate products between stages.
    """
    
    def __init__(self, inputs_stage1: np.ndarray, intermediate: np.ndarray,
                 outputs_stage2: np.ndarray):
        """
        Initialize Series Network DEA model
        
        Parameters:
        -----------
        inputs_stage1 : np.ndarray
            Input matrix for stage 1, shape (n_dmus, n_inputs_stage1)
        intermediate : np.ndarray
            Intermediate products matrix, shape (n_dmus, n_intermediate)
        outputs_stage2 : np.ndarray
            Output matrix for stage 2, shape (n_dmus, n_outputs_stage2)
        """
        self.inputs_stage1 = np.array(inputs_stage1)
        self.intermediate = np.array(intermediate)
        self.outputs_stage2 = np.array(outputs_stage2)
        self.n_dmus = self.inputs_stage1.shape[0]
        self.n_inputs_s1 = self.inputs_stage1.shape[1]
        self.n_intermediate = self.intermediate.shape[1]
        self.n_outputs_s2 = self.outputs_stage2.shape[1]
        
        # Verify dimensions
        if (self.inputs_stage1.shape[0] != self.intermediate.shape[0] or
            self.intermediate.shape[0] != self.outputs_stage2.shape[0]):
            raise ValueError("Number of DMUs must be the same for all stages")
    
    def solve(self, dmu_index: int, epsilon: float = 1e-6) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Series Network DEA Model (4.10)
        
        max sum(u_r * y_rp)
        s.t. sum(u_r * y_rj) - sum(w_l * z_lj) <= 0, j=1,...,n
             sum(w_l * z_lj) - sum(v_i * x_ij) <= 0, j=1,...,n
             sum(v_i * x_ip) = 1
             u_r >= epsilon, v_i >= epsilon, w_l >= epsilon
        
        where:
        - x_ij: inputs to stage 1
        - z_lj: intermediate products
        - y_rj: outputs from stage 2
        - v_i: input weights
        - w_l: intermediate weights
        - u_r: output weights
        
        Returns:
        --------
        efficiency : float
            Efficiency score
        v_weights : np.ndarray
            Optimal input weights (v*)
        w_weights : np.ndarray
            Optimal intermediate weights (w*)
        u_weights : np.ndarray
            Optimal output weights (u*)
        """
        # Variables: [v_1, ..., v_m, w_1, ..., w_L, u_1, ..., u_s]
        n_vars = self.n_inputs_s1 + self.n_intermediate + self.n_outputs_s2
        c = np.zeros(n_vars)
        c[self.n_inputs_s1 + self.n_intermediate:] = -self.outputs_stage2[dmu_index, :]  # negative for minimization
        
        # Constraints: 2*n DMU constraints + 1 normalization + epsilon constraints
        n_constraints = 2 * self.n_dmus + 1 + n_vars
        A = np.zeros((n_constraints, n_vars))
        
        # Stage 2 constraints: sum(u_r * y_rj) - sum(w_l * z_lj) <= 0
        for j in range(self.n_dmus):
            A[j, self.n_inputs_s1 + self.n_intermediate:] = self.outputs_stage2[j, :]  # u_r * y_rj
            A[j, self.n_inputs_s1:self.n_inputs_s1 + self.n_intermediate] = -self.intermediate[j, :]  # -w_l * z_lj
        
        # Stage 1 constraints: sum(w_l * z_lj) - sum(v_i * x_ij) <= 0
        for j in range(self.n_dmus):
            A[self.n_dmus + j, self.n_inputs_s1:self.n_inputs_s1 + self.n_intermediate] = self.intermediate[j, :]  # w_l * z_lj
            A[self.n_dmus + j, :self.n_inputs_s1] = -self.inputs_stage1[j, :]  # -v_i * x_ij
        
        # Normalization constraint: sum(v_i * x_ip) = 1
        A[2 * self.n_dmus, :self.n_inputs_s1] = self.inputs_stage1[dmu_index, :]
        
        # Epsilon constraints
        constraint_idx = 2 * self.n_dmus + 1
        for i in range(self.n_inputs_s1):
            A[constraint_idx, i] = -1.0
            constraint_idx += 1
        for l in range(self.n_intermediate):
            A[constraint_idx, self.n_inputs_s1 + l] = -1.0
            constraint_idx += 1
        for r in range(self.n_outputs_s2):
            A[constraint_idx, self.n_inputs_s1 + self.n_intermediate + r] = -1.0
            constraint_idx += 1
        
        # Right-hand side
        b = np.zeros(n_constraints)
        b[2 * self.n_dmus] = 1.0  # normalization
        for i in range(n_vars):
            b[2 * self.n_dmus + 1 + i] = -epsilon
        
        # Constraint types
        A_eq = A[2 * self.n_dmus:2 * self.n_dmus + 1, :]
        b_eq = b[2 * self.n_dmus:2 * self.n_dmus + 1]
        A_ub = np.vstack([A[:2 * self.n_dmus, :], A[2 * self.n_dmus + 1:, :]])
        b_ub = np.hstack([b[:2 * self.n_dmus], b[2 * self.n_dmus + 1:]])
        
        # Bounds
        bounds = [(0, None)] * n_vars
        
        # Solve
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        efficiency = -result.fun
        v_weights = result.x[:self.n_inputs_s1]
        w_weights = result.x[self.n_inputs_s1:self.n_inputs_s1 + self.n_intermediate]
        u_weights = result.x[self.n_inputs_s1 + self.n_intermediate:]
        
        return efficiency, v_weights, w_weights, u_weights
    
    def evaluate_all(self, epsilon: float = 1e-6) -> pd.DataFrame:
        """
        Evaluate all DMUs
        
        Returns:
        --------
        results : pd.DataFrame
            DataFrame with results
        """
        results = []
        
        for j in range(self.n_dmus):
            eff, v, w, u = self.solve(j, epsilon)
            
            result_dict = {'DMU': j + 1, 'Efficiency': eff}
            for i in range(self.n_inputs_s1):
                result_dict[f'v_{i+1}'] = v[i]
            for l in range(self.n_intermediate):
                result_dict[f'w_{l+1}'] = w[l]
            for r in range(self.n_outputs_s2):
                result_dict[f'u_{r+1}'] = u[r]
            
            results.append(result_dict)
        
        return pd.DataFrame(results)

