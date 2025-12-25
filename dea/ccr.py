"""
CCR (Charnes-Cooper-Rhodes) DEA Models
Based on Chapter 3.2 of Hosseinzadeh Lotfi et al. (2020)
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Optional
import pandas as pd


class CCRModel:
    """
    Input-Oriented CCR DEA Model
    
    The CCR model assumes constant returns to scale (CRS).
    """

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        """
        Initialize CCR model
        
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

    def solve_envelopment(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Input-Oriented CCR Envelopment Model (3.1)
        
        min h
        s.t. -h*x_ip + sum(lambda_j * x_ij) >= 0, i=1,...,m
             sum(lambda_j * y_rj) >= y_rp, r=1,...,s
             lambda_j >= 0, j=1,...,n
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation (0-based)
        
        Returns:
        --------
        efficiency : float
            Efficiency score (h*)
        lambdas : np.ndarray
            Optimal intensity variables (lambda*)
        input_targets : np.ndarray
            Target input values
        output_targets : np.ndarray
            Target output values
        """
        c = np.zeros(self.n_dmus + 1)
        c[0] = 1.0

        n_constraints = self.n_inputs + self.n_outputs
        A = np.zeros((n_constraints, self.n_dmus + 1))

        for i in range(self.n_inputs):
            A[i, 0] = -self.inputs[dmu_index, i]
            A[i, 1:] = self.inputs[:, i]

        for r in range(self.n_outputs):
            A[self.n_inputs + r, 0] = 0.0
            A[self.n_inputs + r, 1:] = -self.outputs[:, r]

        b = np.zeros(n_constraints)
        for r in range(self.n_outputs):
            b[self.n_inputs + r] = -self.outputs[dmu_index, r]

        bounds = [(0, None)] * (self.n_dmus + 1)

        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")

        efficiency = result.x[0]
        lambdas = result.x[1:]

        input_targets = lambdas @ self.inputs
        output_targets = lambdas @ self.outputs

        return efficiency, lambdas, input_targets, output_targets

    def solve_multiplier(self, dmu_index: int, epsilon: float = 1e-6) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Input-Oriented CCR Multiplier Model (3.3)
        
        max sum(u_r * y_rp)
        s.t. -sum(v_i * x_ij) + sum(u_r * y_rj) <= 0, j=1,...,n
             sum(v_i * x_ip) = 1
             u_r >= epsilon, v_i >= epsilon
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation (0-based)
        epsilon : float
            Small positive value for non-Archimedean constraint (default: 1e-6)
        
        Returns:
        --------
        efficiency : float
            Efficiency score
        v_weights : np.ndarray
            Optimal input weights (v*)
        u_weights : np.ndarray
            Optimal output weights (u*)
        lambdas : np.ndarray
            Dual variables (intensity variables from dual problem)
        """
        c = np.zeros(self.n_inputs + self.n_outputs)
        c[self.n_inputs:] = -self.outputs[dmu_index, :]

        n_constraints = self.n_dmus + 1 + self.n_inputs + self.n_outputs
        A = np.zeros((n_constraints, self.n_inputs + self.n_outputs))

        for j in range(self.n_dmus):
            A[j, :self.n_inputs] = -self.inputs[j, :]
            A[j, self.n_inputs:] = self.outputs[j, :]

        A[self.n_dmus, :self.n_inputs] = self.inputs[dmu_index, :]

        for i in range(self.n_inputs):
            A[self.n_dmus + 1 + i, i] = -1.0
        for r in range(self.n_outputs):
            A[self.n_dmus + 1 + self.n_inputs + r, self.n_inputs + r] = -1.0

        b = np.zeros(n_constraints)
        b[self.n_dmus] = 1.0
        for i in range(self.n_inputs):
            b[self.n_dmus + 1 + i] = -epsilon
        for r in range(self.n_outputs):
            b[self.n_dmus + 1 + self.n_inputs + r] = -epsilon

        A_eq = A[self.n_dmus:self.n_dmus+1, :]
        b_eq = b[self.n_dmus:self.n_dmus+1]
        A_ub = np.vstack([A[:self.n_dmus, :], A[self.n_dmus+1:, :]])
        b_ub = np.hstack([b[:self.n_dmus], b[self.n_dmus+1:]])

        bounds = [(0, None)] * (self.n_inputs + self.n_outputs)

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')

        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")

        efficiency = -result.fun
        v_weights = result.x[:self.n_inputs]
        u_weights = result.x[self.n_inputs:]

        lambdas = None

        return efficiency, v_weights, u_weights, lambdas

    def evaluate_all(self, method: str = 'envelopment') -> pd.DataFrame:
        """
        Evaluate all DMUs
        
        Parameters:
        -----------
        method : str
            'envelopment' or 'multiplier'
        
        Returns:
        --------
        results : pd.DataFrame
            DataFrame with efficiency scores and other results
        """
        results = []

        for j in range(self.n_dmus):
            if method == 'envelopment':
                eff, lambdas, input_targets, output_targets = self.solve_envelopment(j)
                result_dict = {
                    'DMU': j + 1,
                    'Efficiency': eff
                }
                for i, lam in enumerate(lambdas):
                    result_dict[f'Lambda_{i+1}'] = lam
                for i in range(self.n_inputs):
                    result_dict[f'Input_Target_{i+1}'] = input_targets[i]
                for r in range(self.n_outputs):
                    result_dict[f'Output_Target_{r+1}'] = output_targets[r]
            else:
                eff, v, u, lambdas = self.solve_multiplier(j)
                result_dict = {
                    'DMU': j + 1,
                    'Efficiency': eff
                }
                for i in range(self.n_inputs):
                    result_dict[f'v_{i+1}'] = v[i]
                for r in range(self.n_outputs):
                    result_dict[f'u_{r+1}'] = u[r]
                if lambdas is not None:
                    for i, lam in enumerate(lambdas):
                        result_dict[f'Lambda_{i+1}'] = lam

            results.append(result_dict)

        return pd.DataFrame(results)

    def solve_output_oriented_envelopment(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Output-Oriented CCR Envelopment Model (3.3.1)
        
        max u
        s.t. sum(lambda_j * x_ij) <= x_ip, i=1,...,m
             -u*y_rp + sum(lambda_j * y_rj) >= 0, r=1,...,s
             lambda_j >= 0, j=1,...,n
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation (0-based)
        
        Returns:
        --------
        efficiency : float
            Efficiency score (u*)
        lambdas : np.ndarray
            Optimal intensity variables (lambda*)
        input_targets : np.ndarray
            Target input values
        output_targets : np.ndarray
            Target output values
        """
        c = np.zeros(self.n_dmus + 1)
        c[0] = -1.0

        n_constraints = self.n_inputs + self.n_outputs
        A = np.zeros((n_constraints, self.n_dmus + 1))

        for i in range(self.n_inputs):
            A[i, 0] = 0.0
            A[i, 1:] = self.inputs[:, i]

        for r in range(self.n_outputs):
            A[self.n_inputs + r, 0] = self.outputs[dmu_index, r]
            A[self.n_inputs + r, 1:] = -self.outputs[:, r]

        b = np.zeros(n_constraints)
        for i in range(self.n_inputs):
            b[i] = self.inputs[dmu_index, i]

        bounds = [(0, None)] * (self.n_dmus + 1)

        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")

        efficiency = -result.fun
        lambdas = result.x[1:]

        input_targets = lambdas @ self.inputs
        output_targets = lambdas @ self.outputs

        return efficiency, lambdas, input_targets, output_targets

    def solve_output_oriented_multiplier(self, dmu_index: int, epsilon: float = 1e-6) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Solve Output-Oriented CCR Multiplier Model (3.3.2)
        
        min sum(v_i * x_ip)
        s.t. -sum(v_i * x_ij) + sum(u_r * y_rj) <= 0, j=1,...,n
             sum(u_r * y_rp) = 1
             u_r >= epsilon, v_i >= epsilon
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation (0-based)
        epsilon : float
            Small positive value for non-Archimedean constraint
        
        Returns:
        --------
        efficiency : float
            Efficiency score
        v_weights : np.ndarray
            Optimal input weights (v*)
        u_weights : np.ndarray
            Optimal output weights (u*)
        """
        c = np.zeros(self.n_inputs + self.n_outputs)
        c[:self.n_inputs] = self.inputs[dmu_index, :]

        n_constraints = self.n_dmus + 1 + self.n_inputs + self.n_outputs
        A = np.zeros((n_constraints, self.n_inputs + self.n_outputs))

        for j in range(self.n_dmus):
            A[j, :self.n_inputs] = -self.inputs[j, :]
            A[j, self.n_inputs:] = self.outputs[j, :]

        A[self.n_dmus, self.n_inputs:] = self.outputs[dmu_index, :]

        for i in range(self.n_inputs):
            A[self.n_dmus + 1 + i, i] = -1.0
        for r in range(self.n_outputs):
            A[self.n_dmus + 1 + self.n_inputs + r, self.n_inputs + r] = -1.0

        b = np.zeros(n_constraints)
        b[self.n_dmus] = 1.0
        for i in range(self.n_inputs):
            b[self.n_dmus + 1 + i] = -epsilon
        for r in range(self.n_outputs):
            b[self.n_dmus + 1 + self.n_inputs + r] = -epsilon

        A_eq = A[self.n_dmus:self.n_dmus+1, :]
        b_eq = b[self.n_dmus:self.n_dmus+1]
        A_ub = np.vstack([A[:self.n_dmus, :], A[self.n_dmus+1:, :]])
        b_ub = np.hstack([b[:self.n_dmus], b[self.n_dmus+1:]])

        bounds = [(0, None)] * (self.n_inputs + self.n_outputs)

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')

        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")

        efficiency = result.fun
        v_weights = result.x[:self.n_inputs]
        u_weights = result.x[self.n_inputs:]

        return efficiency, v_weights, u_weights

