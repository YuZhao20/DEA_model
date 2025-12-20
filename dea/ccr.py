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
        # Objective: minimize h (theta)
        # Variables: [h, lambda_1, lambda_2, ..., lambda_n]
        c = np.zeros(self.n_dmus + 1)
        c[0] = 1.0  # coefficient for h
        
        # Constraints matrix
        # Number of constraints: m (input constraints) + s (output constraints)
        n_constraints = self.n_inputs + self.n_outputs
        A = np.zeros((n_constraints, self.n_dmus + 1))
        
        # Input constraints: -h*x_ip + sum(lambda_j * x_ij) >= 0
        # Rewritten as: sum(lambda_j * x_ij) >= h*x_ip
        # For linprog (A_ub x <= b_ub): -sum(lambda_j * x_ij) + h*x_ip <= 0
        # But we need: sum(lambda_j * x_ij) - h*x_ip >= 0
        # So: -sum(lambda_j * x_ij) + h*x_ip <= 0
        for i in range(self.n_inputs):
            A[i, 0] = -self.inputs[dmu_index, i]  # coefficient for h (negative)
            A[i, 1:] = self.inputs[:, i]  # coefficients for lambdas (positive)
        
        # Output constraints: sum(lambda_j * y_rj) >= y_rp
        # For linprog: -sum(lambda_j * y_rj) <= -y_rp
        for r in range(self.n_outputs):
            A[self.n_inputs + r, 0] = 0.0  # h doesn't appear in output constraints
            A[self.n_inputs + r, 1:] = -self.outputs[:, r]  # coefficients for lambdas (negative for <= constraint)
        
        # Right-hand side
        b = np.zeros(n_constraints)
        # Input constraints: b = 0 (already initialized)
        for r in range(self.n_outputs):
            b[self.n_inputs + r] = -self.outputs[dmu_index, r]  # negative of output value
        
        # Bounds: all variables >= 0
        bounds = [(0, None)] * (self.n_dmus + 1)
        
        # Solve linear program
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        efficiency = result.x[0]
        lambdas = result.x[1:]
        
        # Calculate target inputs and outputs
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
        # Objective: maximize sum(u_r * y_rp)
        # Variables: [v_1, ..., v_m, u_1, ..., u_s]
        c = np.zeros(self.n_inputs + self.n_outputs)
        c[self.n_inputs:] = -self.outputs[dmu_index, :]  # negative because linprog minimizes
        
        # Constraints matrix
        # Number of constraints: n (DMU constraints) + 1 (normalization) + m + s (epsilon constraints)
        n_constraints = self.n_dmus + 1 + self.n_inputs + self.n_outputs
        A = np.zeros((n_constraints, self.n_inputs + self.n_outputs))
        
        # DMU constraints: -sum(v_i * x_ij) + sum(u_r * y_rj) <= 0
        for j in range(self.n_dmus):
            A[j, :self.n_inputs] = -self.inputs[j, :]  # coefficients for v_i
            A[j, self.n_inputs:] = self.outputs[j, :]  # coefficients for u_r
        
        # Normalization constraint: sum(v_i * x_ip) = 1
        A[self.n_dmus, :self.n_inputs] = self.inputs[dmu_index, :]
        
        # Epsilon constraints: v_i >= epsilon, u_r >= epsilon
        # For linprog: -v_i <= -epsilon, -u_r <= -epsilon
        for i in range(self.n_inputs):
            A[self.n_dmus + 1 + i, i] = -1.0
        for r in range(self.n_outputs):
            A[self.n_dmus + 1 + self.n_inputs + r, self.n_inputs + r] = -1.0
        
        # Right-hand side
        b = np.zeros(n_constraints)
        b[self.n_dmus] = 1.0  # normalization constraint
        for i in range(self.n_inputs):
            b[self.n_dmus + 1 + i] = -epsilon
        for r in range(self.n_outputs):
            b[self.n_dmus + 1 + self.n_inputs + r] = -epsilon
        
        # Constraint types
        A_eq = A[self.n_dmus:self.n_dmus+1, :]
        b_eq = b[self.n_dmus:self.n_dmus+1]
        A_ub = np.vstack([A[:self.n_dmus, :], A[self.n_dmus+1:, :]])
        b_ub = np.hstack([b[:self.n_dmus], b[self.n_dmus+1:]])
        
        # Bounds: all variables >= 0
        bounds = [(0, None)] * (self.n_inputs + self.n_outputs)
        
        # Solve linear program
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                        bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        efficiency = -result.fun  # negate because we minimized negative of objective
        v_weights = result.x[:self.n_inputs]
        u_weights = result.x[self.n_inputs:]
        
        # Get dual variables (lambdas) from the dual problem
        # Note: scipy's linprog doesn't directly provide dual variables for inequality constraints
        # We need to solve the envelopment form to get lambdas, or use a different approach
        # For now, return None and calculate separately if needed
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
            else:  # multiplier
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

