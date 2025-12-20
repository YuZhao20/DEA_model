"""
BCC (Banker-Charnes-Cooper) DEA Models
Based on Chapter 3.2.3 of Hosseinzadeh Lotfi et al. (2020)
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Optional
import pandas as pd


class BCCModel:
    """
    Input-Oriented BCC DEA Model
    
    The BCC model assumes variable returns to scale (VRS).
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        """
        Initialize BCC model
        
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
        Solve Input-Oriented BCC Envelopment Model
        
        min h
        s.t. -h*x_ip + sum(lambda_j * x_ij) >= 0, i=1,...,m
             sum(lambda_j * y_rj) >= y_rp, r=1,...,s
             sum(lambda_j) = 1
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
        # Number of constraints: m (input constraints) + s (output constraints) + 1 (convexity)
        n_constraints = self.n_inputs + self.n_outputs + 1
        A = np.zeros((n_constraints, self.n_dmus + 1))
        
        # Input constraints: -h*x_ip + sum(lambda_j * x_ij) >= 0
        # For linprog: sum(lambda_j * x_ij) - h*x_ip >= 0
        # So: -sum(lambda_j * x_ij) + h*x_ip <= 0
        for i in range(self.n_inputs):
            A[i, 0] = -self.inputs[dmu_index, i]  # coefficient for h (negative)
            A[i, 1:] = self.inputs[:, i]  # coefficients for lambdas (positive)
        
        # Output constraints: sum(lambda_j * y_rj) >= y_rp
        for r in range(self.n_outputs):
            A[self.n_inputs + r, 0] = 0.0
            A[self.n_inputs + r, 1:] = -self.outputs[:, r]
        
        # Convexity constraint: sum(lambda_j) = 1
        A[self.n_inputs + self.n_outputs, 0] = 0.0
        A[self.n_inputs + self.n_outputs, 1:] = -np.ones(self.n_dmus)
        
        # Right-hand side
        b = np.zeros(n_constraints)
        for r in range(self.n_outputs):
            b[self.n_inputs + r] = -self.outputs[dmu_index, r]
        b[self.n_inputs + self.n_outputs] = -1.0  # convexity constraint
        
        # Constraint types: inequality for inputs/outputs, equality for convexity
        A_eq = A[self.n_inputs + self.n_outputs:self.n_inputs + self.n_outputs + 1, :]
        b_eq = b[self.n_inputs + self.n_outputs:self.n_inputs + self.n_outputs + 1]
        A_ub = A[:self.n_inputs + self.n_outputs, :]
        b_ub = b[:self.n_inputs + self.n_outputs]
        
        # Bounds: all variables >= 0
        bounds = [(0, None)] * (self.n_dmus + 1)
        
        # Solve linear program
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                        bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        efficiency = result.x[0]
        lambdas = result.x[1:]
        
        # Calculate target inputs and outputs
        input_targets = lambdas @ self.inputs
        output_targets = lambdas @ self.outputs
        
        return efficiency, lambdas, input_targets, output_targets
    
    def solve_multiplier(self, dmu_index: int, epsilon: float = 1e-6) -> Tuple[float, np.ndarray, np.ndarray, float]:
        """
        Solve Input-Oriented BCC Multiplier Model (3.5)
        
        max sum(u_r * y_rp) + u0
        s.t. -sum(v_i * x_ij) + sum(u_r * y_rj) + u0 <= 0, j=1,...,n
             sum(v_i * x_ip) = 1
             u_r >= epsilon, v_i >= epsilon
             u0 free in sign (represented as u0+ - u0-)
        
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
        u0 : float
            Optimal u0 value (free in sign)
        """
        # Objective: maximize sum(u_r * y_rp) + u0
        # Variables: [v_1, ..., v_m, u_1, ..., u_s, u0+, u0-]
        # u0 = u0+ - u0- where u0+, u0- >= 0
        c = np.zeros(self.n_inputs + self.n_outputs + 2)
        c[self.n_inputs:self.n_inputs + self.n_outputs] = -self.outputs[dmu_index, :]  # negative for minimization
        c[self.n_inputs + self.n_outputs] = -1.0  # u0+
        c[self.n_inputs + self.n_outputs + 1] = 1.0  # -u0- (so u0+ - u0-)
        
        # Constraints matrix
        # Number of constraints: n (DMU constraints) + 1 (normalization) + m + s + 2 (epsilon for u0+, u0-)
        n_constraints = self.n_dmus + 1 + self.n_inputs + self.n_outputs + 2
        A = np.zeros((n_constraints, self.n_inputs + self.n_outputs + 2))
        
        # DMU constraints: -sum(v_i * x_ij) + sum(u_r * y_rj) + u0 <= 0
        for j in range(self.n_dmus):
            A[j, :self.n_inputs] = -self.inputs[j, :]  # coefficients for v_i
            A[j, self.n_inputs:self.n_inputs + self.n_outputs] = self.outputs[j, :]  # coefficients for u_r
            A[j, self.n_inputs + self.n_outputs] = 1.0  # u0+
            A[j, self.n_inputs + self.n_outputs + 1] = -1.0  # -u0-
        
        # Normalization constraint: sum(v_i * x_ip) = 1
        A[self.n_dmus, :self.n_inputs] = self.inputs[dmu_index, :]
        
        # Epsilon constraints: v_i >= epsilon, u_r >= epsilon, u0+ >= epsilon, u0- >= epsilon
        for i in range(self.n_inputs):
            A[self.n_dmus + 1 + i, i] = -1.0
        for r in range(self.n_outputs):
            A[self.n_dmus + 1 + self.n_inputs + r, self.n_inputs + r] = -1.0
        A[self.n_dmus + 1 + self.n_inputs + self.n_outputs, self.n_inputs + self.n_outputs] = -1.0  # u0+
        A[self.n_dmus + 1 + self.n_inputs + self.n_outputs + 1, self.n_inputs + self.n_outputs + 1] = -1.0  # u0-
        
        # Right-hand side
        b = np.zeros(n_constraints)
        b[self.n_dmus] = 1.0  # normalization constraint
        for i in range(self.n_inputs):
            b[self.n_dmus + 1 + i] = -epsilon
        for r in range(self.n_outputs):
            b[self.n_dmus + 1 + self.n_inputs + r] = -epsilon
        b[self.n_dmus + 1 + self.n_inputs + self.n_outputs] = -epsilon
        b[self.n_dmus + 1 + self.n_inputs + self.n_outputs + 1] = -epsilon
        
        # Constraint types
        A_eq = A[self.n_dmus:self.n_dmus+1, :]
        b_eq = b[self.n_dmus:self.n_dmus+1]
        A_ub = np.vstack([A[:self.n_dmus, :], A[self.n_dmus+1:, :]])
        b_ub = np.hstack([b[:self.n_dmus], b[self.n_dmus+1:]])
        
        # Bounds: all variables >= 0
        bounds = [(0, None)] * (self.n_inputs + self.n_outputs + 2)
        
        # Solve linear program
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                        bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        efficiency = -result.fun  # negate because we minimized negative of objective
        v_weights = result.x[:self.n_inputs]
        u_weights = result.x[self.n_inputs:self.n_inputs + self.n_outputs]
        u0_plus = result.x[self.n_inputs + self.n_outputs]
        u0_minus = result.x[self.n_inputs + self.n_outputs + 1]
        u0 = u0_plus - u0_minus
        
        return efficiency, v_weights, u_weights, u0
    
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
                eff, v, u, u0 = self.solve_multiplier(j)
                result_dict = {
                    'DMU': j + 1,
                    'Efficiency': eff
                }
                for i in range(self.n_inputs):
                    result_dict[f'v_{i+1}'] = v[i]
                for r in range(self.n_outputs):
                    result_dict[f'u_{r+1}'] = u[r]
                result_dict['u0'] = u0
            
            results.append(result_dict)
        
        return pd.DataFrame(results)
    
    def solve_output_oriented_envelopment(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Output-Oriented BCC Envelopment Model (3.3.4)
        
        max u
        s.t. sum(lambda_j * x_ij) <= x_ip, i=1,...,m
             -u*y_rp + sum(lambda_j * y_rj) >= 0, r=1,...,s
             sum(lambda_j) = 1
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
        # Objective: maximize u
        c = np.zeros(self.n_dmus + 1)
        c[0] = -1.0
        
        # Constraints matrix
        n_constraints = self.n_inputs + self.n_outputs + 1
        A = np.zeros((n_constraints, self.n_dmus + 1))
        
        # Input constraints: sum(lambda_j * x_ij) <= x_ip
        for i in range(self.n_inputs):
            A[i, 0] = 0.0
            A[i, 1:] = self.inputs[:, i]
        
        # Output constraints: -u*y_rp + sum(lambda_j * y_rj) >= 0
        for r in range(self.n_outputs):
            A[self.n_inputs + r, 0] = self.outputs[dmu_index, r]
            A[self.n_inputs + r, 1:] = -self.outputs[:, r]
        
        # Convexity constraint: sum(lambda_j) = 1
        A[self.n_inputs + self.n_outputs, 0] = 0.0
        A[self.n_inputs + self.n_outputs, 1:] = -np.ones(self.n_dmus)
        
        # Right-hand side
        b = np.zeros(n_constraints)
        for i in range(self.n_inputs):
            b[i] = self.inputs[dmu_index, i]
        b[self.n_inputs + self.n_outputs] = -1.0
        
        # Constraint types
        A_eq = A[self.n_inputs + self.n_outputs:self.n_inputs + self.n_outputs + 1, :]
        b_eq = b[self.n_inputs + self.n_outputs:self.n_inputs + self.n_outputs + 1]
        A_ub = A[:self.n_inputs + self.n_outputs, :]
        b_ub = b[:self.n_inputs + self.n_outputs]
        
        # Bounds
        bounds = [(0, None)] * (self.n_dmus + 1)
        
        # Solve
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        efficiency = -result.fun
        lambdas = result.x[1:]
        
        input_targets = lambdas @ self.inputs
        output_targets = lambdas @ self.outputs
        
        return efficiency, lambdas, input_targets, output_targets
    
    def solve_output_oriented_multiplier(self, dmu_index: int, epsilon: float = 1e-6) -> Tuple[float, np.ndarray, np.ndarray, float]:
        """
        Solve Output-Oriented BCC Multiplier Model (3.3.3)
        
        min sum(v_i * x_ip) + u0
        s.t. -sum(v_i * x_ij) + sum(u_r * y_rj) + u0 <= 0, j=1,...,n
             sum(u_r * y_rp) = 1
             u_r >= epsilon, v_i >= epsilon
             u0 free in sign
        
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
        u0 : float
            Optimal u0 value
        """
        # Objective: minimize sum(v_i * x_ip) + u0
        # Variables: [v_1, ..., v_m, u_1, ..., u_s, u0+, u0-]
        c = np.zeros(self.n_inputs + self.n_outputs + 2)
        c[:self.n_inputs] = self.inputs[dmu_index, :]
        c[self.n_inputs + self.n_outputs] = 1.0  # u0+
        c[self.n_inputs + self.n_outputs + 1] = -1.0  # -u0-
        
        # Constraints matrix
        n_constraints = self.n_dmus + 1 + self.n_inputs + self.n_outputs + 2
        A = np.zeros((n_constraints, self.n_inputs + self.n_outputs + 2))
        
        # DMU constraints
        for j in range(self.n_dmus):
            A[j, :self.n_inputs] = -self.inputs[j, :]
            A[j, self.n_inputs:self.n_inputs + self.n_outputs] = self.outputs[j, :]
            A[j, self.n_inputs + self.n_outputs] = 1.0  # u0+
            A[j, self.n_inputs + self.n_outputs + 1] = -1.0  # -u0-
        
        # Normalization constraint: sum(u_r * y_rp) = 1
        A[self.n_dmus, self.n_inputs:self.n_inputs + self.n_outputs] = self.outputs[dmu_index, :]
        
        # Epsilon constraints
        for i in range(self.n_inputs):
            A[self.n_dmus + 1 + i, i] = -1.0
        for r in range(self.n_outputs):
            A[self.n_dmus + 1 + self.n_inputs + r, self.n_inputs + r] = -1.0
        A[self.n_dmus + 1 + self.n_inputs + self.n_outputs, self.n_inputs + self.n_outputs] = -1.0
        A[self.n_dmus + 1 + self.n_inputs + self.n_outputs + 1, self.n_inputs + self.n_outputs + 1] = -1.0
        
        # Right-hand side
        b = np.zeros(n_constraints)
        b[self.n_dmus] = 1.0
        for i in range(self.n_inputs):
            b[self.n_dmus + 1 + i] = -epsilon
        for r in range(self.n_outputs):
            b[self.n_dmus + 1 + self.n_inputs + r] = -epsilon
        b[self.n_dmus + 1 + self.n_inputs + self.n_outputs] = -epsilon
        b[self.n_dmus + 1 + self.n_inputs + self.n_outputs + 1] = -epsilon
        
        # Constraint types
        A_eq = A[self.n_dmus:self.n_dmus+1, :]
        b_eq = b[self.n_dmus:self.n_dmus+1]
        A_ub = np.vstack([A[:self.n_dmus, :], A[self.n_dmus+1:, :]])
        b_ub = np.hstack([b[:self.n_dmus], b[self.n_dmus+1:]])
        
        # Bounds
        bounds = [(0, None)] * (self.n_inputs + self.n_outputs + 2)
        
        # Solve
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        efficiency = result.fun
        v_weights = result.x[:self.n_inputs]
        u_weights = result.x[self.n_inputs:self.n_inputs + self.n_outputs]
        u0_plus = result.x[self.n_inputs + self.n_outputs]
        u0_minus = result.x[self.n_inputs + self.n_outputs + 1]
        u0 = u0_plus - u0_minus
        
        return efficiency, v_weights, u_weights, u0

