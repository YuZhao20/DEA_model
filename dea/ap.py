"""
AP (Anderson-Peterson) Super-Efficiency Models
Based on Chapter 4.2 of Hosseinzadeh Lotfi et al. (2020)
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple
import pandas as pd


class APModel:
    """
    AP (Anderson-Peterson) Super-Efficiency DEA Model
    
    The AP model excludes the DMU under evaluation from the reference set,
    allowing efficient DMUs to be ranked.
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        """
        Initialize AP model
        
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
    
    def solve_input_oriented_envelopment(self, dmu_index: int) -> Tuple[float, np.ndarray]:
        """
        Solve Input-Oriented AP Envelopment Model (4.1)
        
        min h
        s.t. -h*x_ip + sum(lambda_j * x_ij) >= 0, i=1,...,m, j!=p
             sum(lambda_j * y_rj) >= y_rp, r=1,...,s, j!=p
             lambda_j >= 0, j=1,...,n, j!=p
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation (0-based)
        
        Returns:
        --------
        efficiency : float
            Super-efficiency score (h*)
        lambdas : np.ndarray
            Optimal intensity variables (lambda*)
        """
        # Objective: minimize h (theta)
        # Variables: [h, lambda_1, ..., lambda_n] but we'll set lambda_p = 0
        n_vars = self.n_dmus + 1  # h + all lambdas
        c = np.zeros(n_vars)
        c[0] = 1.0  # coefficient for h
        
        # Constraints matrix
        n_constraints = self.n_inputs + self.n_outputs
        A = np.zeros((n_constraints, n_vars))
        
        # Input constraints: -h*x_ip + sum(lambda_j * x_ij) >= 0, j!=p
        # For linprog: sum(lambda_j * x_ij) - h*x_ip >= 0
        # So: -sum(lambda_j * x_ij) + h*x_ip <= 0
        for i in range(self.n_inputs):
            A[i, 0] = -self.inputs[dmu_index, i]  # coefficient for h
            for j in range(self.n_dmus):
                if j != dmu_index:
                    A[i, j + 1] = self.inputs[j, i]  # coefficients for lambdas (j != p)
                else:
                    A[i, j + 1] = 0.0  # lambda_p = 0 (excluded)
        
        # Output constraints: sum(lambda_j * y_rj) >= y_rp, j!=p
        for r in range(self.n_outputs):
            A[self.n_inputs + r, 0] = 0.0  # h doesn't appear
            for j in range(self.n_dmus):
                if j != dmu_index:
                    A[self.n_inputs + r, j + 1] = -self.outputs[j, r]
                else:
                    A[self.n_inputs + r, j + 1] = 0.0  # lambda_p = 0 (excluded)
        
        # Right-hand side
        b = np.zeros(n_constraints)
        for r in range(self.n_outputs):
            b[self.n_inputs + r] = -self.outputs[dmu_index, r]
        
        # Bounds: all variables >= 0
        bounds = [(0, None)] * n_vars
        
        # Solve linear program
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        efficiency = result.x[0]
        lambdas = result.x[1:]
        
        return efficiency, lambdas
    
    def solve_output_oriented_envelopment(self, dmu_index: int) -> Tuple[float, np.ndarray]:
        """
        Solve Output-Oriented AP Envelopment Model (4.3)
        
        max u
        s.t. sum(lambda_j * x_ij) <= x_ip, i=1,...,m, j!=p
             -u*y_rp + sum(lambda_j * y_rj) >= 0, r=1,...,s, j!=p
             lambda_j >= 0, j=1,...,n, j!=p
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation (0-based)
        
        Returns:
        --------
        efficiency : float
            Super-efficiency score (u*)
        lambdas : np.ndarray
            Optimal intensity variables (lambda*)
        """
        # Objective: maximize u
        # Variables: [u, lambda_1, ..., lambda_n] but we'll set lambda_p = 0
        n_vars = self.n_dmus + 1
        c = np.zeros(n_vars)
        c[0] = -1.0  # negative because linprog minimizes
        
        # Constraints matrix
        n_constraints = self.n_inputs + self.n_outputs
        A = np.zeros((n_constraints, n_vars))
        
        # Input constraints: sum(lambda_j * x_ij) <= x_ip, j!=p
        for i in range(self.n_inputs):
            A[i, 0] = 0.0  # u doesn't appear
            for j in range(self.n_dmus):
                if j != dmu_index:
                    A[i, j + 1] = self.inputs[j, i]
                else:
                    A[i, j + 1] = 0.0  # lambda_p = 0 (excluded)
        
        # Output constraints: -u*y_rp + sum(lambda_j * y_rj) >= 0, j!=p
        # For linprog: u*y_rp - sum(lambda_j * y_rj) <= 0
        for r in range(self.n_outputs):
            A[self.n_inputs + r, 0] = self.outputs[dmu_index, r]  # coefficient for u
            for j in range(self.n_dmus):
                if j != dmu_index:
                    A[self.n_inputs + r, j + 1] = -self.outputs[j, r]
                else:
                    A[self.n_inputs + r, j + 1] = 0.0  # lambda_p = 0 (excluded)
        
        # Right-hand side
        b = np.zeros(n_constraints)
        for i in range(self.n_inputs):
            b[i] = self.inputs[dmu_index, i]
        
        # Bounds: all variables >= 0
        bounds = [(0, None)] * n_vars
        
        # Solve linear program
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        efficiency = -result.fun  # negate because we minimized negative
        lambdas = result.x[1:]
        
        return efficiency, lambdas
    
    def solve_input_oriented_multiplier(self, dmu_index: int, epsilon: float = 1e-6) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Solve Input-Oriented AP Multiplier Model (4.5)
        
        max sum(u_r * y_rp)
        s.t. -sum(v_i * x_ij) + sum(u_r * y_rj) <= 0, j=1,...,n, j!=p
             sum(v_i * x_ip) = 1
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
            Super-efficiency score
        v_weights : np.ndarray
            Optimal input weights (v*)
        u_weights : np.ndarray
            Optimal output weights (u*)
        """
        # Objective: maximize sum(u_r * y_rp)
        # Variables: [v_1, ..., v_m, u_1, ..., u_s]
        c = np.zeros(self.n_inputs + self.n_outputs)
        c[self.n_inputs:] = -self.outputs[dmu_index, :]  # negative for minimization
        
        # Constraints: exclude DMU p
        # Number of constraints: (n-1) DMU constraints + 1 normalization + m + s epsilon
        n_constraints = (self.n_dmus - 1) + 1 + self.n_inputs + self.n_outputs
        A = np.zeros((n_constraints, self.n_inputs + self.n_outputs))
        
        # DMU constraints: -sum(v_i * x_ij) + sum(u_r * y_rj) <= 0, j!=p
        constraint_idx = 0
        for j in range(self.n_dmus):
            if j != dmu_index:
                A[constraint_idx, :self.n_inputs] = -self.inputs[j, :]
                A[constraint_idx, self.n_inputs:] = self.outputs[j, :]
                constraint_idx += 1
        
        # Normalization constraint: sum(v_i * x_ip) = 1
        A[constraint_idx, :self.n_inputs] = self.inputs[dmu_index, :]
        constraint_idx += 1
        
        # Epsilon constraints
        for i in range(self.n_inputs):
            A[constraint_idx, i] = -1.0
            constraint_idx += 1
        for r in range(self.n_outputs):
            A[constraint_idx, self.n_inputs + r] = -1.0
            constraint_idx += 1
        
        # Right-hand side
        b = np.zeros(n_constraints)
        b[self.n_dmus - 1] = 1.0  # normalization
        for i in range(self.n_inputs):
            b[self.n_dmus + i] = -epsilon
        for r in range(self.n_outputs):
            b[self.n_dmus + self.n_inputs + r] = -epsilon
        
        # Constraint types
        A_eq = A[self.n_dmus - 1:self.n_dmus, :]
        b_eq = b[self.n_dmus - 1:self.n_dmus]
        A_ub = np.vstack([A[:self.n_dmus - 1, :], A[self.n_dmus:, :]])
        b_ub = np.hstack([b[:self.n_dmus - 1], b[self.n_dmus:]])
        
        # Bounds
        bounds = [(0, None)] * (self.n_inputs + self.n_outputs)
        
        # Solve
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        efficiency = -result.fun
        v_weights = result.x[:self.n_inputs]
        u_weights = result.x[self.n_inputs:]
        
        return efficiency, v_weights, u_weights
    
    def solve_output_oriented_multiplier(self, dmu_index: int, epsilon: float = 1e-6) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Solve Output-Oriented AP Multiplier Model (4.7)
        
        min sum(v_i * x_ip)
        s.t. -sum(v_i * x_ij) + sum(u_r * y_rj) <= 0, j=1,...,n, j!=p
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
            Super-efficiency score
        v_weights : np.ndarray
            Optimal input weights (v*)
        u_weights : np.ndarray
            Optimal output weights (u*)
        """
        # Objective: minimize sum(v_i * x_ip)
        c = np.zeros(self.n_inputs + self.n_outputs)
        c[:self.n_inputs] = self.inputs[dmu_index, :]
        
        # Constraints: exclude DMU p
        n_constraints = (self.n_dmus - 1) + 1 + self.n_inputs + self.n_outputs
        A = np.zeros((n_constraints, self.n_inputs + self.n_outputs))
        
        # DMU constraints: -sum(v_i * x_ij) + sum(u_r * y_rj) <= 0, j!=p
        constraint_idx = 0
        for j in range(self.n_dmus):
            if j != dmu_index:
                A[constraint_idx, :self.n_inputs] = -self.inputs[j, :]
                A[constraint_idx, self.n_inputs:] = self.outputs[j, :]
                constraint_idx += 1
        
        # Normalization constraint: sum(u_r * y_rp) = 1
        A[constraint_idx, self.n_inputs:] = self.outputs[dmu_index, :]
        constraint_idx += 1
        
        # Epsilon constraints
        for i in range(self.n_inputs):
            A[constraint_idx, i] = -1.0
            constraint_idx += 1
        for r in range(self.n_outputs):
            A[constraint_idx, self.n_inputs + r] = -1.0
            constraint_idx += 1
        
        # Right-hand side
        b = np.zeros(n_constraints)
        b[self.n_dmus - 1] = 1.0  # normalization
        for i in range(self.n_inputs):
            b[self.n_dmus + i] = -epsilon
        for r in range(self.n_outputs):
            b[self.n_dmus + self.n_inputs + r] = -epsilon
        
        # Constraint types
        A_eq = A[self.n_dmus - 1:self.n_dmus, :]
        b_eq = b[self.n_dmus - 1:self.n_dmus]
        A_ub = np.vstack([A[:self.n_dmus - 1, :], A[self.n_dmus:, :]])
        b_ub = np.hstack([b[:self.n_dmus - 1], b[self.n_dmus:]])
        
        # Bounds
        bounds = [(0, None)] * (self.n_inputs + self.n_outputs)
        
        # Solve
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        efficiency = result.fun
        v_weights = result.x[:self.n_inputs]
        u_weights = result.x[self.n_inputs:]
        
        return efficiency, v_weights, u_weights
    
    def evaluate_all(self, orientation: str = 'input', method: str = 'envelopment') -> pd.DataFrame:
        """
        Evaluate all DMUs
        
        Parameters:
        -----------
        orientation : str
            'input' or 'output'
        method : str
            'envelopment' or 'multiplier'
        
        Returns:
        --------
        results : pd.DataFrame
            DataFrame with super-efficiency scores
        """
        results = []
        
        for j in range(self.n_dmus):
            if method == 'envelopment':
                if orientation == 'input':
                    eff, lambdas = self.solve_input_oriented_envelopment(j)
                else:
                    eff, lambdas = self.solve_output_oriented_envelopment(j)
                result_dict = {'DMU': j + 1, 'Super_Efficiency': eff}
                for i, lam in enumerate(lambdas):
                    result_dict[f'Lambda_{i+1}'] = lam
            else:  # multiplier
                if orientation == 'input':
                    eff, v, u = self.solve_input_oriented_multiplier(j)
                else:
                    eff, v, u = self.solve_output_oriented_multiplier(j)
                result_dict = {'DMU': j + 1, 'Super_Efficiency': eff}
                for i in range(self.n_inputs):
                    result_dict[f'v_{i+1}'] = v[i]
                for r in range(self.n_outputs):
                    result_dict[f'u_{r+1}'] = u[r]
            
            results.append(result_dict)
        
        return pd.DataFrame(results)

