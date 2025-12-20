"""
Additional Returns to Scale DEA Models
DRS (Decreasing Returns to Scale) and IRS (Increasing Returns to Scale)
Based on Benchmarking package
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple
import pandas as pd


class DRSModel:
    """
    Decreasing Returns to Scale (DRS) DEA Model
    
    DRS assumes that sum of lambdas <= 1 (non-increasing returns to scale)
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve_envelopment(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Input-Oriented DRS Envelopment Model
        
        min h
        s.t. -h*x_ip + sum(lambda_j * x_ij) >= 0, i=1,...,m
             sum(lambda_j * y_rj) >= y_rp, r=1,...,s
             sum(lambda_j) <= 1
             lambda_j >= 0
        
        Returns:
        --------
        efficiency : float
            Efficiency score
        lambdas : np.ndarray
            Optimal intensity variables
        input_targets : np.ndarray
            Target input values
        output_targets : np.ndarray
            Target output values
        """
        # Variables: [h, lambda_1, ..., lambda_n]
        c = np.zeros(self.n_dmus + 1)
        c[0] = 1.0
        
        n_constraints = self.n_inputs + self.n_outputs + 1
        A = np.zeros((n_constraints, self.n_dmus + 1))
        
        # Input constraints
        for i in range(self.n_inputs):
            A[i, 0] = -self.inputs[dmu_index, i]
            A[i, 1:] = self.inputs[:, i]
        
        # Output constraints
        for r in range(self.n_outputs):
            A[self.n_inputs + r, 1:] = -self.outputs[:, r]
        
        # DRS constraint: sum(lambda_j) <= 1
        A[self.n_inputs + self.n_outputs, 1:] = 1.0
        
        b = np.zeros(n_constraints)
        for r in range(self.n_outputs):
            b[self.n_inputs + r] = -self.outputs[dmu_index, r]
        b[self.n_inputs + self.n_outputs] = 1.0  # sum(lambda) <= 1
        
        A_eq = None
        b_eq = None
        A_ub = A
        b_ub = b
        
        bounds = [(0, None)] * (self.n_dmus + 1)
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        efficiency = result.x[0]
        lambdas = result.x[1:]
        input_targets = lambdas @ self.inputs
        output_targets = lambdas @ self.outputs
        
        return efficiency, lambdas, input_targets, output_targets
    
    def solve_output_oriented_envelopment(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Output-Oriented DRS Envelopment Model
        
        max g
        s.t. sum(lambda_j * x_ij) <= x_ip, i=1,...,m
             -g*y_rp + sum(lambda_j * y_rj) >= 0, r=1,...,s
             sum(lambda_j) <= 1
             lambda_j >= 0
        """
        c = np.zeros(self.n_dmus + 1)
        c[0] = -1.0  # maximize g (minimize -g)
        
        n_constraints = self.n_inputs + self.n_outputs + 1
        A = np.zeros((n_constraints, self.n_dmus + 1))
        
        # Input constraints: sum(lambda_j * x_ij) <= x_ip
        for i in range(self.n_inputs):
            A[i, 0] = 0.0  # g doesn't appear
            A[i, 1:] = self.inputs[:, i]
        
        # Output constraints: -g*y_rp + sum(lambda_j * y_rj) >= 0
        # For linprog: g*y_rp - sum(lambda_j * y_rj) <= 0
        for r in range(self.n_outputs):
            A[self.n_inputs + r, 0] = self.outputs[dmu_index, r]  # coefficient for g (positive)
            A[self.n_inputs + r, 1:] = -self.outputs[:, r]  # coefficients for lambdas (negative)
        
        # DRS constraint: sum(lambda_j) <= 1
        A[self.n_inputs + self.n_outputs, 0] = 0.0
        A[self.n_inputs + self.n_outputs, 1:] = 1.0
        
        b = np.zeros(n_constraints)
        for i in range(self.n_inputs):
            b[i] = self.inputs[dmu_index, i]
        # Output constraints: b = 0 (already initialized)
        b[self.n_inputs + self.n_outputs] = 1.0  # sum(lambda) <= 1
        
        bounds = [(0, None)] * (self.n_dmus + 1)
        
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        efficiency = -result.fun
        lambdas = result.x[1:]
        input_targets = lambdas @ self.inputs
        output_targets = lambdas @ self.outputs
        
        return efficiency, lambdas, input_targets, output_targets
    
    def evaluate_all(self, orientation: str = 'input') -> pd.DataFrame:
        results = []
        for j in range(self.n_dmus):
            if orientation == 'input':
                eff, lambdas, input_targets, output_targets = self.solve_envelopment(j)
            else:
                eff, lambdas, input_targets, output_targets = self.solve_output_oriented_envelopment(j)
            
            result_dict = {'DMU': j + 1, 'Efficiency': eff}
            for i, lam in enumerate(lambdas):
                result_dict[f'Lambda_{i+1}'] = lam
            results.append(result_dict)
        return pd.DataFrame(results)


class IRSModel:
    """
    Increasing Returns to Scale (IRS) DEA Model
    
    IRS assumes that sum of lambdas >= 1 (non-decreasing returns to scale)
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve_envelopment(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Input-Oriented IRS Envelopment Model
        
        min h
        s.t. -h*x_ip + sum(lambda_j * x_ij) >= 0, i=1,...,m
             sum(lambda_j * y_rj) >= y_rp, r=1,...,s
             sum(lambda_j) >= 1
             lambda_j >= 0
        """
        c = np.zeros(self.n_dmus + 1)
        c[0] = 1.0
        
        n_constraints = self.n_inputs + self.n_outputs + 1
        A = np.zeros((n_constraints, self.n_dmus + 1))
        
        # Input constraints
        for i in range(self.n_inputs):
            A[i, 0] = -self.inputs[dmu_index, i]
            A[i, 1:] = self.inputs[:, i]
        
        # Output constraints
        for r in range(self.n_outputs):
            A[self.n_inputs + r, 1:] = -self.outputs[:, r]
        
        # IRS constraint: sum(lambda_j) >= 1
        # For linprog: -sum(lambda_j) <= -1
        A[self.n_inputs + self.n_outputs, 1:] = -1.0
        
        b = np.zeros(n_constraints)
        for r in range(self.n_outputs):
            b[self.n_inputs + r] = -self.outputs[dmu_index, r]
        b[self.n_inputs + self.n_outputs] = -1.0  # -sum(lambda) <= -1
        
        bounds = [(0, None)] * (self.n_dmus + 1)
        
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        efficiency = result.x[0]
        lambdas = result.x[1:]
        input_targets = lambdas @ self.inputs
        output_targets = lambdas @ self.outputs
        
        return efficiency, lambdas, input_targets, output_targets
    
    def solve_output_oriented_envelopment(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Output-Oriented IRS Envelopment Model
        
        max g
        s.t. sum(lambda_j * x_ij) <= x_ip, i=1,...,m
             -g*y_rp + sum(lambda_j * y_rj) >= 0, r=1,...,s
             sum(lambda_j) >= 1
             lambda_j >= 0
        """
        c = np.zeros(self.n_dmus + 1)
        c[0] = -1.0  # maximize g (minimize -g)
        
        n_constraints = self.n_inputs + self.n_outputs + 1
        A = np.zeros((n_constraints, self.n_dmus + 1))
        
        # Input constraints: sum(lambda_j * x_ij) <= x_ip
        for i in range(self.n_inputs):
            A[i, 0] = 0.0  # g doesn't appear
            A[i, 1:] = self.inputs[:, i]
        
        # Output constraints: -g*y_rp + sum(lambda_j * y_rj) >= 0
        # For linprog: g*y_rp - sum(lambda_j * y_rj) <= 0
        for r in range(self.n_outputs):
            A[self.n_inputs + r, 0] = self.outputs[dmu_index, r]  # coefficient for g (positive)
            A[self.n_inputs + r, 1:] = -self.outputs[:, r]  # coefficients for lambdas (negative)
        
        # IRS constraint: sum(lambda_j) >= 1
        # For linprog: -sum(lambda_j) <= -1
        A[self.n_inputs + self.n_outputs, 0] = 0.0
        A[self.n_inputs + self.n_outputs, 1:] = -1.0
        
        b = np.zeros(n_constraints)
        for i in range(self.n_inputs):
            b[i] = self.inputs[dmu_index, i]
        # Output constraints: b = 0 (already initialized)
        b[self.n_inputs + self.n_outputs] = -1.0  # -sum(lambda) <= -1
        
        bounds = [(0, None)] * (self.n_dmus + 1)
        
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        efficiency = -result.fun
        lambdas = result.x[1:]
        input_targets = lambdas @ self.inputs
        output_targets = lambdas @ self.outputs
        
        return efficiency, lambdas, input_targets, output_targets
    
    def evaluate_all(self, orientation: str = 'input') -> pd.DataFrame:
        results = []
        for j in range(self.n_dmus):
            if orientation == 'input':
                eff, lambdas, input_targets, output_targets = self.solve_envelopment(j)
            else:
                eff, lambdas, input_targets, output_targets = self.solve_output_oriented_envelopment(j)
            
            result_dict = {'DMU': j + 1, 'Efficiency': eff}
            for i, lam in enumerate(lambdas):
                result_dict[f'Lambda_{i+1}'] = lam
            results.append(result_dict)
        return pd.DataFrame(results)

