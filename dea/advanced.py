"""
Advanced DEA Models from Chapter 4
Including: Norm L1, Congestion, Common Weights, Directional Efficiency
Based on Chapter 4 of Hosseinzadeh Lotfi et al. (2020)
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple
import pandas as pd


class NormL1Model:
    """
    Norm L1 Super-Efficiency Model
    Based on Chapter 4.4
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve(self, dmu_index: int) -> Tuple[float, np.ndarray]:
        """
        Solve Norm L1 Super-Efficiency Model
        
        Uses L1 norm to measure distance from efficient frontier
        """
        # This is a simplified implementation
        # The full model involves minimizing L1 norm distance
        # For now, we'll use a basic approach
        raise NotImplementedError("Norm L1 model implementation in progress")


class CongestionModel:
    """
    Congestion DEA Model
    Based on Chapter 4.13
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Solve Congestion DEA Model (4.13)
        
        Two-phase model:
        Phase 1: Solve BCC model to get efficiency
        Phase 2: Find congestion slacks
        """
        # Phase 1: BCC model
        from .bcc import BCCModel
        bcc = BCCModel(self.inputs, self.outputs)
        eff, lambdas, input_targets, output_targets = bcc.solve_envelopment(dmu_index)
        
        # Phase 2: Maximize input slacks to find congestion
        n_vars = self.n_dmus + self.n_inputs
        c = np.zeros(n_vars)
        c[self.n_dmus:] = 1.0  # maximize input slacks
        
        n_constraints = self.n_inputs + self.n_outputs + 1
        A_eq = np.zeros((n_constraints, n_vars))
        
        # Input constraints: sum(lambda_j * x_ij) + s_i^- = h*x_ip
        for i in range(self.n_inputs):
            A_eq[i, :self.n_dmus] = self.inputs[:, i]
            A_eq[i, self.n_dmus + i] = 1.0
        
        # Output constraints: sum(lambda_j * y_rj) = y_rp
        for r in range(self.n_outputs):
            A_eq[self.n_inputs + r, :self.n_dmus] = self.outputs[:, r]
        
        # Convexity
        A_eq[self.n_inputs + self.n_outputs, :self.n_dmus] = 1.0
        
        b_eq = np.zeros(n_constraints)
        for i in range(self.n_inputs):
            b_eq[i] = eff * self.inputs[dmu_index, i]
        for r in range(self.n_outputs):
            b_eq[self.n_inputs + r] = self.outputs[dmu_index, r]
        b_eq[self.n_inputs + self.n_outputs] = 1.0
        
        bounds = [(0, None)] * n_vars
        
        result = linprog(-c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        congestion_slacks = result.x[self.n_dmus:]
        congestion_sum = np.sum(congestion_slacks)
        
        return eff, congestion_slacks, congestion_sum
    
    def evaluate_all(self) -> pd.DataFrame:
        results = []
        for j in range(self.n_dmus):
            eff, congestion_slacks, congestion_sum = self.solve(j)
            result_dict = {'DMU': j + 1, 'Efficiency': eff, 'Congestion_Sum': congestion_sum}
            for i in range(self.n_inputs):
                result_dict[f'Congestion_{i+1}'] = congestion_slacks[i]
            results.append(result_dict)
        return pd.DataFrame(results)


class CommonWeightsModel:
    """
    Common Set of Weights DEA Model
    Based on Chapter 4.14
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, epsilon: float = 1e-4):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        self.epsilon = epsilon
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve Common Set of Weights Model (4.14)
        
        min sum(d_j)
        s.t. sum(u_r * y_rj) - sum(v_i * x_ij) + d_j = 0, j=1,...,n
             u_r >= epsilon, v_i >= epsilon
             d_j >= 0
        """
        # Variables: [u_1, ..., u_s, v_1, ..., v_m, d_1, ..., d_n]
        n_vars = self.n_outputs + self.n_inputs + self.n_dmus
        c = np.zeros(n_vars)
        c[self.n_outputs + self.n_inputs:] = 1.0  # minimize sum of d_j
        
        # Constraints: n equality constraints + m + s epsilon constraints
        n_constraints = self.n_dmus + self.n_inputs + self.n_outputs
        A_eq = np.zeros((self.n_dmus, n_vars))
        
        # DMU constraints: sum(u_r * y_rj) - sum(v_i * x_ij) + d_j = 0
        for j in range(self.n_dmus):
            A_eq[j, :self.n_outputs] = self.outputs[j, :]  # u_r * y_rj
            A_eq[j, self.n_outputs:self.n_outputs + self.n_inputs] = -self.inputs[j, :]  # -v_i * x_ij
            A_eq[j, self.n_outputs + self.n_inputs + j] = 1.0  # d_j
        
        b_eq = np.zeros(self.n_dmus)
        
        # Epsilon constraints: u_r >= epsilon, v_i >= epsilon
        A_ub = np.zeros((self.n_inputs + self.n_outputs, n_vars))
        for r in range(self.n_outputs):
            A_ub[r, r] = -1.0
        for i in range(self.n_inputs):
            A_ub[self.n_outputs + i, self.n_outputs + i] = -1.0
        
        b_ub = np.zeros(self.n_inputs + self.n_outputs)
        for r in range(self.n_outputs):
            b_ub[r] = -self.epsilon
        for i in range(self.n_inputs):
            b_ub[self.n_outputs + i] = -self.epsilon
        
        bounds = [(0, None)] * n_vars
        
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                        bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
        
        u_weights = result.x[:self.n_outputs]
        v_weights = result.x[self.n_outputs:self.n_outputs + self.n_inputs]
        d_values = result.x[self.n_outputs + self.n_inputs:]
        obj_value = result.fun
        
        return u_weights, v_weights, obj_value


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
    
    def solve(self, dmu_index: int, gx: np.ndarray = None, gy: np.ndarray = None) -> Tuple[float, np.ndarray]:
        """
        Solve Directional Efficiency Model (4.15)
        
        max beta
        s.t. beta*x_ip + sum(lambda_j * x_ij) <= x_ip, i=1,...,m
             -beta*y_rp + sum(lambda_j * y_rj) >= y_rp, r=1,...,s
             lambda_j >= 0
        
        Default direction: gx = -x_p, gy = y_p
        """
        if gx is None:
            gx = -self.inputs[dmu_index, :]
        if gy is None:
            gy = self.outputs[dmu_index, :]
        
        # Objective: maximize beta
        c = np.zeros(self.n_dmus + 1)
        c[0] = -1.0  # negative because linprog minimizes
        
        # Constraints
        n_constraints = self.n_inputs + self.n_outputs
        A = np.zeros((n_constraints, self.n_dmus + 1))
        
        # Input constraints: beta*gx_i + sum(lambda_j * x_ij) <= x_ip
        for i in range(self.n_inputs):
            A[i, 0] = gx[i]  # beta coefficient
            A[i, 1:] = self.inputs[:, i]  # lambda coefficients
        
        # Output constraints: -beta*gy_r + sum(lambda_j * y_rj) >= y_rp
        # For linprog: beta*gy_r - sum(lambda_j * y_rj) <= -y_rp
        for r in range(self.n_outputs):
            A[self.n_inputs + r, 0] = gy[r]  # beta coefficient
            A[self.n_inputs + r, 1:] = -self.outputs[:, r]  # lambda coefficients
        
        # Right-hand side
        b = np.zeros(n_constraints)
        for i in range(self.n_inputs):
            b[i] = self.inputs[dmu_index, i]
        for r in range(self.n_outputs):
            b[self.n_inputs + r] = -self.outputs[dmu_index, r]
        
        bounds = [(0, None)] * (self.n_dmus + 1)
        
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        efficiency = -result.fun
        lambdas = result.x[1:]
        
        return efficiency, lambdas
    
    def evaluate_all(self, gx: np.ndarray = None, gy: np.ndarray = None) -> pd.DataFrame:
        results = []
        for j in range(self.n_dmus):
            eff, lambdas = self.solve(j, gx, gy)
            result_dict = {'DMU': j + 1, 'Efficiency': eff}
            for i, lam in enumerate(lambdas):
                result_dict[f'Lambda_{i+1}'] = lam
            results.append(result_dict)
        return pd.DataFrame(results)

