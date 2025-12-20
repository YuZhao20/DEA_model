"""
Two-Phase DEA Models
Based on Chapter 3.6 of Hosseinzadeh Lotfi et al. (2020)
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple
import pandas as pd


class TwoPhaseModel:
    """
    Two-Phase Input-Oriented DEA Envelopment Models
    
    Phase 1: Find efficiency score
    Phase 2: Maximize slacks given the efficiency score from Phase 1
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        """
        Initialize Two-Phase model
        
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
    
    def solve(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Solve Two-Phase Input-Oriented BCC Envelopment Model (3.6.1)
        
        Phase 1:
        min h
        s.t. -h*x_ip + sum(lambda_j * x_ij) >= 0, i=1,...,m
             sum(lambda_j * y_rj) >= y_rp, r=1,...,s
             sum(lambda_j) = 1
             lambda_j >= 0
        
        Phase 2:
        max sum(s_i^-) + sum(s_r^+)
        s.t. sum(lambda_j * x_ij) + s_i^- = h*x_ip, i=1,...,m
             sum(lambda_j * y_rj) - s_r^+ = y_rp, r=1,...,s
             sum(lambda_j) = 1
             lambda_j >= 0, s_i^- >= 0, s_r^+ >= 0
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation (0-based)
        
        Returns:
        --------
        efficiency : float
            Efficiency score from Phase 1 (h*)
        lambdas : np.ndarray
            Optimal intensity variables from Phase 2
        input_slacks : np.ndarray
            Optimal input slacks from Phase 2
        output_slacks : np.ndarray
            Optimal output slacks from Phase 2
        sum_slacks : float
            Sum of optimal slacks from Phase 2
        """
        # Phase 1: Solve BCC envelopment model
        c1 = np.zeros(self.n_dmus + 1)
        c1[0] = 1.0  # minimize h
        
        n_constraints1 = self.n_inputs + self.n_outputs + 1
        A1 = np.zeros((n_constraints1, self.n_dmus + 1))
        
        # Input constraints
        for i in range(self.n_inputs):
            A1[i, 0] = -self.inputs[dmu_index, i]
            A1[i, 1:] = self.inputs[:, i]
        
        # Output constraints
        for r in range(self.n_outputs):
            A1[self.n_inputs + r, 1:] = -self.outputs[:, r]
        
        # Convexity constraint
        A1[self.n_inputs + self.n_outputs, 1:] = -np.ones(self.n_dmus)
        
        b1 = np.zeros(n_constraints1)
        for r in range(self.n_outputs):
            b1[self.n_inputs + r] = -self.outputs[dmu_index, r]
        b1[self.n_inputs + self.n_outputs] = -1.0
        
        A1_eq = A1[self.n_inputs + self.n_outputs:self.n_inputs + self.n_outputs + 1, :]
        b1_eq = b1[self.n_inputs + self.n_outputs:self.n_inputs + self.n_outputs + 1]
        A1_ub = A1[:self.n_inputs + self.n_outputs, :]
        b1_ub = b1[:self.n_inputs + self.n_outputs]
        
        bounds1 = [(0, None)] * (self.n_dmus + 1)
        
        result1 = linprog(c1, A_ub=A1_ub, b_ub=b1_ub, A_eq=A1_eq, b_eq=b1_eq,
                         bounds=bounds1, method='highs')
        
        if not result1.success:
            raise RuntimeError(f"Phase 1 optimization failed for DMU {dmu_index}: {result1.message}")
        
        efficiency = result1.x[0]
        
        # Phase 2: Maximize slacks given efficiency from Phase 1
        n_vars2 = self.n_dmus + self.n_inputs + self.n_outputs
        c2 = np.zeros(n_vars2)
        c2[self.n_dmus:self.n_dmus + self.n_inputs] = 1.0  # input slacks
        c2[self.n_dmus + self.n_inputs:] = 1.0  # output slacks
        
        n_constraints2 = self.n_inputs + self.n_outputs + 1
        A2_eq = np.zeros((n_constraints2, n_vars2))
        
        # Input constraints: sum(lambda_j * x_ij) + s_i^- = h*x_ip
        for i in range(self.n_inputs):
            A2_eq[i, :self.n_dmus] = self.inputs[:, i]
            A2_eq[i, self.n_dmus + i] = 1.0
        
        # Output constraints: sum(lambda_j * y_rj) - s_r^+ = y_rp
        for r in range(self.n_outputs):
            A2_eq[self.n_inputs + r, :self.n_dmus] = self.outputs[:, r]
            A2_eq[self.n_inputs + r, self.n_dmus + self.n_inputs + r] = -1.0
        
        # Convexity constraint
        A2_eq[self.n_inputs + self.n_outputs, :self.n_dmus] = 1.0
        
        b2_eq = np.zeros(n_constraints2)
        for i in range(self.n_inputs):
            b2_eq[i] = efficiency * self.inputs[dmu_index, i]
        for r in range(self.n_outputs):
            b2_eq[self.n_inputs + r] = self.outputs[dmu_index, r]
        b2_eq[self.n_inputs + self.n_outputs] = 1.0
        
        bounds2 = [(0, None)] * n_vars2
        
        result2 = linprog(-c2, A_eq=A2_eq, b_eq=b2_eq, bounds=bounds2, method='highs')
        
        if not result2.success:
            raise RuntimeError(f"Phase 2 optimization failed for DMU {dmu_index}: {result2.message}")
        
        lambdas = result2.x[:self.n_dmus]
        input_slacks = result2.x[self.n_dmus:self.n_dmus + self.n_inputs]
        output_slacks = result2.x[self.n_dmus + self.n_inputs:]
        sum_slacks = -result2.fun
        
        return efficiency, lambdas, input_slacks, output_slacks, sum_slacks
    
    def evaluate_all(self) -> pd.DataFrame:
        """
        Evaluate all DMUs
        
        Returns:
        --------
        results : pd.DataFrame
            DataFrame with results
        """
        results = []
        
        for j in range(self.n_dmus):
            eff, lambdas, input_slacks, output_slacks, sum_slacks = self.solve(j)
            
            result_dict = {
                'DMU': j + 1,
                'Efficiency': eff,
                'Sum_Slacks': sum_slacks
            }
            for i, lam in enumerate(lambdas):
                result_dict[f'Lambda_{i+1}'] = lam
            for i in range(self.n_inputs):
                result_dict[f'S-_{i+1}'] = input_slacks[i]
            for r in range(self.n_outputs):
                result_dict[f'S+_{r+1}'] = output_slacks[r]
            
            results.append(result_dict)
        
        return pd.DataFrame(results)
    
    def solve_ccr(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Solve Two-Phase Input-Oriented CCR Envelopment Model (3.6.2)
        
        Phase 1:
        min h
        s.t. -h*x_ip + sum(lambda_j * x_ij) >= 0, i=1,...,m
             sum(lambda_j * y_rj) >= y_rp, r=1,...,s
             lambda_j >= 0
        
        Phase 2:
        max sum(s_i^-) + sum(s_r^+)
        s.t. sum(lambda_j * x_ij) + s_i^- = h*x_ip, i=1,...,m
             sum(lambda_j * y_rj) - s_r^+ = y_rp, r=1,...,s
             lambda_j >= 0, s_i^- >= 0, s_r^+ >= 0
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation (0-based)
        
        Returns:
        --------
        efficiency : float
            Efficiency score from Phase 1 (h*)
        lambdas : np.ndarray
            Optimal intensity variables from Phase 2
        input_slacks : np.ndarray
            Optimal input slacks from Phase 2
        output_slacks : np.ndarray
            Optimal output slacks from Phase 2
        sum_slacks : float
            Sum of optimal slacks from Phase 2
        """
        from .ccr import CCRModel
        
        # Phase 1: Solve CCR envelopment model
        ccr = CCRModel(self.inputs, self.outputs)
        efficiency, _, _, _ = ccr.solve_envelopment(dmu_index)
        
        # Phase 2: Maximize slacks given efficiency from Phase 1
        n_vars2 = self.n_dmus + self.n_inputs + self.n_outputs
        c2 = np.zeros(n_vars2)
        c2[self.n_dmus:self.n_dmus + self.n_inputs] = 1.0  # input slacks
        c2[self.n_dmus + self.n_inputs:] = 1.0  # output slacks
        
        n_constraints2 = self.n_inputs + self.n_outputs
        A2_eq = np.zeros((n_constraints2, n_vars2))
        
        # Input constraints: sum(lambda_j * x_ij) + s_i^- = h*x_ip
        for i in range(self.n_inputs):
            A2_eq[i, :self.n_dmus] = self.inputs[:, i]
            A2_eq[i, self.n_dmus + i] = 1.0
        
        # Output constraints: sum(lambda_j * y_rj) - s_r^+ = y_rp
        for r in range(self.n_outputs):
            A2_eq[self.n_inputs + r, :self.n_dmus] = self.outputs[:, r]
            A2_eq[self.n_inputs + r, self.n_dmus + self.n_inputs + r] = -1.0
        
        b2_eq = np.zeros(n_constraints2)
        for i in range(self.n_inputs):
            b2_eq[i] = efficiency * self.inputs[dmu_index, i]
        for r in range(self.n_outputs):
            b2_eq[self.n_inputs + r] = self.outputs[dmu_index, r]
        
        bounds2 = [(0, None)] * n_vars2
        
        result2 = linprog(-c2, A_eq=A2_eq, b_eq=b2_eq, bounds=bounds2, method='highs')
        
        if not result2.success:
            raise RuntimeError(f"Phase 2 optimization failed for DMU {dmu_index}: {result2.message}")
        
        lambdas = result2.x[:self.n_dmus]
        input_slacks = result2.x[self.n_dmus:self.n_dmus + self.n_inputs]
        output_slacks = result2.x[self.n_dmus + self.n_inputs:]
        sum_slacks = -result2.fun
        
        return efficiency, lambdas, input_slacks, output_slacks, sum_slacks
    
    def evaluate_all_ccr(self) -> pd.DataFrame:
        """
        Evaluate all DMUs using Two-Phase CCR model
        
        Returns:
        --------
        results : pd.DataFrame
            DataFrame with results
        """
        results = []
        
        for j in range(self.n_dmus):
            eff, lambdas, input_slacks, output_slacks, sum_slacks = self.solve_ccr(j)
            
            result_dict = {
                'DMU': j + 1,
                'Efficiency': eff,
                'Sum_Slacks': sum_slacks
            }
            for i, lam in enumerate(lambdas):
                result_dict[f'Lambda_{i+1}'] = lam
            for i in range(self.n_inputs):
                result_dict[f'S-_{i+1}'] = input_slacks[i]
            for r in range(self.n_outputs):
                result_dict[f'S+_{r+1}'] = output_slacks[r]
            
            results.append(result_dict)
        
        return pd.DataFrame(results)

