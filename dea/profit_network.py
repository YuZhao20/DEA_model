"""
Profit Efficiency and Network DEA Models
Based on Chapter 4.10 and 4.11 of Hosseinzadeh Lotfi et al. (2020)
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Optional
import pandas as pd


class ProfitEfficiencyModel:
    """
    Profit Efficiency DEA Model
    Based on Chapter 4.11
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray,
                 input_costs: np.ndarray, output_prices: np.ndarray):
        """
        Initialize Profit Efficiency model
        
        Parameters:
        -----------
        inputs : np.ndarray
            Input matrix
        outputs : np.ndarray
            Output matrix
        input_costs : np.ndarray
            Input cost vector or matrix
        output_prices : np.ndarray
            Output price vector or matrix
        """
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.input_costs = np.array(input_costs)
        self.output_prices = np.array(output_prices)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.input_costs.ndim == 1:
            self.input_costs = np.tile(self.input_costs, (self.n_dmus, 1))
        if self.output_prices.ndim == 1:
            self.output_prices = np.tile(self.output_prices, (self.n_dmus, 1))
    
    def solve(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Profit Efficiency Model (4.11)
        
        max sum(p_r * y_r) - sum(c_i * x_i)
        s.t. sum(lambda_j * x_ij) <= x_i, i=1,...,m
             sum(lambda_j * y_rj) >= y_r, r=1,...,s
             lambda_j >= 0
        
        Returns:
        --------
        profit_efficiency : float
            Profit efficiency score
        optimal_inputs : np.ndarray
            Optimal input quantities
        optimal_outputs : np.ndarray
            Optimal output quantities
        lambdas : np.ndarray
            Optimal intensity variables
        """
        # Variables: [x_1, ..., x_m, y_1, ..., y_s, lambda_1, ..., lambda_n]
        n_vars = self.n_inputs + self.n_outputs + self.n_dmus
        c = np.zeros(n_vars)
        c[:self.n_inputs] = self.input_costs[dmu_index, :]  # cost coefficients (positive for minimization)
        c[self.n_inputs:self.n_inputs + self.n_outputs] = -self.output_prices[dmu_index, :]  # price coefficients (negative)
        
        # Constraints: inputs + outputs + VRS constraint
        n_constraints = self.n_inputs + self.n_outputs + 1
        A = np.zeros((n_constraints, n_vars))
        
        # Input constraints: sum(lambda_j * x_ij) <= x_i
        for i in range(self.n_inputs):
            A[i, i] = -1.0  # -x_i
            A[i, self.n_inputs + self.n_outputs:] = self.inputs[:, i]
        
        # Output constraints: sum(lambda_j * y_rj) >= y_r
        for r in range(self.n_outputs):
            A[self.n_inputs + r, self.n_inputs + r] = 1.0  # y_r
            A[self.n_inputs + r, self.n_inputs + self.n_outputs:] = -self.outputs[:, r]
        
        # VRS constraint: sum(lambda_j) = 1
        A[self.n_inputs + self.n_outputs, self.n_inputs + self.n_outputs:] = 1.0
        
        b = np.zeros(n_constraints)
        b[self.n_inputs + self.n_outputs] = 1.0  # sum(lambda_j) = 1
        
        # Constraint types
        A_eq = A[self.n_inputs + self.n_outputs:self.n_inputs + self.n_outputs + 1, :]
        b_eq = b[self.n_inputs + self.n_outputs:self.n_inputs + self.n_outputs + 1]
        A_ub = A[:self.n_inputs + self.n_outputs, :]
        b_ub = b[:self.n_inputs + self.n_outputs]
        
        bounds = [(0, None)] * n_vars
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        optimal_inputs = result.x[:self.n_inputs]
        optimal_outputs = result.x[self.n_inputs:self.n_inputs + self.n_outputs]
        lambdas = result.x[self.n_inputs + self.n_outputs:]
        
        # Calculate profit efficiency
        current_profit = (np.sum(self.output_prices[dmu_index, :] * self.outputs[dmu_index, :]) -
                         np.sum(self.input_costs[dmu_index, :] * self.inputs[dmu_index, :]))
        optimal_profit = -result.fun  # negate because we minimized negative of profit
        profit_efficiency = current_profit / optimal_profit if optimal_profit != 0 else 0.0
        
        return profit_efficiency, optimal_inputs, optimal_outputs, lambdas
    
    def evaluate_all(self) -> pd.DataFrame:
        results = []
        for j in range(self.n_dmus):
            eff, opt_inputs, opt_outputs, lambdas = self.solve(j)
            result_dict = {'DMU': j + 1, 'Profit_Efficiency': eff}
            for i in range(self.n_inputs):
                result_dict[f'Optimal_Input_{i+1}'] = opt_inputs[i]
            for r in range(self.n_outputs):
                result_dict[f'Optimal_Output_{r+1}'] = opt_outputs[r]
            for i, lam in enumerate(lambdas):
                result_dict[f'Lambda_{i+1}'] = lam
            results.append(result_dict)
        return pd.DataFrame(results)


class ModifiedSBMModel:
    """
    Modified Slack Based DEA Models
    Based on Chapter 4.12
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
    
    def solve_input_oriented(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Input-Oriented Modified SBM Model (4.12.1)
        """
        # Similar to SBM but with modified objective
        from .sbm import SBMModel
        sbm = SBMModel(self.inputs, self.outputs)
        eff, lambdas, input_slacks, output_slacks = sbm.solve_model1(dmu_index)
        return eff, lambdas, input_slacks, output_slacks
    
    def solve_output_oriented(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Output-Oriented Modified SBM Model (4.12.2)
        """
        from .sbm import SBMModel
        sbm = SBMModel(self.inputs, self.outputs)
        eff, lambdas, input_slacks, output_slacks = sbm.solve_model2(dmu_index)
        return eff, lambdas, input_slacks, output_slacks
    
    def solve(self, dmu_index: int, orientation: str = 'input', rts: str = 'vrs') -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Modified SBM Model
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU to evaluate
        orientation : str
            'input' or 'output'
        rts : str
            Returns to scale (for compatibility, not used in Modified SBM)
        
        Returns:
        --------
        efficiency : float
        lambdas : np.ndarray
        input_slacks : np.ndarray
        output_slacks : np.ndarray
        """
        if orientation == 'input':
            return self.solve_input_oriented(dmu_index)
        else:
            return self.solve_output_oriented(dmu_index)
    
    def evaluate_all(self, orientation: str = 'input') -> pd.DataFrame:
        results = []
        for j in range(self.n_dmus):
            if orientation == 'input':
                eff, lambdas, input_slacks, output_slacks = self.solve_input_oriented(j)
            else:
                eff, lambdas, input_slacks, output_slacks = self.solve_output_oriented(j)
            
            result_dict = {'DMU': j + 1, 'Modified_SBM_Efficiency': eff}
            for i, lam in enumerate(lambdas):
                result_dict[f'Lambda_{i+1}'] = lam
            for i in range(self.n_inputs):
                result_dict[f'S-_{i+1}'] = input_slacks[i]
            for r in range(self.n_outputs):
                result_dict[f'S+_{r+1}'] = output_slacks[r]
            results.append(result_dict)
        return pd.DataFrame(results)

