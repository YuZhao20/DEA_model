"""
Cost and Revenue Efficiency DEA Models
Based on Chapter 4.6 and 4.7 of Hosseinzadeh Lotfi et al. (2020)
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple
import pandas as pd


class CostEfficiencyModel:
    """
    Cost Efficiency DEA Model
    Based on Chapter 4.6
    """

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, input_costs: np.ndarray):
        """
        Initialize Cost Efficiency model
        
        Parameters:
        -----------
        inputs : np.ndarray
            Input matrix of shape (n_dmus, n_inputs)
        outputs : np.ndarray
            Output matrix of shape (n_dmus, n_outputs)
        input_costs : np.ndarray
            Input cost vector of shape (n_inputs,) or matrix (n_dmus, n_inputs)
        """
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.input_costs = np.array(input_costs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]

        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")

        if self.input_costs.ndim == 1:
            self.input_costs = np.tile(self.input_costs, (self.n_dmus, 1))

    def solve(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Solve Cost Efficiency Model (4.6)
        
        min sum(c_i * x_i)
        s.t. sum(lambda_j * x_ij) <= x_i, i=1,...,m
             sum(lambda_j * y_rj) >= y_rp, r=1,...,s
             lambda_j >= 0
        
        Returns:
        --------
        cost_efficiency : float
            Cost efficiency score
        optimal_inputs : np.ndarray
            Optimal input quantities
        lambdas : np.ndarray
            Optimal intensity variables
        """
        n_vars = self.n_inputs + self.n_dmus
        c = np.zeros(n_vars)
        c[:self.n_inputs] = self.input_costs[dmu_index, :]

        n_constraints = self.n_inputs + self.n_outputs
        A = np.zeros((n_constraints, n_vars))

        for i in range(self.n_inputs):
            A[i, i] = -1.0
            A[i, self.n_inputs:] = self.inputs[:, i]

        for r in range(self.n_outputs):
            A[self.n_inputs + r, self.n_inputs:] = -self.outputs[:, r]

        b = np.zeros(n_constraints)
        for r in range(self.n_outputs):
            b[self.n_inputs + r] = -self.outputs[dmu_index, r]

        bounds = [(0, None)] * n_vars

        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")

        optimal_inputs = result.x[:self.n_inputs]
        lambdas = result.x[self.n_inputs:]

        current_cost = np.sum(self.input_costs[dmu_index, :] * self.inputs[dmu_index, :])
        optimal_cost = result.fun
        cost_efficiency = optimal_cost / current_cost if current_cost > 0 else 0.0

        return cost_efficiency, optimal_inputs, lambdas

    def evaluate_all(self) -> pd.DataFrame:
        results = []
        for j in range(self.n_dmus):
            eff, optimal_inputs, lambdas = self.solve(j)
            result_dict = {
                'DMU': j + 1,
                'Cost_Efficiency': eff
            }
            for i in range(self.n_inputs):
                result_dict[f'Optimal_Input_{i+1}'] = optimal_inputs[i]
            for i, lam in enumerate(lambdas):
                result_dict[f'Lambda_{i+1}'] = lam
            results.append(result_dict)
        return pd.DataFrame(results)


class RevenueEfficiencyModel:
    """
    Revenue Efficiency DEA Model
    Based on Chapter 4.7
    """

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, output_prices: np.ndarray):
        """
        Initialize Revenue Efficiency model
        
        Parameters:
        -----------
        inputs : np.ndarray
            Input matrix of shape (n_dmus, n_inputs)
        outputs : np.ndarray
            Output matrix of shape (n_dmus, n_outputs)
        output_prices : np.ndarray
            Output price vector of shape (n_outputs,) or matrix (n_dmus, n_outputs)
        """
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.output_prices = np.array(output_prices)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]

        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")

        if self.output_prices.ndim == 1:
            self.output_prices = np.tile(self.output_prices, (self.n_dmus, 1))

    def solve(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Solve Revenue Efficiency Model (4.7)
        
        max sum(p_r * y_r)
        s.t. sum(lambda_j * x_ij) <= x_ip, i=1,...,m
             sum(lambda_j * y_rj) >= y_r, r=1,...,s
             lambda_j >= 0
        
        Returns:
        --------
        revenue_efficiency : float
            Revenue efficiency score
        optimal_outputs : np.ndarray
            Optimal output quantities
        lambdas : np.ndarray
            Optimal intensity variables
        """
        n_vars = self.n_outputs + self.n_dmus
        c = np.zeros(n_vars)
        c[:self.n_outputs] = -self.output_prices[dmu_index, :]

        n_constraints = self.n_inputs + self.n_outputs
        A = np.zeros((n_constraints, n_vars))

        for i in range(self.n_inputs):
            A[i, self.n_outputs:] = self.inputs[:, i]

        for r in range(self.n_outputs):
            A[self.n_inputs + r, r] = 1.0
            A[self.n_inputs + r, self.n_outputs:] = -self.outputs[:, r]

        b = np.zeros(n_constraints)
        for i in range(self.n_inputs):
            b[i] = self.inputs[dmu_index, i]

        bounds = [(0, None)] * n_vars

        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")

        optimal_outputs = result.x[:self.n_outputs]
        lambdas = result.x[self.n_outputs:]

        current_revenue = np.sum(self.output_prices[dmu_index, :] * self.outputs[dmu_index, :])
        optimal_revenue = -result.fun
        revenue_efficiency = current_revenue / optimal_revenue if optimal_revenue > 0 else 0.0

        return revenue_efficiency, optimal_outputs, lambdas

    def evaluate_all(self) -> pd.DataFrame:
        results = []
        for j in range(self.n_dmus):
            eff, optimal_outputs, lambdas = self.solve(j)
            result_dict = {
                'DMU': j + 1,
                'Revenue_Efficiency': eff
            }
            for r in range(self.n_outputs):
                result_dict[f'Optimal_Output_{r+1}'] = optimal_outputs[r]
            for i, lam in enumerate(lambdas):
                result_dict[f'Lambda_{i+1}'] = lam
            results.append(result_dict)
        return pd.DataFrame(results)

