"""
Additive Min (mADD) Model
Based on Aparicio et al. (2007)
Finds closest targets on the efficient frontier
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Optional
import pandas as pd
from .additive import AdditiveModel


class AddMinModel:
    """
    Additive Min Model
    
    Finds the minimum distance to the Pareto-efficient frontier
    using weighted slacks.
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve(self, dmu_index: int, rts: str = 'vrs',
              weight_slack_i: Optional[np.ndarray] = None,
              weight_slack_o: Optional[np.ndarray] = None,
              orientation: Optional[str] = None) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Additive Min Model
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU to evaluate
        rts : str
            Returns to scale: 'crs', 'vrs', 'nirs', 'ndrs'
        weight_slack_i : np.ndarray, optional
            Weights for input slacks
        weight_slack_o : np.ndarray, optional
            Weights for output slacks
        orientation : str, optional
            'io' (input-oriented) or 'oo' (output-oriented)
        
        Returns:
        --------
        objval : float
            Objective value (sum of weighted slacks)
        lambdas : np.ndarray
            Optimal intensity variables
        slack_input : np.ndarray
            Input slacks
        slack_output : np.ndarray
            Output slacks
        target_input : np.ndarray
            Target input values
        target_output : np.ndarray
            Target output values
        """
        x_p = self.inputs[dmu_index, :]
        y_p = self.outputs[dmu_index, :]
        
        # Default weights
        if weight_slack_i is None:
            weight_slack_i = np.ones(self.n_inputs)
        elif np.isscalar(weight_slack_i):
            weight_slack_i = np.full(self.n_inputs, weight_slack_i)
        
        if weight_slack_o is None:
            weight_slack_o = np.ones(self.n_outputs)
        elif np.isscalar(weight_slack_o):
            weight_slack_o = np.full(self.n_outputs, weight_slack_o)
        
        # Orientation handling
        if orientation == 'io':
            weight_slack_o = np.zeros(self.n_outputs)
        elif orientation == 'oo':
            weight_slack_i = np.zeros(self.n_inputs)
        
        # Variables: [lambda_1, ..., lambda_n, s_1^-, ..., s_m^-, s_1^+, ..., s_s^+]
        n_vars = self.n_dmus + self.n_inputs + self.n_outputs
        c = np.zeros(n_vars)
        c[self.n_dmus:self.n_dmus + self.n_inputs] = weight_slack_i
        c[self.n_dmus + self.n_inputs:] = weight_slack_o
        
        # Constraints
        n_constraints = self.n_inputs + self.n_outputs
        A_eq = np.zeros((n_constraints, n_vars))
        b_eq = np.zeros(n_constraints)
        
        # Input constraints: sum(lambda_j * x_ij) + s_i^- = x_ip
        for i in range(self.n_inputs):
            A_eq[i, :self.n_dmus] = self.inputs[:, i]
            A_eq[i, self.n_dmus + i] = 1.0
            b_eq[i] = x_p[i]
        
        # Output constraints: sum(lambda_j * y_rj) - s_r^+ = y_rp
        for r in range(self.n_outputs):
            A_eq[self.n_inputs + r, :self.n_dmus] = self.outputs[:, r]
            A_eq[self.n_inputs + r, self.n_dmus + self.n_inputs + r] = -1.0
            b_eq[self.n_inputs + r] = y_p[r]
        
        # RTS constraints
        if rts == 'vrs':
            A_eq = np.vstack([A_eq, np.zeros((1, n_vars))])
            A_eq[-1, :self.n_dmus] = 1.0
            b_eq = np.append(b_eq, 1.0)
        elif rts == 'nirs':
            A_ub = np.zeros((1, n_vars))
            A_ub[0, :self.n_dmus] = 1.0
            b_ub = np.array([1.0])
        elif rts == 'ndrs':
            A_ub = np.zeros((1, n_vars))
            A_ub[0, :self.n_dmus] = -1.0
            b_ub = np.array([-1.0])
        else:  # crs
            A_ub = None
            b_ub = None
        
        bounds = [(0, None)] * n_vars
        
        if rts == 'crs' or rts == 'vrs':
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        else:
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                            bounds=bounds, method='highs')
        
        if not result.success:
            return (np.nan, np.full(self.n_dmus, np.nan),
                   np.full(self.n_inputs, np.nan), np.full(self.n_outputs, np.nan),
                   np.full(self.n_inputs, np.nan), np.full(self.n_outputs, np.nan))
        
        objval = result.fun
        lambdas = result.x[:self.n_dmus]
        slack_input = result.x[self.n_dmus:self.n_dmus + self.n_inputs]
        slack_output = result.x[self.n_dmus + self.n_inputs:]
        
        target_input = self.inputs.T @ lambdas
        target_output = self.outputs.T @ lambdas

        return objval, lambdas, slack_input, slack_output, target_input, target_output

    def evaluate_all(self, rts: str = 'vrs', orientation: str = None) -> pd.DataFrame:
        """
        Evaluate all DMUs using AddMin model

        Parameters:
        -----------
        rts : str
            Returns to scale: 'crs', 'vrs', 'nirs', 'ndrs'
        orientation : str, optional
            'io' (input-oriented) or 'oo' (output-oriented)

        Returns:
        --------
        pd.DataFrame
            DataFrame with efficiency scores for all DMUs
        """
        results = []
        for j in range(self.n_dmus):
            objval, lambdas, slack_input, slack_output, target_input, target_output = \
                self.solve(j, rts=rts, orientation=orientation)

            result_dict = {
                'DMU': j + 1,
                'Objective': objval
            }
            for i in range(self.n_inputs):
                result_dict[f'Slack_Input_{i+1}'] = slack_input[i]
            for r in range(self.n_outputs):
                result_dict[f'Slack_Output_{r+1}'] = slack_output[r]
            results.append(result_dict)

        return pd.DataFrame(results)

