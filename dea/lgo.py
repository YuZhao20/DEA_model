"""
Linear Goal-Oriented (LGO) DEA Model
Based on deaR package model_lgo
Generalized oriented DEA model with linear constraints
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Optional
import pandas as pd


class LGOModel:
    """
    Linear Goal-Oriented DEA Model
    
    Solves linear generalized oriented DEA models with customizable
    input and output orientation parameters.
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve(self, dmu_index: int, d_input: Optional[np.ndarray] = None,
              d_output: Optional[np.ndarray] = None, rts: str = 'vrs',
              maxslack: bool = True, weight_slack_i: Optional[np.ndarray] = None,
              weight_slack_o: Optional[np.ndarray] = None,
              L: float = 1.0, U: float = 1.0) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Linear Goal-Oriented DEA Model
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU to evaluate
        d_input : np.ndarray, optional
            Input orientation parameters (default: 1 for all inputs)
        d_output : np.ndarray, optional
            Output orientation parameters (default: 1 for all outputs)
        rts : str
            Returns to scale: 'crs', 'vrs', 'nirs', 'ndrs', 'grs'
        maxslack : bool
            If True, compute max slack solution in second stage
        weight_slack_i : np.ndarray, optional
            Weights for input slacks
        weight_slack_o : np.ndarray, optional
            Weights for output slacks
        L : float
            Lower bound for GRS
        U : float
            Upper bound for GRS
        
        Returns:
        --------
        rho : float
            Efficiency score
        beta : float
            Directional distance parameter
        lambdas : np.ndarray
            Optimal intensity variables
        target_input : np.ndarray
            Target input values
        target_output : np.ndarray
            Target output values
        slack_input : np.ndarray
            Input slacks
        slack_output : np.ndarray
            Output slacks
        effproj_input : np.ndarray
            Efficient projection inputs
        effproj_output : np.ndarray
            Efficient projection outputs
        """
        x_p = self.inputs[dmu_index, :]
        y_p = self.outputs[dmu_index, :]
        
        # Default orientation parameters
        if d_input is None:
            d_input = np.ones(self.n_inputs)
        elif np.isscalar(d_input):
            d_input = np.full(self.n_inputs, d_input)
        else:
            d_input = np.array(d_input)
            if len(d_input) != self.n_inputs:
                raise ValueError("d_input length must match number of inputs")
        
        if d_output is None:
            d_output = np.ones(self.n_outputs)
        elif np.isscalar(d_output):
            d_output = np.full(self.n_outputs, d_output)
        else:
            d_output = np.array(d_output)
            if len(d_output) != self.n_outputs:
                raise ValueError("d_output length must match number of outputs")
        
        if np.any(d_input < 0) or np.any(d_output < 0):
            raise ValueError("Orientation parameters must be non-negative")
        
        # Directional vectors
        dir_input = d_input * x_p
        dir_output = d_output * y_p
        
        # Stage 1: Maximize beta
        # Variables: [beta, lambda_1, ..., lambda_n]
        n_vars = 1 + self.n_dmus
        c = np.zeros(n_vars)
        c[0] = -1.0  # maximize beta (negative for minimization)
        
        # Constraints
        n_constraints = self.n_inputs + self.n_outputs
        A_ub = np.zeros((n_constraints, n_vars))
        b_ub = np.zeros(n_constraints)
        
        # Input constraints: beta*dir_input[i] + sum(lambda_j * x_ij) <= x_ip
        for i in range(self.n_inputs):
            A_ub[i, 0] = dir_input[i]
            A_ub[i, 1:] = self.inputs[:, i]
            b_ub[i] = x_p[i]
        
        # Output constraints: -beta*dir_output[r] + sum(lambda_j * y_rj) >= y_rp
        # For linprog: beta*dir_output[r] - sum(lambda_j * y_rj) <= -y_rp
        for r in range(self.n_outputs):
            A_ub[self.n_inputs + r, 0] = -dir_output[r]
            A_ub[self.n_inputs + r, 1:] = -self.outputs[:, r]
            b_ub[self.n_inputs + r] = -y_p[r]
        
        # RTS constraints
        A_eq = None
        b_eq = None
        if rts == 'vrs':
            A_eq = np.zeros((1, n_vars))
            A_eq[0, 1:] = 1.0
            b_eq = np.array([1.0])
        elif rts == 'nirs':
            A_ub = np.vstack([A_ub, np.zeros((1, n_vars))])
            A_ub[-1, 1:] = 1.0
            b_ub = np.append(b_ub, 1.0)
        elif rts == 'ndrs':
            A_ub = np.vstack([A_ub, np.zeros((1, n_vars))])
            A_ub[-1, 1:] = -1.0
            b_ub = np.append(b_ub, -1.0)
        elif rts == 'grs':
            A_ub = np.vstack([A_ub, np.zeros((2, n_vars))])
            A_ub[-2, 1:] = 1.0
            A_ub[-1, 1:] = -1.0
            b_ub = np.append(b_ub, [U, -L])
        
        bounds = [(0, None)] * n_vars
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')
        
        if not result.success:
            return (np.nan, np.nan, np.full(self.n_dmus, np.nan),
                   np.full(self.n_inputs, np.nan), np.full(self.n_outputs, np.nan),
                   np.full(self.n_inputs, np.nan), np.full(self.n_outputs, np.nan),
                   np.full(self.n_inputs, np.nan), np.full(self.n_outputs, np.nan))
        
        beta = result.x[0]
        lambda_stage1 = result.x[1:]
        
        # Calculate efficiency rho
        rho = (1 - np.sum(dir_input / x_p) * beta / self.n_inputs) / \
              (1 + beta * np.sum(dir_output / y_p) / self.n_outputs)
        
        if maxslack:
            # Stage 2: Maximize slacks
            if weight_slack_i is None:
                weight_slack_i = np.ones(self.n_inputs)
            elif np.isscalar(weight_slack_i):
                weight_slack_i = np.full(self.n_inputs, weight_slack_i)
            
            if weight_slack_o is None:
                weight_slack_o = np.ones(self.n_outputs)
            elif np.isscalar(weight_slack_o):
                weight_slack_o = np.full(self.n_outputs, weight_slack_o)
            
            # Variables: [lambda_1, ..., lambda_n, s_1^-, ..., s_m^-, s_1^+, ..., s_s^+]
            n_vars2 = self.n_dmus + self.n_inputs + self.n_outputs
            c2 = np.zeros(n_vars2)
            c2[self.n_dmus:self.n_dmus + self.n_inputs] = -weight_slack_i
            c2[self.n_dmus + self.n_inputs:] = -weight_slack_o
            
            # Constraints
            n_constraints2 = self.n_inputs + self.n_outputs
            A_eq2 = np.zeros((n_constraints2, n_vars2))
            b_eq2 = np.zeros(n_constraints2)
            
            # Input constraints: sum(lambda_j * x_ij) + s_i^- = x_ip - beta*dir_input[i]
            for i in range(self.n_inputs):
                A_eq2[i, :self.n_dmus] = self.inputs[:, i]
                A_eq2[i, self.n_dmus + i] = 1.0
                b_eq2[i] = x_p[i] - beta * dir_input[i]
            
            # Output constraints: sum(lambda_j * y_rj) - s_r^+ = y_rp + beta*dir_output[r]
            for r in range(self.n_outputs):
                A_eq2[self.n_inputs + r, :self.n_dmus] = self.outputs[:, r]
                A_eq2[self.n_inputs + r, self.n_dmus + self.n_inputs + r] = -1.0
                b_eq2[self.n_inputs + r] = y_p[r] + beta * dir_output[r]
            
            # RTS constraints
            A_ub2 = None
            b_ub2 = None
            if rts == 'vrs':
                A_eq2 = np.vstack([A_eq2, np.zeros((1, n_vars2))])
                A_eq2[-1, :self.n_dmus] = 1.0
                b_eq2 = np.append(b_eq2, 1.0)
            elif rts == 'nirs':
                A_ub2 = np.zeros((1, n_vars2))
                A_ub2[0, :self.n_dmus] = 1.0
                b_ub2 = np.array([1.0])
            elif rts == 'ndrs':
                A_ub2 = np.zeros((1, n_vars2))
                A_ub2[0, :self.n_dmus] = -1.0
                b_ub2 = np.array([-1.0])
            elif rts == 'grs':
                A_ub2 = np.zeros((2, n_vars2))
                A_ub2[0, :self.n_dmus] = 1.0
                A_ub2[1, :self.n_dmus] = -1.0
                b_ub2 = np.array([U, -L])
            
            bounds2 = [(0, None)] * n_vars2
            
            result2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2,
                             bounds=bounds2, method='highs')
            
            if result2.success:
                lambdas = result2.x[:self.n_dmus]
                slack_input = result2.x[self.n_dmus:self.n_dmus + self.n_inputs]
                slack_output = result2.x[self.n_dmus + self.n_inputs:]
                effproj_input = self.inputs.T @ lambdas
                effproj_output = self.outputs.T @ lambdas
            else:
                lambdas = lambda_stage1
                slack_input = np.zeros(self.n_inputs)
                slack_output = np.zeros(self.n_outputs)
                effproj_input = self.inputs.T @ lambdas
                effproj_output = self.outputs.T @ lambdas
        else:
            lambdas = lambda_stage1
            effproj_input = self.inputs.T @ lambdas
            effproj_output = self.outputs.T @ lambdas
            slack_input = x_p - beta * dir_input - effproj_input
            slack_output = effproj_output - y_p - beta * dir_output
        
        target_input = effproj_input + slack_input
        target_output = effproj_output - slack_output

        return (rho, beta, lambdas, target_input, target_output,
               slack_input, slack_output, effproj_input, effproj_output)

    def evaluate_all(self, d_input: np.ndarray = None, d_output: np.ndarray = None,
                     rts: str = 'vrs') -> pd.DataFrame:
        """
        Evaluate all DMUs using LGO model

        Parameters:
        -----------
        d_input : np.ndarray, optional
            Input orientation parameters
        d_output : np.ndarray, optional
            Output orientation parameters
        rts : str
            Returns to scale: 'crs', 'vrs', 'nirs', 'ndrs', 'grs'

        Returns:
        --------
        pd.DataFrame
            DataFrame with efficiency scores for all DMUs
        """
        results = []
        for j in range(self.n_dmus):
            rho, beta, lambdas, target_input, target_output, \
                slack_input, slack_output, effproj_input, effproj_output = \
                self.solve(j, d_input, d_output, rts)

            result_dict = {
                'DMU': j + 1,
                'Efficiency': rho,
                'Beta': beta
            }
            for i, lam in enumerate(lambdas):
                result_dict[f'Lambda_{i+1}'] = lam
            results.append(result_dict)

        return pd.DataFrame(results)

