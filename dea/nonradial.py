"""
Non-Radial DEA Model
Based on deaR package model_nonradial
Allows for non-proportional reductions in each input or augmentations in each output
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Optional
import pandas as pd


class NonRadialModel:
    """
    Non-Radial DEA Model
    
    This model allows for non-proportional reductions in each input or 
    augmentations in each output, unlike radial models that assume proportional changes.
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve(self, dmu_index: int, orientation: str = 'io', rts: str = 'vrs',
              maxslack: bool = True, weight_slack: Optional[np.ndarray] = None,
              L: float = 1.0, U: float = 1.0) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Non-Radial DEA Model
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU to evaluate
        orientation : str
            'io' for input-oriented, 'oo' for output-oriented
        rts : str
            Returns to scale: 'crs', 'vrs', 'nirs', 'ndrs', 'grs'
        maxslack : bool
            If True, compute max slack solution in second stage
        weight_slack : np.ndarray, optional
            Weights for slacks in max slack solution
        L : float
            Lower bound for GRS (default 1.0)
        U : float
            Upper bound for GRS (default 1.0)
        
        Returns:
        --------
        mean_efficiency : float
            Mean efficiency score (average of individual efficiencies)
        efficiency : np.ndarray
            Individual efficiency scores for each input (io) or output (oo)
        lambdas : np.ndarray
            Optimal intensity variables
        slack_output : np.ndarray (io) or slack_input : np.ndarray (oo)
            Output slacks (io) or input slacks (oo)
        target_input : np.ndarray
            Target input values
        target_output : np.ndarray
            Target output values
        """
        if orientation == 'io':
            return self._solve_input_oriented(dmu_index, rts, maxslack, weight_slack, L, U)
        else:
            return self._solve_output_oriented(dmu_index, rts, maxslack, weight_slack, L, U)
    
    def _solve_input_oriented(self, dmu_index: int, rts: str, maxslack: bool,
                             weight_slack: Optional[np.ndarray], L: float, U: float):
        """Solve input-oriented non-radial model"""
        x_p = self.inputs[dmu_index, :]
        y_p = self.outputs[dmu_index, :]
        
        # Stage 1: Minimize mean efficiency
        # Variables: [theta_1, ..., theta_m, lambda_1, ..., lambda_n]
        n_vars = self.n_inputs + self.n_dmus
        c = np.zeros(n_vars)
        # Objective: minimize mean of theta_i (excluding non-controllable)
        c[:self.n_inputs] = 1.0 / self.n_inputs
        
        # Constraints
        n_constraints = self.n_inputs + self.n_outputs + self.n_inputs
        A_ub = np.zeros((n_constraints, n_vars))
        b_ub = np.zeros(n_constraints)

        # Input constraints: sum(lambda_j * x_ij) <= theta_i * x_ip
        # For linprog: sum(lambda_j * x_ij) - theta_i * x_ip <= 0
        for i in range(self.n_inputs):
            A_ub[i, i] = -x_p[i]  # -theta_i * x_ip
            A_ub[i, self.n_inputs:] = self.inputs[:, i]  # sum(lambda_j * x_ij)

        # Output constraints: sum(lambda_j * y_rj) >= y_rp
        # For linprog: -sum(lambda_j * y_rj) <= -y_rp
        for r in range(self.n_outputs):
            A_ub[self.n_inputs + r, self.n_inputs:] = -self.outputs[:, r]
            b_ub[self.n_inputs + r] = -y_p[r]

        # Theta constraints: theta_i <= 1
        for i in range(self.n_inputs):
            A_ub[self.n_inputs + self.n_outputs + i, i] = 1.0
            b_ub[self.n_inputs + self.n_outputs + i] = 1.0
        
        # RTS constraints
        A_eq = None
        b_eq = None
        if rts == 'vrs':
            A_eq = np.zeros((1, n_vars))
            A_eq[0, self.n_inputs:] = 1.0
            b_eq = np.array([1.0])
        elif rts == 'nirs':
            A_ub = np.vstack([A_ub, np.zeros((1, n_vars))])
            A_ub[-1, self.n_inputs:] = 1.0
            b_ub = np.append(b_ub, 1.0)
        elif rts == 'ndrs':
            A_ub = np.vstack([A_ub, np.zeros((1, n_vars))])
            A_ub[-1, self.n_inputs:] = -1.0
            b_ub = np.append(b_ub, -1.0)
        elif rts == 'grs':
            A_ub = np.vstack([A_ub, np.zeros((2, n_vars))])
            A_ub[-2, self.n_inputs:] = 1.0
            A_ub[-1, self.n_inputs:] = -1.0
            b_ub = np.append(b_ub, [U, -L])
        
        bounds = [(0, None)] * n_vars
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')
        
        if not result.success:
            return (np.nan, np.full(self.n_inputs, np.nan), 
                   np.full(self.n_dmus, np.nan), np.full(self.n_outputs, np.nan),
                   np.full(self.n_inputs, np.nan), np.full(self.n_outputs, np.nan))
        
        theta = result.x[:self.n_inputs]
        lambda_stage1 = result.x[self.n_inputs:]
        mean_eff = np.mean(theta)
        
        if maxslack:
            # Stage 2: Maximize slacks
            if weight_slack is None:
                weight_slack = np.ones(self.n_outputs)
            elif weight_slack.ndim == 0:
                weight_slack = np.full(self.n_outputs, weight_slack)
            
            # Variables: [lambda_1, ..., lambda_n, s_1^+, ..., s_s^+]
            n_vars2 = self.n_dmus + self.n_outputs
            c2 = np.zeros(n_vars2)
            c2[self.n_dmus:] = -weight_slack  # maximize slacks (negative for minimization)
            
            # Constraints
            n_constraints2 = self.n_inputs + self.n_outputs
            A_eq2 = np.zeros((n_constraints2, n_vars2))
            b_eq2 = np.zeros(n_constraints2)
            
            # Input constraints: sum(lambda_j * x_ij) = theta_i * x_ip
            for i in range(self.n_inputs):
                A_eq2[i, :self.n_dmus] = self.inputs[:, i]
                b_eq2[i] = theta[i] * x_p[i]
            
            # Output constraints: sum(lambda_j * y_rj) - s_r^+ = y_rp
            for r in range(self.n_outputs):
                A_eq2[self.n_inputs + r, :self.n_dmus] = self.outputs[:, r]
                A_eq2[self.n_inputs + r, self.n_dmus + r] = -1.0
                b_eq2[self.n_inputs + r] = y_p[r]
            
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
                slack_output = result2.x[self.n_dmus:]
                target_input = self.inputs.T @ lambdas
                target_output = self.outputs.T @ lambdas
            else:
                lambdas = lambda_stage1
                slack_output = np.zeros(self.n_outputs)
                target_input = self.inputs.T @ lambdas
                target_output = self.outputs.T @ lambdas
        else:
            lambdas = lambda_stage1
            slack_output = np.zeros(self.n_outputs)
            target_input = self.inputs.T @ lambdas
            target_output = self.outputs.T @ lambdas
        
        return (mean_eff, theta, lambdas, slack_output, target_input, target_output)
    
    def _solve_output_oriented(self, dmu_index: int, rts: str, maxslack: bool,
                              weight_slack: Optional[np.ndarray], L: float, U: float):
        """Solve output-oriented non-radial model"""
        x_p = self.inputs[dmu_index, :]
        y_p = self.outputs[dmu_index, :]
        
        # Stage 1: Maximize mean efficiency
        # Variables: [phi_1, ..., phi_s, lambda_1, ..., lambda_n]
        n_vars = self.n_outputs + self.n_dmus
        c = np.zeros(n_vars)
        # Objective: maximize mean of phi_r (negative for minimization)
        c[:self.n_outputs] = -1.0 / self.n_outputs
        
        # Constraints
        n_constraints = self.n_inputs + self.n_outputs + self.n_outputs
        A_ub = np.zeros((n_constraints, n_vars))
        b_ub = np.zeros(n_constraints)
        
        # Input constraints: sum(lambda_j * x_ij) <= x_ip
        # For linprog: sum(lambda_j * x_ij) <= x_ip
        for i in range(self.n_inputs):
            A_ub[i, self.n_outputs:] = self.inputs[:, i]
            b_ub[i] = x_p[i]
        
        # Output constraints: -phi_r * y_rp + sum(lambda_j * y_rj) >= 0
        # For linprog: phi_r * y_rp - sum(lambda_j * y_rj) <= 0
        for r in range(self.n_outputs):
            A_ub[self.n_inputs + r, r] = y_p[r]
            A_ub[self.n_inputs + r, self.n_outputs:] = -self.outputs[:, r]
        
        # Phi constraints: phi_r >= 1
        # For linprog: -phi_r <= -1
        for r in range(self.n_outputs):
            A_ub[self.n_inputs + self.n_outputs + r, r] = -1.0
            b_ub[self.n_inputs + self.n_outputs + r] = -1.0
        
        # RTS constraints
        A_eq = None
        b_eq = None
        if rts == 'vrs':
            A_eq = np.zeros((1, n_vars))
            A_eq[0, self.n_outputs:] = 1.0
            b_eq = np.array([1.0])
        elif rts == 'nirs':
            A_ub = np.vstack([A_ub, np.zeros((1, n_vars))])
            A_ub[-1, self.n_outputs:] = 1.0
            b_ub = np.append(b_ub, 1.0)
        elif rts == 'ndrs':
            A_ub = np.vstack([A_ub, np.zeros((1, n_vars))])
            A_ub[-1, self.n_outputs:] = -1.0
            b_ub = np.append(b_ub, -1.0)
        elif rts == 'grs':
            A_ub = np.vstack([A_ub, np.zeros((2, n_vars))])
            A_ub[-2, self.n_outputs:] = 1.0
            A_ub[-1, self.n_outputs:] = -1.0
            b_ub = np.append(b_ub, [U, -L])
        
        bounds = [(0, None)] * n_vars
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')
        
        if not result.success:
            return (np.nan, np.full(self.n_outputs, np.nan), 
                   np.full(self.n_dmus, np.nan), np.full(self.n_inputs, np.nan),
                   np.full(self.n_inputs, np.nan), np.full(self.n_outputs, np.nan))
        
        phi = result.x[:self.n_outputs]
        lambda_stage1 = result.x[self.n_outputs:]
        mean_eff = np.mean(phi)
        
        if maxslack:
            # Stage 2: Maximize slacks
            if weight_slack is None:
                weight_slack = np.ones(self.n_inputs)
            elif weight_slack.ndim == 0:
                weight_slack = np.full(self.n_inputs, weight_slack)
            
            # Variables: [lambda_1, ..., lambda_n, s_1^-, ..., s_m^-]
            n_vars2 = self.n_dmus + self.n_inputs
            c2 = np.zeros(n_vars2)
            c2[self.n_dmus:] = -weight_slack  # maximize slacks (negative for minimization)
            
            # Constraints
            n_constraints2 = self.n_inputs + self.n_outputs
            A_eq2 = np.zeros((n_constraints2, n_vars2))
            b_eq2 = np.zeros(n_constraints2)
            
            # Input constraints: sum(lambda_j * x_ij) + s_i^- = x_ip
            for i in range(self.n_inputs):
                A_eq2[i, :self.n_dmus] = self.inputs[:, i]
                A_eq2[i, self.n_dmus + i] = 1.0
                b_eq2[i] = x_p[i]
            
            # Output constraints: sum(lambda_j * y_rj) = phi_r * y_rp
            for r in range(self.n_outputs):
                A_eq2[self.n_inputs + r, :self.n_dmus] = self.outputs[:, r]
                b_eq2[self.n_inputs + r] = phi[r] * y_p[r]
            
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
                slack_input = result2.x[self.n_dmus:]
                target_input = self.inputs.T @ lambdas
                target_output = self.outputs.T @ lambdas
            else:
                lambdas = lambda_stage1
                slack_input = np.zeros(self.n_inputs)
                target_input = self.inputs.T @ lambdas
                target_output = self.outputs.T @ lambdas
        else:
            lambdas = lambda_stage1
            slack_input = np.zeros(self.n_inputs)
            target_input = self.inputs.T @ lambdas
            target_output = self.outputs.T @ lambdas
        
        return (mean_eff, phi, lambdas, slack_input, target_input, target_output)
    
    def evaluate_all(self, orientation: str = 'io', rts: str = 'vrs',
                    maxslack: bool = True) -> pd.DataFrame:
        """Evaluate all DMUs"""
        results = []
        for i in range(self.n_dmus):
            mean_eff, eff, lambdas, slack, target_in, target_out = self.solve(
                i, orientation, rts, maxslack
            )
            results.append({
                'DMU': i,
                'Mean_Efficiency': mean_eff,
                'Efficiency': eff,
                'Lambdas': lambdas,
                'Slack': slack,
                'Target_Input': target_in,
                'Target_Output': target_out
            })
        return pd.DataFrame(results)

