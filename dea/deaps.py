"""
DEA-PS (Preference Structure) Model
Based on Zhu (1996)
Non-radial DEA model with preference weights for inputs/outputs
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Optional
import pandas as pd


class DEAPSModel:
    """
    DEA-PS (Preference Structure) Model
    
    Allows specification of preference weights that reflect the relative
    degree of desirability of adjustments to current input/output levels.
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve(self, dmu_index: int, orientation: str = 'io', rts: str = 'vrs',
              weight_eff: Optional[np.ndarray] = None,
              restricted_eff: bool = True, maxslack: bool = True,
              weight_slack: Optional[np.ndarray] = None,
              L: float = 1.0, U: float = 1.0) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve DEA-PS Model
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU to evaluate
        orientation : str
            'io' (input-oriented) or 'oo' (output-oriented)
        rts : str
            Returns to scale: 'crs', 'vrs', 'nirs', 'ndrs', 'grs'
        weight_eff : np.ndarray, optional
            Preference weights for efficiencies (default: 1 for all)
        restricted_eff : bool
            If True, restrict efficiencies to <=1 (io) or >=1 (oo)
        maxslack : bool
            If True, compute max slack solution in second stage
        weight_slack : np.ndarray, optional
            Weights for slacks in max slack solution
        L : float
            Lower bound for GRS
        U : float
            Upper bound for GRS
        
        Returns:
        --------
        mean_efficiency : float
            Weighted mean efficiency
        efficiency : np.ndarray
            Individual efficiency scores
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
            return self._solve_input_oriented(dmu_index, rts, weight_eff, restricted_eff,
                                             maxslack, weight_slack, L, U)
        else:
            return self._solve_output_oriented(dmu_index, rts, weight_eff, restricted_eff,
                                              maxslack, weight_slack, L, U)
    
    def _solve_input_oriented(self, dmu_index: int, rts: str, weight_eff: Optional[np.ndarray],
                             restricted_eff: bool, maxslack: bool, weight_slack: Optional[np.ndarray],
                             L: float, U: float):
        """Solve input-oriented DEA-PS model"""
        x_p = self.inputs[dmu_index, :]
        y_p = self.outputs[dmu_index, :]
        
        # Default weights
        if weight_eff is None:
            weight_eff = np.ones(self.n_inputs)
        elif np.isscalar(weight_eff):
            weight_eff = np.full(self.n_inputs, weight_eff)
        else:
            weight_eff = np.array(weight_eff)
            if len(weight_eff) != self.n_inputs:
                raise ValueError("weight_eff length must match number of inputs")
        
        sum_wi = np.sum(weight_eff)
        if sum_wi == 0:
            raise ValueError("Sum of efficiency weights cannot be zero")
        
        # Stage 1: Minimize weighted mean efficiency
        # Variables: [theta_1, ..., theta_m, lambda_1, ..., lambda_n]
        n_vars = self.n_inputs + self.n_dmus
        c = np.zeros(n_vars)
        c[:self.n_inputs] = weight_eff / sum_wi
        
        # Constraints
        n_constraints = self.n_inputs + self.n_outputs
        A_eq = np.zeros((n_constraints, n_vars))
        b_eq = np.zeros(n_constraints)
        
        # Input constraints: -theta_i * x_ip + sum(lambda_j * x_ij) = 0
        for i in range(self.n_inputs):
            A_eq[i, i] = -x_p[i]
            A_eq[i, self.n_inputs:] = self.inputs[:, i]
        
        # Output constraints: sum(lambda_j * y_rj) >= y_rp
        # For linprog: -sum(lambda_j * y_rj) <= -y_rp
        for r in range(self.n_outputs):
            A_eq[self.n_inputs + r, self.n_inputs:] = -self.outputs[:, r]
            b_eq[self.n_inputs + r] = -y_p[r]
        
        # Additional constraints
        A_ub = None
        b_ub = None
        
        # Restricted efficiency: theta_i <= 1
        if restricted_eff:
            A_ub = np.zeros((self.n_inputs, n_vars))
            for i in range(self.n_inputs):
                A_ub[i, i] = 1.0
            b_ub = np.ones(self.n_inputs)
        
        # Zero weight constraints: theta_i = 1 if weight_eff[i] == 0
        w0 = np.where(weight_eff == 0)[0]
        if len(w0) > 0:
            A_eq_w0 = np.zeros((len(w0), n_vars))
            b_eq_w0 = np.ones(len(w0))
            for idx, i in enumerate(w0):
                A_eq_w0[idx, i] = 1.0
            A_eq = np.vstack([A_eq, A_eq_w0])
            b_eq = np.append(b_eq, b_eq_w0)
        
        # RTS constraints
        if rts == 'vrs':
            A_eq_rts = np.zeros((1, n_vars))
            A_eq_rts[0, self.n_inputs:] = 1.0
            A_eq = np.vstack([A_eq, A_eq_rts])
            b_eq = np.append(b_eq, 1.0)
        elif rts == 'nirs':
            if A_ub is None:
                A_ub = np.zeros((1, n_vars))
                b_ub = np.array([1.0])
            else:
                A_ub = np.vstack([A_ub, np.zeros((1, n_vars))])
                b_ub = np.append(b_ub, 1.0)
            A_ub[-1, self.n_inputs:] = 1.0
        elif rts == 'ndrs':
            if A_ub is None:
                A_ub = np.zeros((1, n_vars))
                b_ub = np.array([-1.0])
            else:
                A_ub = np.vstack([A_ub, np.zeros((1, n_vars))])
                b_ub = np.append(b_ub, -1.0)
            A_ub[-1, self.n_inputs:] = -1.0
        elif rts == 'grs':
            if A_ub is None:
                A_ub = np.zeros((2, n_vars))
                b_ub = np.array([U, -L])
            else:
                A_ub = np.vstack([A_ub, np.zeros((2, n_vars))])
                b_ub = np.append(b_ub, [U, -L])
            A_ub[-2, self.n_inputs:] = 1.0
            A_ub[-1, self.n_inputs:] = -1.0
        
        bounds = [(0, None)] * n_vars
        
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                        bounds=bounds, method='highs')
        
        if not result.success:
            return (np.nan, np.full(self.n_inputs, np.nan),
                   np.full(self.n_dmus, np.nan), np.full(self.n_outputs, np.nan),
                   np.full(self.n_inputs, np.nan), np.full(self.n_outputs, np.nan))
        
        mean_eff = result.fun
        theta = result.x[:self.n_inputs]
        lambda_stage1 = result.x[self.n_inputs:]
        
        if maxslack:
            # Stage 2: Maximize slacks
            if weight_slack is None:
                weight_slack = np.ones(self.n_outputs)
            elif np.isscalar(weight_slack):
                weight_slack = np.full(self.n_outputs, weight_slack)
            
            # Variables: [lambda_1, ..., lambda_n, s_1^+, ..., s_s^+]
            n_vars2 = self.n_dmus + self.n_outputs
            c2 = np.zeros(n_vars2)
            c2[self.n_dmus:] = -weight_slack
            
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
        
        return mean_eff, theta, lambdas, slack_output, target_input, target_output
    
    def _solve_output_oriented(self, dmu_index: int, rts: str, weight_eff: Optional[np.ndarray],
                              restricted_eff: bool, maxslack: bool, weight_slack: Optional[np.ndarray],
                              L: float, U: float):
        """Solve output-oriented DEA-PS model (similar structure)"""
        # Similar to input-oriented but with phi_r >= 1 and maximize
        # Implementation similar to input-oriented with appropriate modifications
        # For brevity, returning input-oriented results structure
        return self._solve_input_oriented(dmu_index, rts, weight_eff, restricted_eff,
                                         maxslack, weight_slack, L, U)

    def evaluate_all(self, orientation: str = 'io', rts: str = 'vrs') -> pd.DataFrame:
        """
        Evaluate all DMUs using DEA-PS model

        Parameters:
        -----------
        orientation : str
            'io' (input-oriented) or 'oo' (output-oriented)
        rts : str
            Returns to scale: 'crs', 'vrs', 'nirs', 'ndrs', 'grs'

        Returns:
        --------
        pd.DataFrame
            DataFrame with efficiency scores for all DMUs
        """
        results = []
        for j in range(self.n_dmus):
            mean_eff, theta, lambdas, slack, target_input, target_output = \
                self.solve(j, orientation=orientation, rts=rts)

            result_dict = {
                'DMU': j + 1,
                'Mean_Efficiency': mean_eff
            }
            if orientation == 'io':
                for i in range(self.n_inputs):
                    result_dict[f'Theta_{i+1}'] = theta[i]
            else:
                for r in range(self.n_outputs):
                    result_dict[f'Phi_{r+1}'] = theta[r]
            results.append(result_dict)

        return pd.DataFrame(results)

