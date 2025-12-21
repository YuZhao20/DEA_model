"""
Additive Super-Efficiency Model
Based on Du, Liang and Zhu (2010)
Extension of SBM super-efficiency to additive DEA model
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Optional
import pandas as pd


class AddSuperEffModel:
    """
    Additive Super-Efficiency Model
    
    Evaluates efficient DMUs by excluding them from the reference set.
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve(self, dmu_index: int, rts: str = 'crs',
              weight_slack_i: Optional[np.ndarray] = None,
              weight_slack_o: Optional[np.ndarray] = None,
              orientation: Optional[str] = None,
              L: float = 1.0, U: float = 1.0) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Additive Super-Efficiency Model
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU to evaluate
        rts : str
            Returns to scale: 'crs', 'vrs', 'nirs', 'ndrs', 'grs'
        weight_slack_i : np.ndarray, optional
            Weights for input super-slacks (default: 1/x_ip)
        weight_slack_o : np.ndarray, optional
            Weights for output super-slacks (default: 1/y_rp)
        orientation : str, optional
            'io' (input-oriented) or 'oo' (output-oriented)
        L : float
            Lower bound for GRS
        U : float
            Upper bound for GRS
        
        Returns:
        --------
        delta : float
            Super-efficiency score
        objval : float
            Objective value
        lambdas : np.ndarray
            Optimal intensity variables (excluding evaluated DMU)
        t_input : np.ndarray
            Input super-slacks
        t_output : np.ndarray
            Output super-slacks
        target_input : np.ndarray
            Target input values
        target_output : np.ndarray
            Target output values
        """
        x_p = self.inputs[dmu_index, :]
        y_p = self.outputs[dmu_index, :]
        
        # Handle zero inputs/outputs (Case 2 from Tone 2001)
        nzimin = np.array([np.min(self.inputs[self.inputs[:, i] > 0, i]) 
                          if np.any(self.inputs[:, i] > 0) else 1.0 
                          for i in range(self.n_inputs)]) / 100
        nzomin = np.array([np.min(self.outputs[self.outputs[:, r] > 0, r]) 
                          if np.any(self.outputs[:, r] > 0) else 1.0 
                          for r in range(self.n_outputs)]) / 100
        
        x_p_adj = x_p.copy()
        y_p_adj = y_p.copy()
        for i in range(self.n_inputs):
            if x_p[i] == 0:
                x_p_adj[i] = nzimin[i]
        for r in range(self.n_outputs):
            if y_p[r] == 0:
                y_p_adj[r] = nzomin[r]
        
        # Default weights (unit invariant)
        if weight_slack_i is None:
            weight_slack_i = 1.0 / x_p_adj / (self.n_inputs + self.n_outputs)
        elif np.isscalar(weight_slack_i):
            weight_slack_i = np.full(self.n_inputs, weight_slack_i)
        
        if weight_slack_o is None:
            weight_slack_o = 1.0 / y_p_adj / (self.n_inputs + self.n_outputs)
        elif np.isscalar(weight_slack_o):
            weight_slack_o = np.full(self.n_outputs, weight_slack_o)
        
        # Orientation handling
        if orientation == 'io':
            weight_slack_o = np.zeros(self.n_outputs)
        elif orientation == 'oo':
            weight_slack_i = np.zeros(self.n_inputs)
        
        # Reference set excludes evaluated DMU
        ref_indices = [j for j in range(self.n_dmus) if j != dmu_index]
        n_ref = len(ref_indices)
        
        if n_ref == 0:
            return (np.nan, np.nan, np.array([]), np.full(self.n_inputs, np.nan),
                   np.full(self.n_outputs, np.nan), np.full(self.n_inputs, np.nan),
                   np.full(self.n_outputs, np.nan))
        
        input_ref = self.inputs[ref_indices, :].T
        output_ref = self.outputs[ref_indices, :].T
        
        # Variables: [lambda_1, ..., lambda_n_ref, t_1, ..., t_m, t_1^+, ..., t_s^+]
        n_vars = n_ref + self.n_inputs + self.n_outputs
        c = np.zeros(n_vars)
        c[n_ref:n_ref + self.n_inputs] = weight_slack_i
        c[n_ref + self.n_inputs:] = weight_slack_o
        
        # Constraints
        n_constraints = self.n_inputs + self.n_outputs + 1
        A_ub = np.zeros((n_constraints, n_vars))
        b_ub = np.zeros(n_constraints)
        
        # Input constraints: sum(lambda_j * x_ij) - t_i <= x_ip
        for i in range(self.n_inputs):
            A_ub[i, :n_ref] = input_ref[i, :]
            A_ub[i, n_ref + i] = -1.0
            b_ub[i] = x_p_adj[i]
        
        # Output constraints: sum(lambda_j * y_rj) + t_r^+ >= y_rp
        # For linprog: -sum(lambda_j * y_rj) - t_r^+ <= -y_rp
        for r in range(self.n_outputs):
            A_ub[self.n_inputs + r, :n_ref] = -output_ref[r, :]
            A_ub[self.n_inputs + r, n_ref + self.n_inputs + r] = -1.0
            b_ub[self.n_inputs + r] = -y_p_adj[r]
        
        # Constraint: sum(lambda_j) = 1 (for VRS) or <= 1 (NIRS) or >= 1 (NDRS)
        if rts == 'vrs':
            A_eq = np.zeros((1, n_vars))
            A_eq[0, :n_ref] = 1.0
            b_eq = np.array([1.0])
        elif rts == 'nirs':
            A_ub = np.vstack([A_ub, np.zeros((1, n_vars))])
            A_ub[-1, :n_ref] = 1.0
            b_ub = np.append(b_ub, 1.0)
            A_eq = None
            b_eq = None
        elif rts == 'ndrs':
            A_ub = np.vstack([A_ub, np.zeros((1, n_vars))])
            A_ub[-1, :n_ref] = -1.0
            b_ub = np.append(b_ub, -1.0)
            A_eq = None
            b_eq = None
        elif rts == 'grs':
            A_ub = np.vstack([A_ub, np.zeros((2, n_vars))])
            A_ub[-2, :n_ref] = 1.0
            A_ub[-1, :n_ref] = -1.0
            b_ub = np.append(b_ub, [U, -L])
            A_eq = None
            b_eq = None
        else:  # crs
            A_eq = None
            b_eq = None
        
        # Zero weight constraints: if weight is 0, corresponding slack must be 0
        w0i = np.where(weight_slack_i == 0)[0]
        w0o = np.where(weight_slack_o == 0)[0]
        
        if len(w0i) > 0 or len(w0o) > 0:
            n_w0 = len(w0i) + len(w0o)
            A_eq_w0 = np.zeros((n_w0, n_vars))
            b_eq_w0 = np.zeros(n_w0)
            row = 0
            for i in w0i:
                A_eq_w0[row, n_ref + i] = 1.0
                row += 1
            for r in w0o:
                A_eq_w0[row, n_ref + self.n_inputs + r] = 1.0
                row += 1
            
            if A_eq is not None:
                A_eq = np.vstack([A_eq, A_eq_w0])
                b_eq = np.append(b_eq, b_eq_w0)
            else:
                A_eq = A_eq_w0
                b_eq = b_eq_w0
        
        bounds = [(0, None)] * n_vars
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')
        
        if not result.success:
            return (np.nan, np.nan, np.full(n_ref, np.nan),
                   np.full(self.n_inputs, np.nan), np.full(self.n_outputs, np.nan),
                   np.full(self.n_inputs, np.nan), np.full(self.n_outputs, np.nan))
        
        objval = result.fun
        lambdas_full = np.zeros(self.n_dmus)
        lambdas_full[ref_indices] = result.x[:n_ref]
        t_input = result.x[n_ref:n_ref + self.n_inputs]
        t_output = result.x[n_ref + self.n_inputs:]
        
        # Calculate delta
        delta_num = 1 + np.sum(t_input / x_p_adj) / self.n_inputs
        delta_den = 1 - np.sum(t_output / y_p_adj) / self.n_outputs
        delta = delta_num / delta_den if delta_den > 0 else np.inf
        
        target_input = input_ref @ result.x[:n_ref]
        target_output = output_ref @ result.x[:n_ref]

        return delta, objval, lambdas_full, t_input, t_output, target_input, target_output

    def evaluate_all(self, rts: str = 'crs', orientation: str = None) -> pd.DataFrame:
        """
        Evaluate all DMUs using Additive Super-Efficiency model

        Parameters:
        -----------
        rts : str
            Returns to scale: 'crs', 'vrs', 'nirs', 'ndrs', 'grs'
        orientation : str, optional
            'io' (input-oriented) or 'oo' (output-oriented)

        Returns:
        --------
        pd.DataFrame
            DataFrame with super-efficiency scores for all DMUs
        """
        results = []
        for j in range(self.n_dmus):
            delta, objval, lambdas, t_input, t_output, target_input, target_output = \
                self.solve(j, rts=rts, orientation=orientation)

            result_dict = {
                'DMU': j + 1,
                'Super_Efficiency': delta,
                'Objective': objval
            }
            results.append(result_dict)

        return pd.DataFrame(results)

