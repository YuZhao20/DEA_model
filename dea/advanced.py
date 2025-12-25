"""
Advanced DEA Models from Chapter 4
Including: Directional Distance Function (DDF)
Based on Chapter 4 of Hosseinzadeh Lotfi et al. (2020)
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Optional
import pandas as pd


class DirectionalEfficiencyModel:
    """
    Directional Distance Function (DDF) Model
    
    Reference: Chambers, R. G., Chung, Y., & Färe, R. (1996). 
    Benefit and distance functions. Journal of Economic Theory, 70(2), 407-419.
    
    The model measures efficiency in a specified direction (g_x, g_y), allowing
    simultaneous reduction of inputs and expansion of outputs.
    
    Standard formulation:
        max β
        s.t. Σλ_j x_ij ≤ x_ip - β g_xi,  i=1,...,m
             Σλ_j y_rj ≥ y_rp + β g_yr,  r=1,...,s
             Σλ_j = 1 (VRS)
             λ_j ≥ 0, β ≥ 0
    
    Where:
        β: inefficiency measure (0 = efficient)
        g_x: input direction vector (reduction direction)
        g_y: output direction vector (expansion direction)
    """

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        """
        Initialize the Directional Distance Function model.
        
        Parameters:
        -----------
        inputs : np.ndarray
            Input matrix (n_dmus × n_inputs)
        outputs : np.ndarray
            Output matrix (n_dmus × n_outputs)
        """
        self.inputs = np.array(inputs, dtype=float)
        self.outputs = np.array(outputs, dtype=float)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]

        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")

    def solve(self, dmu_index: int,
              gx: Optional[np.ndarray] = None,
              gy: Optional[np.ndarray] = None,
              rts: str = 'vrs') -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Directional Distance Function for a single DMU.
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation (0-based)
        gx : np.ndarray, optional
            Input direction vector. 
            Default: gx = x_p (proportional to DMU's inputs)
        gy : np.ndarray, optional
            Output direction vector.
            Default: gy = y_p (proportional to DMU's outputs)
        rts : str
            Returns to scale: 'crs', 'vrs', 'drs', 'irs' (default: 'vrs')
        
        Returns:
        --------
        beta : float
            Directional inefficiency score (0 = efficient, >0 = inefficient)
        lambdas : np.ndarray
            Optimal intensity variables
        input_slacks : np.ndarray
            Input slacks
        output_slacks : np.ndarray
            Output slacks
        """
        x_p = self.inputs[dmu_index, :]
        y_p = self.outputs[dmu_index, :]

        if gx is None:
            gx = x_p.copy()
        else:
            gx = np.array(gx, dtype=float).flatten()
            if len(gx) != self.n_inputs:
                raise ValueError(f"gx must have {self.n_inputs} elements")

        if gy is None:
            gy = y_p.copy()
        else:
            gy = np.array(gy, dtype=float).flatten()
            if len(gy) != self.n_outputs:
                raise ValueError(f"gy must have {self.n_outputs} elements")

        gx = np.where(np.abs(gx) < 1e-10, 1e-6, np.abs(gx))
        gy = np.where(np.abs(gy) < 1e-10, 1e-6, np.abs(gy))

        n_vars = 1 + self.n_dmus + self.n_inputs + self.n_outputs

        c = np.zeros(n_vars)
        c[0] = -1.0

        A_ub_list = []
        b_ub_list = []
        A_eq_list = []
        b_eq_list = []

        for i in range(self.n_inputs):
            row = np.zeros(n_vars)
            row[0] = gx[i]
            row[1:1 + self.n_dmus] = self.inputs[:, i]
            row[1 + self.n_dmus + i] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(x_p[i])

        for r in range(self.n_outputs):
            row = np.zeros(n_vars)
            row[0] = -gy[r]
            row[1:1 + self.n_dmus] = -self.outputs[:, r]
            row[1 + self.n_dmus + self.n_inputs + r] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(-y_p[r])

        if rts == 'vrs':
            row = np.zeros(n_vars)
            row[1:1 + self.n_dmus] = 1.0
            A_eq_list.append(row)
            b_eq_list.append(1.0)
        elif rts == 'drs':
            row = np.zeros(n_vars)
            row[1:1 + self.n_dmus] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(1.0)
        elif rts == 'irs':
            row = np.zeros(n_vars)
            row[1:1 + self.n_dmus] = -1.0
            A_ub_list.append(row)
            b_ub_list.append(-1.0)

        A_ub = np.array(A_ub_list) if A_ub_list else None
        b_ub = np.array(b_ub_list) if b_ub_list else None
        A_eq = np.array(A_eq_list) if A_eq_list else None
        b_eq = np.array(b_eq_list) if b_eq_list else None

        bounds = [(0, None)] * n_vars

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')

        if not result.success:
            bounds[0] = (None, None)
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')

        if not result.success:
            return 0.0, np.zeros(self.n_dmus), np.zeros(self.n_inputs), np.zeros(self.n_outputs)

        beta = -result.fun
        lambdas = result.x[1:1 + self.n_dmus]
        input_slacks = result.x[1 + self.n_dmus:1 + self.n_dmus + self.n_inputs]
        output_slacks = result.x[1 + self.n_dmus + self.n_inputs:]

        return beta, lambdas, input_slacks, output_slacks

    def evaluate_all(self, gx: Optional[np.ndarray] = None,
                     gy: Optional[np.ndarray] = None,
                     rts: str = 'vrs') -> pd.DataFrame:
        """
        Evaluate all DMUs using the Directional Distance Function.
        
        Parameters:
        -----------
        gx : np.ndarray, optional
            Input direction vector (same for all DMUs if specified).
            If None, uses each DMU's own input values as direction.
        gy : np.ndarray, optional
            Output direction vector (same for all DMUs if specified).
            If None, uses each DMU's own output values as direction.
        rts : str
            Returns to scale: 'crs', 'vrs', 'drs', 'irs'
        
        Returns:
        --------
        pd.DataFrame
            Results with columns: DMU, Beta (inefficiency), Efficiency
        """
        results = []
        for j in range(self.n_dmus):
            gx_j = gx if gx is not None else None
            gy_j = gy if gy is not None else None

            beta, lambdas, input_slacks, output_slacks = self.solve(j, gx_j, gy_j, rts)

            efficiency = 1.0 / (1.0 + max(beta, 0))

            result_dict = {
                'DMU': j + 1,
                'Beta': round(beta, 6),
                'Efficiency': round(efficiency, 6)
            }

            for i in range(self.n_inputs):
                result_dict[f'Input_Slack_{i+1}'] = round(input_slacks[i], 6)
            for r in range(self.n_outputs):
                result_dict[f'Output_Slack_{r+1}'] = round(output_slacks[r], 6)

            results.append(result_dict)

        return pd.DataFrame(results)
