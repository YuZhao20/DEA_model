"""
Slacks-Based Measure (SBM) DEA Models
Based on Chapter 4.9 of Hosseinzadeh Lotfi et al. (2020)
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple
import pandas as pd


class SBMModel:
    """
    Slacks-Based Measure (SBM) DEA Model
    Based on Chapter 4.9
    """

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]

        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")

    def solve_model1(self, dmu_index: int, rts: str = 'vrs') -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve First Model of SBM (4.9.1) - Input-oriented
        
        Linearized SBM model using Charnes-Cooper transformation
        Variables: [t, Lambda_1, ..., Lambda_n, S_1^-, ..., S_m^-, S_1^+, ..., S_s^+]
        where Lambda = t * lambda, S^- = t * s^-, S^+ = t * s^+
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation (0-based)
        rts : str
            Returns to scale: 'crs', 'vrs', 'drs', 'irs' (default: 'vrs')
        """
        n_vars = 1 + self.n_dmus + self.n_inputs + self.n_outputs
        c = np.zeros(n_vars)
        c[0] = 1.0

        n_eq_constraints = 1 + self.n_inputs + self.n_outputs
        if rts == 'vrs':
            n_eq_constraints += 1

        A_eq = np.zeros((n_eq_constraints, n_vars))
        A_ub_list = []

        row = 0
        A_eq[row, 0] = 1.0
        for r in range(self.n_outputs):
            if self.outputs[dmu_index, r] > 1e-10:
                A_eq[row, 1 + self.n_dmus + self.n_inputs + r] = 1.0 / (self.n_outputs * self.outputs[dmu_index, r])
        row += 1

        for i in range(self.n_inputs):
            A_eq[row, 0] = -self.inputs[dmu_index, i]
            A_eq[row, 1:1 + self.n_dmus] = self.inputs[:, i]
            A_eq[row, 1 + self.n_dmus + i] = 1.0
            row += 1

        for r in range(self.n_outputs):
            A_eq[row, 1:1 + self.n_dmus] = self.outputs[:, r]
            A_eq[row, 1 + self.n_dmus + self.n_inputs + r] = -1.0
            A_eq[row, 0] = -self.outputs[dmu_index, r]
            row += 1

        if rts == 'vrs':
            A_eq[row, 1:1 + self.n_dmus] = 1.0
            A_eq[row, 0] = -1.0
            row += 1
        elif rts == 'drs':
            row_ub = np.zeros(n_vars)
            row_ub[1:1 + self.n_dmus] = 1.0
            row_ub[0] = -1.0
            A_ub_list.append(row_ub)
        elif rts == 'irs':
            row_ub = np.zeros(n_vars)
            row_ub[1:1 + self.n_dmus] = -1.0
            row_ub[0] = 1.0
            A_ub_list.append(row_ub)

        b_eq = np.zeros(n_eq_constraints)
        b_eq[0] = 1.0
        if rts == 'vrs':
            b_eq[-1] = 0.0

        if len(A_ub_list) > 0:
            A_ub = np.array(A_ub_list)
            b_ub = np.zeros(len(A_ub_list))
        else:
            A_ub = None
            b_ub = None

        bounds = [(0, None)] * n_vars

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                        bounds=bounds, method='highs')

        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")

        t = result.x[0]
        Lambda = result.x[1:1 + self.n_dmus]
        S_minus = result.x[1 + self.n_dmus:1 + self.n_dmus + self.n_inputs]
        S_plus = result.x[1 + self.n_dmus + self.n_inputs:]

        if t > 1e-10:
            lambdas = Lambda / t
            input_slacks = S_minus / t
            output_slacks = S_plus / t
        else:
            lambdas = np.zeros(self.n_dmus)
            input_slacks = np.zeros(self.n_inputs)
            output_slacks = np.zeros(self.n_outputs)

        sbm_eff = max(0.0, min(1.0, t))

        return sbm_eff, lambdas, input_slacks, output_slacks

    def solve_model2(self, dmu_index: int, rts: str = 'vrs') -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Second Model of SBM (4.9.2) - Output-oriented
        
        Linearized SBM model using Charnes-Cooper transformation
        Variables: [t, Lambda_1, ..., Lambda_n, S_1^-, ..., S_m^-, S_1^+, ..., S_s^+]
        where Lambda = t * lambda, S^- = t * s^-, S^+ = t * s^+
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation (0-based)
        rts : str
            Returns to scale: 'crs', 'vrs', 'drs', 'irs' (default: 'vrs')
        """
        n_vars = 1 + self.n_dmus + self.n_inputs + self.n_outputs
        c = np.zeros(n_vars)
        c[0] = -1.0

        n_eq_constraints = 1 + self.n_inputs + self.n_outputs
        if rts == 'vrs':
            n_eq_constraints += 1

        A_eq = np.zeros((n_eq_constraints, n_vars))
        A_ub_list = []

        row = 0
        A_eq[row, 0] = 1.0
        for i in range(self.n_inputs):
            if self.inputs[dmu_index, i] > 1e-10:
                A_eq[row, 1 + self.n_dmus + i] = -1.0 / (self.n_inputs * self.inputs[dmu_index, i])
        row += 1

        for i in range(self.n_inputs):
            A_eq[row, 0] = -self.inputs[dmu_index, i]
            A_eq[row, 1:1 + self.n_dmus] = self.inputs[:, i]
            A_eq[row, 1 + self.n_dmus + i] = 1.0
            row += 1

        for r in range(self.n_outputs):
            A_eq[row, 1:1 + self.n_dmus] = self.outputs[:, r]
            A_eq[row, 1 + self.n_dmus + self.n_inputs + r] = -1.0
            A_eq[row, 0] = -self.outputs[dmu_index, r]
            row += 1

        if rts == 'vrs':
            A_eq[row, 1:1 + self.n_dmus] = 1.0
            A_eq[row, 0] = -1.0
            row += 1
        elif rts == 'drs':
            row_ub = np.zeros(n_vars)
            row_ub[1:1 + self.n_dmus] = 1.0
            row_ub[0] = -1.0
            A_ub_list.append(row_ub)
        elif rts == 'irs':
            row_ub = np.zeros(n_vars)
            row_ub[1:1 + self.n_dmus] = -1.0
            row_ub[0] = 1.0
            A_ub_list.append(row_ub)

        b_eq = np.zeros(n_eq_constraints)
        b_eq[0] = 1.0
        if rts == 'vrs':
            b_eq[-1] = 0.0

        if len(A_ub_list) > 0:
            A_ub = np.array(A_ub_list)
            b_ub = np.zeros(len(A_ub_list))
        else:
            A_ub = None
            b_ub = None

        bounds = [(0, None)] * n_vars

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                        bounds=bounds, method='highs')

        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")

        t = result.x[0]
        Lambda = result.x[1:1 + self.n_dmus]
        S_minus = result.x[1 + self.n_dmus:1 + self.n_dmus + self.n_inputs]
        S_plus = result.x[1 + self.n_dmus + self.n_inputs:]

        if t > 1e-10:
            lambdas = Lambda / t
            input_slacks = S_minus / t
            output_slacks = S_plus / t
        else:
            lambdas = np.zeros(self.n_dmus)
            input_slacks = np.zeros(self.n_inputs)
            output_slacks = np.zeros(self.n_outputs)

        if t > 1e-10:
            sbm_eff = max(0.0, min(1.0, 1.0 / t))
        else:
            sbm_eff = 0.0

        return sbm_eff, lambdas, input_slacks, output_slacks

    def evaluate_all(self, model_type: int = 1, rts: str = 'vrs') -> pd.DataFrame:
        """
        Evaluate all DMUs
        
        Parameters:
        -----------
        model_type : int
            1 for first model, 2 for second model
        rts : str
            Returns to scale: 'crs', 'vrs', 'drs', 'irs' (default: 'vrs')
        """
        results = []
        for j in range(self.n_dmus):
            try:
                if model_type == 1:
                    eff, lambdas, input_slacks, output_slacks = self.solve_model1(j, rts=rts)
                else:
                    eff, lambdas, input_slacks, output_slacks = self.solve_model2(j, rts=rts)

                result_dict = {'DMU': j + 1, 'SBM_Efficiency': eff}
                for i, lam in enumerate(lambdas):
                    result_dict[f'Lambda_{i+1}'] = lam
                for i in range(self.n_inputs):
                    result_dict[f'S-_{i+1}'] = input_slacks[i]
                for r in range(self.n_outputs):
                    result_dict[f'S+_{r+1}'] = output_slacks[r]

                results.append(result_dict)
            except Exception as e:
                result_dict = {'DMU': j + 1, 'SBM_Efficiency': 0.0}
                for i in range(self.n_dmus):
                    result_dict[f'Lambda_{i+1}'] = 0.0
                for i in range(self.n_inputs):
                    result_dict[f'S-_{i+1}'] = 0.0
                for r in range(self.n_outputs):
                    result_dict[f'S+_{r+1}'] = 0.0
                results.append(result_dict)

        return pd.DataFrame(results)
