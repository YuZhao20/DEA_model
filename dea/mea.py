"""
Multi-directional Efficiency Analysis (MEA)
Based on Benchmarking package
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Optional
import pandas as pd


class MEAModel:
    """
    Multi-directional Efficiency Analysis (MEA)
    
    MEA calculates potential improvements in each direction (input or output)
    by finding the minimum step to the frontier for each dimension.
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, rts: str = 'vrs'):
        """
        Initialize MEA model
        
        Parameters:
        -----------
        inputs : np.ndarray
            Input matrix
        outputs : np.ndarray
            Output matrix
        rts : str
            Returns to scale: 'crs', 'vrs', 'drs', 'irs', 'fdh'
        """
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.rts = rts.lower()
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def _min_direction(self, dmu_index: int, orientation: str = 'in') -> np.ndarray:
        """
        Calculate minimum direction for each input/output
        
        For input orientation: minimum step for each input
        For output orientation: maximum step for each output
        """
        x_p = self.inputs[dmu_index, :]
        y_p = self.outputs[dmu_index, :]
        
        if orientation == 'in':
            md = self.n_inputs
            directions = np.zeros(md)
            
            for h in range(md):
                # Find minimum step for input h
                # Variables: [beta, lambda_1, ..., lambda_n]
                n_vars = self.n_dmus + 1
                c = np.zeros(n_vars)
                c[0] = 1.0  # minimize beta
                
                n_constraints = self.n_inputs + self.n_outputs
                A = np.zeros((n_constraints, n_vars))
                
                # Input constraints
                for i in range(self.n_inputs):
                    if i == h:
                        A[i, 0] = -x_p[i]  # -beta * x_ph
                    A[i, 1:] = self.inputs[:, i]
                
                # Output constraints
                for r in range(self.n_outputs):
                    A[self.n_inputs + r, 1:] = -self.outputs[:, r]
                
                b = np.zeros(n_constraints)
                for r in range(self.n_outputs):
                    b[self.n_inputs + r] = -y_p[r]
                
                # RTS constraint
                if self.rts == 'vrs':
                    n_constraints += 1
                    A_new = np.zeros((n_constraints, n_vars))
                    A_new[:self.n_inputs + self.n_outputs, :] = A
                    A_new[self.n_inputs + self.n_outputs, 1:] = 1.0
                    A = A_new
                    b = np.append(b, 1.0)
                
                bounds = [(0, None)] * n_vars
                
                result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
                
                if result.success:
                    directions[h] = result.x[0]
                else:
                    directions[h] = np.inf
                    
        elif orientation == 'out':
            md = self.n_outputs
            directions = np.zeros(md)
            
            for h in range(md):
                # Find maximum step for output h
                n_vars = self.n_dmus + 1
                c = np.zeros(n_vars)
                c[0] = -1.0  # maximize beta (minimize -beta)
                
                n_constraints = self.n_inputs + self.n_outputs
                A = np.zeros((n_constraints, n_vars))
                
                # Input constraints
                for i in range(self.n_inputs):
                    A[i, 1:] = self.inputs[:, i]
                
                # Output constraints
                for r in range(self.n_outputs):
                    if r == h:
                        A[self.n_inputs + r, 0] = -y_p[r]  # -beta * y_ph
                    A[self.n_inputs + r, 1:] = -self.outputs[:, r]
                
                b = np.zeros(n_constraints)
                for i in range(self.n_inputs):
                    b[i] = x_p[i]
                
                if self.rts == 'vrs':
                    n_constraints += 1
                    A_new = np.zeros((n_constraints, n_vars))
                    A_new[:self.n_inputs + self.n_outputs, :] = A
                    A_new[self.n_inputs + self.n_outputs, 1:] = 1.0
                    A = A_new
                    b = np.append(b, 1.0)
                
                bounds = [(0, None)] * n_vars
                
                result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
                
                if result.success:
                    directions[h] = -result.x[0]  # negate because we minimized negative
                else:
                    directions[h] = -np.inf
        else:
            raise ValueError("Orientation must be 'in' or 'out'")
        
        return directions
    
    def solve(self, dmu_index: int, orientation: str = 'in') -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Solve MEA for a DMU
        
        Returns:
        --------
        efficiency : float
            Overall efficiency score
        directions : np.ndarray
            Direction vector (potential improvements)
        lambdas : np.ndarray
            Optimal intensity variables
        """
        directions = self._min_direction(dmu_index, orientation)
        
        # Use directional efficiency with calculated directions
        from .advanced import DirectionalEfficiencyModel
        
        if orientation == 'in':
            gx = directions
            gy = None
        else:
            gx = None
            gy = directions
        
        dea_model = DirectionalEfficiencyModel(self.inputs, self.outputs)
        eff, lambdas = dea_model.solve(dmu_index, gx=gx, gy=gy)
        
        return eff, directions, lambdas
    
    def evaluate_all(self, orientation: str = 'in') -> pd.DataFrame:
        results = []
        for j in range(self.n_dmus):
            eff, directions, lambdas = self.solve(j, orientation)
            
            result_dict = {'DMU': j + 1, 'Efficiency': eff}
            if orientation == 'in':
                for i in range(self.n_inputs):
                    result_dict[f'Direction_Input_{i+1}'] = directions[i]
            else:
                for r in range(self.n_outputs):
                    result_dict[f'Direction_Output_{r+1}'] = directions[r]
            for i, lam in enumerate(lambdas):
                result_dict[f'Lambda_{i+1}'] = lam
            results.append(result_dict)
        return pd.DataFrame(results)

