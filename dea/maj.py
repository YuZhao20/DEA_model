"""
MAJ Super-Efficiency Model
Based on Chapter 4.3 of Hosseinzadeh Lotfi et al. (2020)
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple
import pandas as pd


class MAJModel:
    """
    MAJ (Mehrabian-Aghayi-Jahanshahloo) Super-Efficiency Model
    
    This model uses slack variables to measure super-efficiency.
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        """
        Initialize MAJ model
        
        Parameters:
        -----------
        inputs : np.ndarray
            Input matrix of shape (n_dmus, n_inputs)
        outputs : np.ndarray
            Output matrix of shape (n_dmus, n_outputs)
        """
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve(self, dmu_index: int) -> Tuple[float, float]:
        """
        Solve MAJ Super-Efficiency Model (4.9)
        
        min 1 + w+ - w-
        s.t. sum(lambda_j * x_ij) - w+ + w- <= x_ip, i=1,...,m, j!=p
             sum(lambda_j * y_rj) >= y_rp, r=1,...,s, j!=p
             lambda_j >= 0, j!=p
             w+ >= 0, w- >= 0
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation (0-based)
        
        Returns:
        --------
        w_star : float
            Optimal value of w+ - w-
        super_efficiency : float
            Super-efficiency score (1 + w*)
        """
        # Objective: minimize 1 + w+ - w-
        # Variables: [lambda_1, ..., lambda_{p-1}, lambda_{p+1}, ..., lambda_n, w+, w-]
        n_lambdas = self.n_dmus - 1  # exclude DMU p
        n_vars = n_lambdas + 2  # lambdas + w+ + w-
        c = np.zeros(n_vars)
        c[n_lambdas] = 1.0  # w+
        c[n_lambdas + 1] = -1.0  # -w-
        
        # Constraints matrix
        n_constraints = self.n_inputs + self.n_outputs
        A = np.zeros((n_constraints, n_vars))
        
        # Create index mapping (exclude DMU p)
        dmu_indices = [j for j in range(self.n_dmus) if j != dmu_index]
        
        # Input constraints: sum(lambda_j * x_ij) - w+ + w- <= x_ip, j!=p
        for i in range(self.n_inputs):
            for idx, j in enumerate(dmu_indices):
                A[i, idx] = self.inputs[j, i]
            A[i, n_lambdas] = -1.0  # -w+
            A[i, n_lambdas + 1] = 1.0  # +w-
        
        # Output constraints: sum(lambda_j * y_rj) >= y_rp, j!=p
        for r in range(self.n_outputs):
            for idx, j in enumerate(dmu_indices):
                A[self.n_inputs + r, idx] = -self.outputs[j, r]
            A[self.n_inputs + r, n_lambdas] = 0.0
            A[self.n_inputs + r, n_lambdas + 1] = 0.0
        
        # Right-hand side
        b = np.zeros(n_constraints)
        for i in range(self.n_inputs):
            b[i] = self.inputs[dmu_index, i]
        for r in range(self.n_outputs):
            b[self.n_inputs + r] = -self.outputs[dmu_index, r]
        
        # Bounds: all variables >= 0
        bounds = [(0, None)] * n_vars
        
        # Solve linear program
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        w_star = result.fun  # w+ - w-
        super_efficiency = 1.0 + w_star
        
        return w_star, super_efficiency
    
    def evaluate_all(self) -> pd.DataFrame:
        """
        Evaluate all DMUs
        
        Returns:
        --------
        results : pd.DataFrame
            DataFrame with W* and Super_Efficiency scores
        """
        results = []
        
        for j in range(self.n_dmus):
            w_star, super_eff = self.solve(j)
            results.append({
                'DMU': j + 1,
                'W*': w_star,
                'Super_Efficiency_MAJ': super_eff
            })
        
        return pd.DataFrame(results)

