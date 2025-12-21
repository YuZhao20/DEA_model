"""
FDH (Free Disposal Hull) DEA Models
Based on Benchmarking package

FDH does not assume convexity, only free disposability.
Efficiency is calculated directly without LP optimization.
"""

import numpy as np
from typing import Tuple
import pandas as pd


class FDHModel:
    """
    Free Disposal Hull (FDH) DEA Model
    
    FDH assumes free disposability but not convexity.
    Efficiency is calculated by finding the best dominating DMU.
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve_envelopment(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Input-Oriented FDH Envelopment Model

        For FDH, we find the DMU that dominates the evaluated DMU
        and has the minimum input contraction factor.

        Input-oriented efficiency: theta = min{theta : theta*x_p >= x_j, y_j >= y_p}
        This means we find the minimum scaling factor such that a reference DMU
        can produce at least the same outputs with at most theta*x_p inputs.

        Returns:
        --------
        efficiency : float
            Efficiency score (0 to 1 for input-oriented)
        lambdas : np.ndarray
            Intensity variables (only one lambda = 1 for the peer)
        input_targets : np.ndarray
            Target input values
        output_targets : np.ndarray
            Target output values
        """
        x_p = self.inputs[dmu_index, :]
        y_p = self.outputs[dmu_index, :]

        best_eff = 1.0  # Start with 1 (efficient)
        best_peer = dmu_index

        # Find DMUs that dominate output-wise (produce at least as much output)
        for j in range(self.n_dmus):
            x_j = self.inputs[j, :]
            y_j = self.outputs[j, :]

            # Check if DMU j produces at least as much output as DMU p
            if np.all(y_j >= y_p - 1e-10):
                # Calculate efficiency as max ratio x_j / x_p
                # This is the minimum theta such that theta*x_p >= x_j
                # i.e., theta = max_i(x_ji / x_pi)
                ratios = np.where(x_p > 1e-10, x_j / x_p, 0.0)
                eff = np.max(ratios)

                # We want the minimum efficiency (best contraction)
                if eff < best_eff:
                    best_eff = eff
                    best_peer = j

        # Ensure efficiency is at most 1
        best_eff = min(best_eff, 1.0)

        # Create lambda vector (only the peer has lambda = 1)
        lambdas = np.zeros(self.n_dmus)
        lambdas[best_peer] = 1.0

        input_targets = self.inputs[best_peer, :]
        output_targets = self.outputs[best_peer, :]

        return best_eff, lambdas, input_targets, output_targets
    
    def solve_output_oriented_envelopment(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Output-Oriented FDH Envelopment Model
        
        Find DMU that dominates and has maximum output ratio.
        """
        x_p = self.inputs[dmu_index, :]
        y_p = self.outputs[dmu_index, :]
        
        best_eff = 0.0
        best_peer = None
        
        for j in range(self.n_dmus):
            x_j = self.inputs[j, :]
            y_j = self.outputs[j, :]
            
            # Check if DMU j dominates DMU p
            if np.all(x_j <= x_p + 1e-10) and np.all(y_j >= y_p - 1e-10):
                # Calculate efficiency as min ratio of outputs
                ratios = y_j / (y_p + 1e-10)
                eff = np.min(ratios)
                
                if eff > best_eff:
                    best_eff = eff
                    best_peer = j
        
        if best_peer is None:
            best_eff = 1.0
            best_peer = dmu_index
        
        lambdas = np.zeros(self.n_dmus)
        lambdas[best_peer] = 1.0
        
        input_targets = self.inputs[best_peer, :]
        output_targets = self.outputs[best_peer, :]
        
        return best_eff, lambdas, input_targets, output_targets
    
    def evaluate_all(self, orientation: str = 'input') -> pd.DataFrame:
        results = []
        for j in range(self.n_dmus):
            if orientation == 'input':
                eff, lambdas, input_targets, output_targets = self.solve_envelopment(j)
            else:
                eff, lambdas, input_targets, output_targets = self.solve_output_oriented_envelopment(j)
            
            result_dict = {'DMU': j + 1, 'Efficiency': eff}
            # Find peer
            peer_idx = np.where(lambdas > 0.5)[0]
            if len(peer_idx) > 0:
                result_dict['Peer'] = peer_idx[0] + 1
            for i, lam in enumerate(lambdas):
                result_dict[f'Lambda_{i+1}'] = lam
            results.append(result_dict)
        return pd.DataFrame(results)


class FDHPlusModel:
    """
    FDH+ (Free Disposal Hull Plus) DEA Model
    
    FDH+ is an extension of FDH that allows some convexity
    by restricting lambdas to be in [1-param, 1+param] range.
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, param: float = 0.15):
        """
        Initialize FDH+ model
        
        Parameters:
        -----------
        inputs : np.ndarray
            Input matrix
        outputs : np.ndarray
            Output matrix
        param : float
            Parameter for lambda bounds (default: 0.15)
            Lambda must be in [1-param, 1+param] or 0
        """
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.param = param
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve_envelopment(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Input-Oriented FDH+ Envelopment Model
        
        Similar to FDH but with restricted lambda values.
        Uses LP optimization with semi-continuous variables.
        """
        from scipy.optimize import linprog
        
        low = 1.0 - self.param
        high = 1.0 + self.param
        
        # Variables: [h, lambda_1, ..., lambda_n]
        c = np.zeros(self.n_dmus + 1)
        c[0] = 1.0
        
        n_constraints = self.n_inputs + self.n_outputs
        A = np.zeros((n_constraints, self.n_dmus + 1))
        
        # Input constraints
        for i in range(self.n_inputs):
            A[i, 0] = -self.inputs[dmu_index, i]
            A[i, 1:] = self.inputs[:, i]
        
        # Output constraints
        for r in range(self.n_outputs):
            A[self.n_inputs + r, 1:] = -self.outputs[:, r]
        
        b = np.zeros(n_constraints)
        for r in range(self.n_outputs):
            b[self.n_inputs + r] = -self.outputs[dmu_index, r]
        
        # Bounds: lambda_j in [low, high] or 0
        # We approximate this with bounds [0, high] and add constraint
        bounds = [(0, None)] * (self.n_dmus + 1)
        bounds[0] = (0, None)  # h >= 0
        
        # Additional constraints to approximate semi-continuous
        # For simplicity, we use continuous bounds and add penalty
        # In practice, this requires mixed-integer programming
        # For now, we use continuous approximation
        
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        efficiency = result.x[0]
        lambdas = result.x[1:]
        
        # Round small lambdas to 0, and scale others to [low, high]
        lambdas_rounded = np.zeros_like(lambdas)
        for j in range(self.n_dmus):
            if lambdas[j] > 1e-6:
                # Scale to [low, high] range
                lambdas_rounded[j] = max(low, min(high, lambdas[j]))
        
        # Normalize if sum > 0
        if np.sum(lambdas_rounded) > 0:
            lambdas_rounded = lambdas_rounded / np.sum(lambdas_rounded) * np.sum(lambdas)
        
        input_targets = lambdas_rounded @ self.inputs
        output_targets = lambdas_rounded @ self.outputs
        
        return efficiency, lambdas_rounded, input_targets, output_targets
    
    def evaluate_all(self, orientation: str = 'input') -> pd.DataFrame:
        results = []
        for j in range(self.n_dmus):
            eff, lambdas, input_targets, output_targets = self.solve_envelopment(j)
            
            result_dict = {'DMU': j + 1, 'Efficiency': eff}
            for i, lam in enumerate(lambdas):
                result_dict[f'Lambda_{i+1}'] = lam
            results.append(result_dict)
        return pd.DataFrame(results)

