"""
Merger Analysis DEA Model
Based on Benchmarking package

Decomposes efficiency gains from mergers of firms.
"""

import numpy as np
from typing import Tuple
import pandas as pd


class MergerAnalysisModel:
    """
    Merger Analysis Model
    
    Analyzes potential gains from merging firms and decomposes them into:
    - Learning effect
    - Harmony effect
    - Size effect
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, rts: str = 'vrs'):
        """
        Initialize Merger Analysis model
        
        Parameters:
        -----------
        inputs : np.ndarray
            Input matrix
        outputs : np.ndarray
            Output matrix
        rts : str
            Returns to scale: 'crs', 'vrs', 'drs', 'irs'
        """
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.rts = rts.lower()
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def _solve_dea(self, inputs_eval: np.ndarray, outputs_eval: np.ndarray,
                   inputs_ref: np.ndarray = None, outputs_ref: np.ndarray = None,
                   orientation: str = 'in') -> np.ndarray:
        """Solve DEA for given inputs/outputs"""
        from .bcc import BCCModel
        from .ccr import CCRModel
        
        if inputs_ref is None:
            inputs_ref = self.inputs
        if outputs_ref is None:
            outputs_ref = self.outputs
        
        if self.rts == 'vrs':
            model = BCCModel(inputs_ref, outputs_ref)
        elif self.rts == 'crs':
            model = CCRModel(inputs_ref, outputs_ref)
        else:
            model = BCCModel(inputs_ref, outputs_ref)
        
        efficiencies = []
        for i in range(len(inputs_eval)):
            if orientation == 'in':
                eff, _, _, _ = model.solve_envelopment(i)
            else:
                eff, _, _, _ = model.solve_output_oriented_envelopment(i)
            efficiencies.append(eff)
        
        return np.array(efficiencies)
    
    def solve(self, merger_matrix: np.ndarray, orientation: str = 'in') -> dict:
        """
        Solve Merger Analysis
        
        Parameters:
        -----------
        merger_matrix : np.ndarray
            Binary matrix of shape (n_mergers, n_dmus)
            Each row defines a merger (1 = included, 0 = excluded)
        orientation : str
            'in' or 'out'
        
        Returns:
        --------
        results : dict
            Dictionary with efficiency measures and effects
        """
        n_mergers = merger_matrix.shape[0]
        
        # Calculate merged inputs and outputs
        X_merger = merger_matrix @ self.inputs
        Y_merger = merger_matrix @ self.outputs
        
        # Potential gains (efficiency of merged firms)
        E = self._solve_dea(X_merger, Y_merger, self.inputs, self.outputs, orientation)
        
        # Individual efficiencies
        e = self._solve_dea(self.inputs, self.outputs, self.inputs, self.outputs, orientation)
        
        # Project individual firms to efficient frontier
        if orientation == 'in':
            X_eff = np.diag(e) @ self.inputs
            X_merger_proj = merger_matrix @ X_eff
            Y_merger_proj = Y_merger
        else:  # output
            Y_eff = np.diag(e) @ self.outputs
            X_merger_proj = X_merger
            Y_merger_proj = merger_matrix @ Y_eff
        
        # Pure gains from mergers (after eliminating individual inefficiency)
        E_star = self._solve_dea(X_merger_proj, Y_merger_proj, self.inputs, self.outputs, orientation)
        
        # Learning effect
        LE = E / (E_star + 1e-10)  # Avoid division by zero
        
        # Harmony effect (average inputs/outputs)
        merger_sizes = np.sum(merger_matrix, axis=1)
        X_harm = np.diag(1.0 / merger_sizes) @ X_merger_proj
        Y_harm = np.diag(1.0 / merger_sizes) @ Y_merger_proj
        
        HA = self._solve_dea(X_harm, Y_harm, self.inputs, self.outputs, orientation)
        
        # Size effect
        SI = E_star / (HA + 1e-10)
        
        return {
            'E': E,  # Potential gains
            'E_star': E_star,  # Pure merger gains
            'learning': LE,  # Learning effect
            'harmony': HA,  # Harmony effect
            'size': SI  # Size effect
        }
    
    def evaluate_all(self, merger_matrix: np.ndarray, orientation: str = 'in') -> pd.DataFrame:
        """
        Evaluate all mergers
        
        Parameters:
        -----------
        merger_matrix : np.ndarray
            Binary matrix defining mergers
        orientation : str
            'in' or 'out'
        """
        results_dict = self.solve(merger_matrix, orientation)
        
        n_mergers = merger_matrix.shape[0]
        results = []
        
        for i in range(n_mergers):
            result_dict = {
                'Merger': i + 1,
                'E': results_dict['E'][i],
                'E_star': results_dict['E_star'][i],
                'Learning_Effect': results_dict['learning'][i],
                'Harmony_Effect': results_dict['harmony'][i],
                'Size_Effect': results_dict['size'][i]
            }
            results.append(result_dict)
        
        return pd.DataFrame(results)

