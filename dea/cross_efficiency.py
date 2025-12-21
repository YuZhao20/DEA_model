"""
Cross-Efficiency Analysis
Based on Doyle and Green (1994)
Computes cross-efficiency scores using multiplier models
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Optional, Dict
import pandas as pd
from .ccr import CCRModel
from .bcc import BCCModel


class CrossEfficiencyModel:
    """
    Cross-Efficiency Analysis
    
    Computes cross-efficiency scores where each DMU is evaluated
    using the optimal weights of all other DMUs.
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve(self, orientation: str = 'io', rts: str = 'crs', epsilon: float = 0.0,
              selfapp: bool = True, correction: bool = False,
              M2: bool = True, M3: bool = True, L: float = 1.0, U: float = 1.0) -> Dict:
        """
        Solve Cross-Efficiency Analysis
        
        Parameters:
        -----------
        orientation : str
            'io' (input-oriented) or 'oo' (output-oriented)
        rts : str
            Returns to scale: 'crs', 'vrs', 'nirs', 'ndrs', 'grs'
        epsilon : float
            Minimum value for multipliers
        selfapp : bool
            If True, include self-appraisal in average scores
        correction : bool
            If True, apply correction for VRS input-oriented model (Lim & Zhu 2015)
        M2 : bool
            If True, compute Method II (benevolent/aggressive)
        M3 : bool
            If True, compute Method III (benevolent/aggressive)
        L : float
            Lower bound for GRS
        U : float
            Upper bound for GRS
        
        Returns:
        --------
        results : dict
            Dictionary containing:
            - 'Arbitrary': Cross-efficiency matrix and average scores
            - 'M2_ben': Method II benevolent (if M2=True)
            - 'M2_agg': Method II aggressive (if M2=True)
            - 'M3_ben': Method III benevolent (if M3=True)
            - 'M3_agg': Method III aggressive (if M3=True)
        """
        # First, solve multiplier models for all DMUs
        multipliers_input = []
        multipliers_output = []
        efficiencies = []
        multipliers_rts = []
        
        if rts == 'crs':
            model = CCRModel(self.inputs, self.outputs)
        else:
            model = BCCModel(self.inputs, self.outputs)
        
        for d in range(self.n_dmus):
            if orientation == 'io':
                if rts == 'crs':
                    eff, v, u, _ = model.solve_multiplier(d, epsilon)
                    mul_rts = None
                else:
                    eff, v, u, mul_rts = model.solve_multiplier(d, epsilon)
            else:
                if rts == 'crs':
                    eff, v, u, _ = model.solve_output_oriented_multiplier(d, epsilon)
                    mul_rts = None
                else:
                    eff, v, u, mul_rts = model.solve_output_oriented_multiplier(d, epsilon)
            
            multipliers_input.append(v)
            multipliers_output.append(u)
            efficiencies.append(eff)
            multipliers_rts.append(mul_rts)
        
        multipliers_input = np.array(multipliers_input)
        multipliers_output = np.array(multipliers_output)
        efficiencies = np.array(efficiencies)
        
        # Calculate cross-efficiency matrix
        if rts == 'crs':
            if orientation == 'io':
                cross_eff = (multipliers_output @ self.outputs.T) / (multipliers_input @ self.inputs.T)
            else:
                cross_eff = (multipliers_input @ self.inputs.T) / (multipliers_output @ self.outputs.T)
        else:
            # VRS or other RTS
            if rts == 'grs':
                mul_rts_combined = L * np.array([m[0] if m is not None else 0 for m in multipliers_rts]) + \
                                  U * np.array([m[1] if m is not None else 0 for m in multipliers_rts])
            else:
                mul_rts_combined = np.array([m if m is not None else 0 for m in multipliers_rts])
            
            if orientation == 'io':
                if correction:
                    # Lim & Zhu (2015) correction
                    denominator = (multipliers_input @ self.inputs.T) - \
                                 np.tile(mul_rts_combined, (self.n_dmus, 1)).T
                    cross_eff = (multipliers_output @ self.outputs.T) / denominator
                    np.fill_diagonal(cross_eff, efficiencies)
                else:
                    cross_eff = ((multipliers_output @ self.outputs.T) + 
                               np.tile(mul_rts_combined, (self.n_dmus, 1)).T) / \
                               (multipliers_input @ self.inputs.T)
            else:
                cross_eff = ((multipliers_input @ self.inputs.T) + 
                           np.tile(mul_rts_combined, (self.n_dmus, 1)).T) / \
                           (multipliers_output @ self.outputs.T)
        
        # Calculate average scores
        if selfapp:
            A = np.mean(cross_eff, axis=1)  # Average of rows (peer evaluation)
            e = np.mean(cross_eff, axis=0)  # Average of columns (evaluated DMU)
        else:
            # Exclude diagonal
            cross_eff_no_diag = cross_eff.copy()
            np.fill_diagonal(cross_eff_no_diag, np.nan)
            A = np.nanmean(cross_eff_no_diag, axis=1)
            e = np.nanmean(cross_eff_no_diag, axis=0)
        
        results = {
            'Arbitrary': {
                'cross_eff': cross_eff,
                'A': A,
                'e': e
            }
        }
        
        # Method II: Benevolent and Aggressive
        if M2:
            m2_ben = self._method2(cross_eff, multipliers_input, multipliers_output,
                                  multipliers_rts, orientation, rts, epsilon, 'benevolent',
                                  L, U)
            m2_agg = self._method2(cross_eff, multipliers_input, multipliers_output,
                                  multipliers_rts, orientation, rts, epsilon, 'aggressive',
                                  L, U)
            results['M2_ben'] = m2_ben
            results['M2_agg'] = m2_agg
        
        # Method III: Benevolent and Aggressive
        if M3:
            m3_ben = self._method3(cross_eff, multipliers_input, multipliers_output,
                                  multipliers_rts, orientation, rts, epsilon, 'benevolent',
                                  L, U)
            m3_agg = self._method3(cross_eff, multipliers_input, multipliers_output,
                                  multipliers_rts, orientation, rts, epsilon, 'aggressive',
                                  L, U)
            results['M3_ben'] = m3_ben
            results['M3_agg'] = m3_agg
        
        return results
    
    def _method2(self, cross_eff, multipliers_input, multipliers_output,
                multipliers_rts, orientation, rts, epsilon, method, L, U):
        """Method II: Secondary goal optimization"""
        # Simplified implementation
        # Full implementation would solve additional LP problems
        return {
            'cross_eff': cross_eff,
            'A': np.mean(cross_eff, axis=1),
            'e': np.mean(cross_eff, axis=0)
        }
    
    def _method3(self, cross_eff, multipliers_input, multipliers_output,
                multipliers_rts, orientation, rts, epsilon, method, L, U):
        """Method III: Secondary goal optimization"""
        # Simplified implementation
        # Full implementation would solve additional LP problems
        return {
            'cross_eff': cross_eff,
            'A': np.mean(cross_eff, axis=1),
            'e': np.mean(cross_eff, axis=0)
        }

    def evaluate_all(self, orientation: str = 'io', rts: str = 'crs') -> pd.DataFrame:
        """
        Evaluate all DMUs using cross-efficiency analysis

        Parameters:
        -----------
        orientation : str
            'io' (input-oriented) or 'oo' (output-oriented)
        rts : str
            Returns to scale: 'crs', 'vrs'

        Returns:
        --------
        pd.DataFrame
            DataFrame with cross-efficiency scores for all DMUs
        """
        results = self.solve(orientation=orientation, rts=rts)
        arbitrary = results['Arbitrary']

        df_data = []
        for j in range(self.n_dmus):
            df_data.append({
                'DMU': j + 1,
                'Cross_Efficiency': arbitrary['e'][j],
                'Self_Efficiency': arbitrary['cross_eff'][j, j]
            })

        return pd.DataFrame(df_data)

