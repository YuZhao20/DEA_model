"""
Returns to Scale DEA Models
Based on Chapter 4.5 of Hosseinzadeh Lotfi et al. (2020)
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Dict
import pandas as pd


class ReturnsToScaleModel:
    """
    Returns to Scale DEA Models
    Based on Chapter 4.5
    """

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]

        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")

    def solve_envelopment(self, dmu_index: int) -> Tuple[float, float, str]:
        """
        Solve Returns to Scale - CCR Envelopment Model (4.5.1)
        
        Determines returns to scale by comparing CCR and BCC efficiency scores
        and checking sum of lambdas.
        
        Returns:
        --------
        ccr_efficiency : float
            CCR efficiency score
        sum_lambdas : float
            Sum of optimal lambda values from BCC model
        rts_type : str
            Returns to scale type: 'CRS', 'NIRS', 'NDRS', or 'VRS'
        """
        from .ccr import CCRModel
        from .bcc import BCCModel

        ccr = CCRModel(self.inputs, self.outputs)
        ccr_eff, ccr_lambdas, _, _ = ccr.solve_envelopment(dmu_index)

        bcc = BCCModel(self.inputs, self.outputs)
        bcc_eff, bcc_lambdas, _, _ = bcc.solve_envelopment(dmu_index)

        sum_lambdas = np.sum(bcc_lambdas)

        if abs(ccr_eff - 1.0) < 1e-6 and abs(sum_lambdas - 1.0) < 1e-6:
            rts_type = 'CRS'
        elif ccr_eff < 1.0 and sum_lambdas > 1.0:
            rts_type = 'NIRS'
        elif ccr_eff > 1.0 and sum_lambdas < 1.0:
            rts_type = 'NDRS'
        else:
            rts_type = 'VRS'

        return ccr_eff, sum_lambdas, rts_type

    def solve_multiplier(self, dmu_index: int, epsilon: float = 1e-6) -> Tuple[float, float, str]:
        """
        Solve Returns to Scale - DEA Multiplier Model (4.5.2)
        
        Uses BCC multiplier model to determine returns to scale via u0 value.
        
        Returns:
        --------
        bcc_efficiency : float
            BCC efficiency score
        u0 : float
            Optimal u0 value
        rts_type : str
            Returns to scale type based on u0
        """
        from .bcc import BCCModel

        bcc = BCCModel(self.inputs, self.outputs)
        bcc_eff, _, _, u0 = bcc.solve_multiplier(dmu_index, epsilon)

        if abs(u0) < 1e-6:
            rts_type = 'CRS'
        elif u0 < 0:
            rts_type = 'NIRS'
        elif u0 > 0:
            rts_type = 'NDRS'
        else:
            rts_type = 'VRS'

        return bcc_eff, u0, rts_type

    def evaluate_all(self, method: str = 'envelopment') -> pd.DataFrame:
        """
        Evaluate all DMUs
        
        Parameters:
        -----------
        method : str
            'envelopment' or 'multiplier'
        
        Returns:
        --------
        results : pd.DataFrame
            DataFrame with returns to scale analysis
        """
        results = []

        for j in range(self.n_dmus):
            if method == 'envelopment':
                ccr_eff, sum_lambdas, rts_type = self.solve_envelopment(j)
                result_dict = {
                    'DMU': j + 1,
                    'CCR_Efficiency': ccr_eff,
                    'Sum_Lambdas': sum_lambdas,
                    'RTS_Type': rts_type
                }
            else:
                bcc_eff, u0, rts_type = self.solve_multiplier(j)
                result_dict = {
                    'DMU': j + 1,
                    'BCC_Efficiency': bcc_eff,
                    'u0': u0,
                    'RTS_Type': rts_type
                }

            results.append(result_dict)

        return pd.DataFrame(results)

