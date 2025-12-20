"""
Range Directional Model (RDM)
Based on Portela et al. (2004)
Handles negative data and undesirable inputs/outputs
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Optional
import pandas as pd
from .lgo import LGOModel


class RDMModel:
    """
    Range Directional Model
    
    Handles negative data and undesirable inputs/outputs by using
    range-based directional vectors.
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve(self, dmu_index: int, orientation: str = 'no', irdm: bool = False,
              maxslack: bool = True, weight_slack_i: Optional[np.ndarray] = None,
              weight_slack_o: Optional[np.ndarray] = None) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Range Directional Model
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU to evaluate
        orientation : str
            'no' (non-oriented), 'io' (input-oriented), 'oo' (output-oriented)
        irdm : bool
            If True, applies Inverse Range Directional Model
        maxslack : bool
            If True, compute max slack solution
        weight_slack_i : np.ndarray, optional
            Weights for input slacks
        weight_slack_o : np.ndarray, optional
            Weights for output slacks
        
        Returns:
        --------
        rho : float
            Efficiency score
        beta : float
            Directional distance parameter
        lambdas : np.ndarray
            Optimal intensity variables
        target_input : np.ndarray
            Target input values
        target_output : np.ndarray
            Target output values
        """
        # Calculate directional vectors based on range
        dir_input = np.zeros(self.n_inputs)
        dir_output = np.zeros(self.n_outputs)
        
        if orientation != 'oo':
            # dir_input = x_p - min(x_j) for all j in reference set
            dir_input = self.inputs[dmu_index, :] - np.min(self.inputs, axis=0)
        
        if orientation != 'io':
            # dir_output = max(y_j) - y_p for all j in reference set
            dir_output = np.max(self.outputs, axis=0) - self.outputs[dmu_index, :]
        
        if irdm:
            # Inverse RDM: use reciprocal of non-zero directions
            dir_input[dir_input != 0] = 1.0 / dir_input[dir_input != 0]
            dir_output[dir_output != 0] = 1.0 / dir_output[dir_output != 0]
        
        # Use LGO model with calculated directions
        lgo_model = LGOModel(self.inputs, self.outputs)
        rho, beta, lambdas, target_input, target_output, slack_input, slack_output, _, _ = \
            lgo_model.solve(dmu_index, d_input=dir_input, d_output=dir_output,
                          rts='vrs', maxslack=maxslack,
                          weight_slack_i=weight_slack_i, weight_slack_o=weight_slack_o)
        
        return rho, beta, lambdas, target_input, target_output
    
    def evaluate_all(self, orientation: str = 'no', irdm: bool = False,
                    maxslack: bool = True) -> pd.DataFrame:
        """Evaluate all DMUs"""
        results = []
        for i in range(self.n_dmus):
            rho, beta, lambdas, target_in, target_out = self.solve(
                i, orientation, irdm, maxslack
            )
            results.append({
                'DMU': i,
                'Efficiency': rho,
                'Beta': beta,
                'Lambdas': lambdas,
                'Target_Input': target_in,
                'Target_Output': target_out
            })
        return pd.DataFrame(results)

