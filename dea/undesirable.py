"""
Undesirable Inputs/Outputs Handling
Based on Seiford and Zhu (2002)
Transforms undesirable inputs/outputs for use with standard DEA models
"""

import numpy as np
from typing import Optional, Tuple


def transform_undesirable(inputs: np.ndarray, outputs: np.ndarray,
                          ud_inputs: Optional[np.ndarray] = None,
                          ud_outputs: Optional[np.ndarray] = None,
                          vtrans_i: Optional[np.ndarray] = None,
                          vtrans_o: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform undesirable inputs/outputs according to Seiford and Zhu (2002)
    
    For undesirable inputs: x_new = vtrans_i - x_old
    For undesirable outputs: y_new = vtrans_o - y_old
    
    Parameters:
    -----------
    inputs : np.ndarray
        Input matrix (n_dmus x n_inputs)
    outputs : np.ndarray
        Output matrix (n_dmus x n_outputs)
    ud_inputs : np.ndarray, optional
        Indices of undesirable inputs (0-based)
    ud_outputs : np.ndarray, optional
        Indices of undesirable outputs (0-based)
    vtrans_i : np.ndarray, optional
        Translation vector for undesirable inputs. If None, uses max + 1
    vtrans_o : np.ndarray, optional
        Translation vector for undesirable outputs. If None, uses max + 1
    
    Returns:
    --------
    transformed_inputs : np.ndarray
        Transformed input matrix
    transformed_outputs : np.ndarray
        Transformed output matrix
    vtrans_i_used : np.ndarray
        Translation vector actually used for inputs
    vtrans_o_used : np.ndarray
        Translation vector actually used for outputs
    """
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    transformed_inputs = inputs.copy()
    transformed_outputs = outputs.copy()
    
    n_inputs = inputs.shape[1]
    n_outputs = outputs.shape[1]
    
    # Handle undesirable inputs
    vtrans_i_used = None
    if ud_inputs is not None and len(ud_inputs) > 0:
        ud_inputs = np.array(ud_inputs)
        if np.any(ud_inputs < 0) or np.any(ud_inputs >= n_inputs):
            raise ValueError("Invalid undesirable input indices")
        
        nui = len(ud_inputs)
        
        if vtrans_i is None:
            vtrans_i_used = np.array([np.max(inputs[:, i]) + 1 for i in ud_inputs])
        elif np.isscalar(vtrans_i):
            vtrans_i_used = np.full(nui, vtrans_i)
        else:
            vtrans_i_used = np.array(vtrans_i)
            if len(vtrans_i_used) != nui:
                raise ValueError("vtrans_i length must match number of undesirable inputs")
        
        for idx, i in enumerate(ud_inputs):
            transformed_inputs[:, i] = vtrans_i_used[idx] - inputs[:, i]
            if np.any(transformed_inputs[:, i] < 0):
                raise ValueError(f"Negative transformed input at index {i}. Increase vtrans_i[{idx}]")
    
    # Handle undesirable outputs
    vtrans_o_used = None
    if ud_outputs is not None and len(ud_outputs) > 0:
        ud_outputs = np.array(ud_outputs)
        if np.any(ud_outputs < 0) or np.any(ud_outputs >= n_outputs):
            raise ValueError("Invalid undesirable output indices")
        
        nuo = len(ud_outputs)
        
        if vtrans_o is None:
            vtrans_o_used = np.array([np.max(outputs[:, r]) + 1 for r in ud_outputs])
        elif np.isscalar(vtrans_o):
            vtrans_o_used = np.full(nuo, vtrans_o)
        else:
            vtrans_o_used = np.array(vtrans_o)
            if len(vtrans_o_used) != nuo:
                raise ValueError("vtrans_o length must match number of undesirable outputs")
        
        for idx, r in enumerate(ud_outputs):
            transformed_outputs[:, r] = vtrans_o_used[idx] - outputs[:, r]
            if np.any(transformed_outputs[:, r] < 0):
                raise ValueError(f"Negative transformed output at index {r}. Increase vtrans_o[{idx}]")
    
    return transformed_inputs, transformed_outputs, vtrans_i_used, vtrans_o_used

