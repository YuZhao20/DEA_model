"""
Efficiency Ladder Analysis
Based on Benchmarking package

Efficiency ladder shows how efficiency changes as the most influential peer
is removed sequentially.
"""

import numpy as np
from typing import List, Tuple
import pandas as pd


class EfficiencyLadderModel:
    """
    Efficiency Ladder Model
    
    Removes peers sequentially and tracks efficiency changes.
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, rts: str = 'vrs'):
        """
        Initialize Efficiency Ladder model
        
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
    
    def _get_peers(self, dmu_index: int, excluded_indices: List[int] = None) -> List[int]:
        """Get peer indices for a DMU"""
        from .bcc import BCCModel
        from .ccr import CCRModel
        
        if self.rts == 'vrs':
            model = BCCModel(self.inputs, self.outputs)
        elif self.rts == 'crs':
            model = CCRModel(self.inputs, self.outputs)
        else:
            model = BCCModel(self.inputs, self.outputs)  # Default to BCC
        
        eff, lambdas, _, _ = model.solve_envelopment(dmu_index)
        
        # Find peers (lambda > threshold)
        peers = [i for i in range(len(lambdas)) if lambdas[i] > 1e-6]
        
        # Exclude specified indices
        if excluded_indices:
            peers = [p for p in peers if p not in excluded_indices]
        
        return peers
    
    def _solve_with_excluded(self, dmu_index: int, excluded_indices: List[int]) -> float:
        """Solve DEA with excluded DMUs"""
        from .bcc import BCCModel
        from .ccr import CCRModel
        
        # Create filtered inputs and outputs
        available_indices = [i for i in range(self.n_dmus) if i not in excluded_indices]
        
        if len(available_indices) == 0:
            return np.inf
        
        inputs_filtered = self.inputs[available_indices, :]
        outputs_filtered = self.outputs[available_indices, :]
        
        # Find position of dmu_index in filtered array
        if dmu_index in excluded_indices:
            return np.inf
        
        dmu_pos = available_indices.index(dmu_index)
        
        if self.rts == 'vrs':
            model = BCCModel(inputs_filtered, outputs_filtered)
        elif self.rts == 'crs':
            model = CCRModel(inputs_filtered, outputs_filtered)
        else:
            model = BCCModel(inputs_filtered, outputs_filtered)
        
        try:
            eff, _, _, _ = model.solve_envelopment(dmu_pos)
            return eff
        except:
            return np.inf
    
    def solve(self, dmu_index: int, max_steps: int = None, orientation: str = 'in') -> Tuple[List[float], List[int], List[int]]:
        """
        Solve Efficiency Ladder for a DMU
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU to analyze
        max_steps : int
            Maximum number of steps (default: all peers)
        orientation : str
            'in' or 'out'
        
        Returns:
        --------
        efficiencies : List[float]
            Efficiency scores at each step
        removed_peers : List[int]
            Indices of removed peers in order
        final_peers : List[int]
            Remaining peers after all removals
        """
        if max_steps is None:
            max_steps = self.n_dmus
        
        excluded_indices = []
        efficiencies = []
        removed_peers = []
        
        for step in range(max_steps):
            # Solve with current exclusions
            eff = self._solve_with_excluded(dmu_index, excluded_indices)
            
            if np.isinf(eff) or np.isnan(eff):
                break
            
            efficiencies.append(eff)
            
            # Get current peers
            peers = self._get_peers(dmu_index, excluded_indices)
            
            if len(peers) == 0:
                break
            
            # For each peer, calculate efficiency if removed
            peer_effects = []
            for peer in peers:
                test_excluded = excluded_indices + [peer]
                test_eff = self._solve_with_excluded(dmu_index, test_excluded)
                peer_effects.append((peer, test_eff))
            
            # Find peer with maximum effect (largest efficiency increase)
            if not peer_effects:
                break
            
            # Sort by efficiency (descending) to find peer with largest impact
            peer_effects.sort(key=lambda x: x[1], reverse=True)
            best_peer, best_eff = peer_effects[0]
            
            # If removing this peer doesn't increase efficiency, stop
            if best_eff <= eff + 1e-6:
                break
            
            # Remove the peer with maximum effect
            excluded_indices.append(best_peer)
            removed_peers.append(best_peer)
        
        # Get final peers
        final_peers = self._get_peers(dmu_index, excluded_indices)
        
        return efficiencies, removed_peers, final_peers
    
    def evaluate_all(self, max_steps: int = None) -> pd.DataFrame:
        results = []
        for j in range(self.n_dmus):
            effs, removed, final = self.solve(j, max_steps)
            
            result_dict = {
                'DMU': j + 1,
                'Initial_Efficiency': effs[0] if effs else np.nan,
                'Final_Efficiency': effs[-1] if effs else np.nan,
                'Steps': len(effs),
                'Removed_Peers': str(removed),
                'Final_Peers': str(final)
            }
            results.append(result_dict)
        return pd.DataFrame(results)

