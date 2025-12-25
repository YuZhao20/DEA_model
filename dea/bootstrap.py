"""
Bootstrap DEA Model
Based on Benchmarking package

Bootstrap method for DEA efficiency scores following Simar & Wilson (1998).
"""

import numpy as np
from typing import Tuple, Optional
import pandas as pd
from scipy.stats import gaussian_kde


class BootstrapDEAModel:
    """
    Bootstrap DEA Model
    
    Implements bootstrap method for DEA efficiency scores to provide
    confidence intervals and bias correction.
    """

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray,
                 rts: str = 'vrs', orientation: str = 'in'):
        """
        Initialize Bootstrap DEA model
        
        Parameters:
        -----------
        inputs : np.ndarray
            Input matrix
        outputs : np.ndarray
            Output matrix
        rts : str
            Returns to scale: 'crs', 'vrs', 'drs', 'irs'
        orientation : str
            'in' or 'out'
        """
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.rts = rts.lower()
        self.orientation = orientation.lower()
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]

        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")

    def _solve_dea(self, inputs_eval: np.ndarray, outputs_eval: np.ndarray,
                   inputs_ref: np.ndarray = None, outputs_ref: np.ndarray = None) -> np.ndarray:
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
            if self.orientation == 'in':
                eff, _, _, _ = model.solve_envelopment(i)
            else:
                eff, _, _, _ = model.solve_output_oriented_envelopment(i)
            efficiencies.append(eff)

        return np.array(efficiencies)

    def _calculate_bandwidth(self, efficiencies: np.ndarray) -> float:
        """Calculate bandwidth for kernel density estimation"""
        if self.orientation == 'in':
            dist = 1.0 / (efficiencies + 1e-10)
        else:
            dist = efficiencies

        zeff = dist[dist > 1.0 + 1e-6]
        if len(zeff) == 0:
            return 0.1

        neff = np.concatenate([zeff, 2 - zeff])

        adjust = (np.std(dist) / np.std(neff)) * (len(neff) / len(dist)) ** 0.2

        h = 1.06 * np.std(neff) * len(neff) ** (-0.2) * adjust

        return max(h, 0.01)

    def _smooth_bootstrap_sample(self, efficiencies: np.ndarray, n_rep: int) -> np.ndarray:
        """Generate smoothed bootstrap sample"""
        if self.orientation == 'in':
            dist = 1.0 / (efficiencies + 1e-10)
        else:
            dist = efficiencies

        h = self._calculate_bandwidth(efficiencies)

        try:
            kde = gaussian_kde(dist)
            samples = kde.resample(n_rep)[0]
        except:
            samples = np.random.choice(dist, size=n_rep, replace=True)
            samples = samples + np.random.normal(0, h, size=n_rep)

        samples = np.where(samples < 1.0, 2.0 - samples, samples)

        if self.orientation == 'in':
            samples = 1.0 / samples

        return samples

    def solve(self, n_rep: int = 200, alpha: float = 0.05,
              seed: Optional[int] = None) -> dict:
        """
        Solve Bootstrap DEA
        
        Parameters:
        -----------
        n_rep : int
            Number of bootstrap replications
        alpha : float
            Significance level for confidence intervals
        seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        results : dict
            Dictionary with bootstrap results
        """
        if seed is not None:
            np.random.seed(seed)

        eff_initial = self._solve_dea(self.inputs, self.outputs, self.inputs, self.outputs)

        boot_samples = np.zeros((self.n_dmus, n_rep))

        for dmu_idx in range(self.n_dmus):
            boot_eff = self._smooth_bootstrap_sample(eff_initial, n_rep)
            boot_samples[dmu_idx, :] = boot_eff

        eff_bc = 2 * eff_initial - np.mean(boot_samples, axis=1)
        bias = eff_initial - np.mean(boot_samples, axis=1)
        var = np.var(boot_samples, axis=1)

        conf_int = np.zeros((self.n_dmus, 2))
        for i in range(self.n_dmus):
            sorted_boot = np.sort(boot_samples[i, :])
            lower_idx = int(n_rep * alpha / 2)
            upper_idx = int(n_rep * (1 - alpha / 2))
            conf_int[i, 0] = sorted_boot[lower_idx]
            conf_int[i, 1] = sorted_boot[upper_idx]

        return {
            'eff': eff_initial,
            'eff_bc': eff_bc,
            'bias': bias,
            'var': var,
            'conf_int': conf_int,
            'boot': boot_samples
        }

    def evaluate_all(self, n_rep: int = 200, alpha: float = 0.05,
                     seed: Optional[int] = None) -> pd.DataFrame:
        """Evaluate all DMUs with bootstrap"""
        results_dict = self.solve(n_rep, alpha, seed)

        results = []
        for i in range(self.n_dmus):
            result_dict = {
                'DMU': i + 1,
                'Efficiency': results_dict['eff'][i],
                'Bias_Corrected': results_dict['eff_bc'][i],
                'Bias': results_dict['bias'][i],
                'Variance': results_dict['var'][i],
                'CI_Lower': results_dict['conf_int'][i, 0],
                'CI_Upper': results_dict['conf_int'][i, 1]
            }
            results.append(result_dict)

        return pd.DataFrame(results)

