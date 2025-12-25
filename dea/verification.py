"""
DEA Model Verification Module
Compare results with existing Python DEA libraries
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class DEAVerifier:
    """
    Verify DEA model results against existing Python libraries
    """

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        """
        Initialize verifier
        
        Parameters:
        -----------
        inputs : np.ndarray
            Input matrix (n_dmus × n_inputs)
        outputs : np.ndarray
            Output matrix (n_dmus × n_outputs)
        """
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]

    def verify_ccr(self, our_results: pd.DataFrame, method: str = 'envelopment') -> Dict:
        """
        Verify CCR model results
        
        Parameters:
        -----------
        our_results : pd.DataFrame
            Results from our implementation
        method : str
            'envelopment' or 'multiplier'
        
        Returns:
        --------
        verification : dict
            Verification results with comparison
        """
        verification = {
            'model': 'CCR',
            'method': method,
            'status': 'not_verified',
            'message': '',
            'comparison': None,
            'library': None
        }

        try:
            try:
                from Pyfrontier.frontier_model import EnvelopDEA

                input_data = self.inputs
                output_data = self.outputs

                orient = 'in'
                frontier = 'CRS'

                dea_model = EnvelopDEA(frontier=frontier, orient=orient)
                dea_model.fit(input_data, output_data)
                results = dea_model.results

                pyfrontier_results = np.array([r.score for r in results])

                if 'Efficiency' in our_results.columns:
                    diff = np.abs(our_results['Efficiency'].values - pyfrontier_results)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)

                    verification['status'] = 'verified' if max_diff < 1e-4 else 'warning'
                    verification['message'] = f'Pyfrontier comparison: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}'
                    verification['comparison'] = {
                        'our_efficiency': our_results['Efficiency'].values,
                        'external_efficiency': pyfrontier_results,
                        'difference': diff,
                        'max_difference': max_diff,
                        'mean_difference': mean_diff
                    }
                    verification['library'] = 'Pyfrontier'
                    return verification

            except ImportError:
                pass
            except Exception as e:
                verification['message'] = f'Pyfrontier verification failed: {str(e)}'

            try:
                import pyDEA
                from pyDEA.core.data_processing.input_data import InputData
                from pyDEA.core.models.CCR_model import CCRModel as PyDEACCR

                data = InputData()
                for j in range(self.n_inputs):
                    data.add_input_category(f'Input_{j+1}')
                for j in range(self.n_outputs):
                    data.add_output_category(f'Output_{j+1}')

                for i in range(self.n_dmus):
                    data.add_dmu(f'DMU_{i+1}')
                    for j in range(self.n_inputs):
                        data.set_value(f'DMU_{i+1}', f'Input_{j+1}', self.inputs[i, j])
                    for j in range(self.n_outputs):
                        data.set_value(f'DMU_{i+1}', f'Output_{j+1}', self.outputs[i, j])

                model = PyDEACCR(data)
                pydea_results = []
                for i in range(self.n_dmus):
                    eff = model.evaluate_dmu(f'DMU_{i+1}')
                    pydea_results.append(eff)
                pydea_eff = np.array(pydea_results)

                if 'Efficiency' in our_results.columns:
                    diff = np.abs(our_results['Efficiency'].values - pydea_eff)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)

                    verification['status'] = 'verified' if max_diff < 1e-4 else 'warning'
                    verification['message'] = f'pyDEA comparison: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}'
                    verification['comparison'] = {
                        'our_efficiency': our_results['Efficiency'].values,
                        'external_efficiency': pydea_eff,
                        'difference': diff,
                        'max_difference': max_diff,
                        'mean_difference': mean_diff
                    }
                    verification['library'] = 'pyDEA'
                    return verification

            except ImportError:
                pass
            except Exception as e:
                verification['message'] = f'pyDEA verification failed: {str(e)}'

            if verification['status'] == 'not_verified':
                verification['message'] = 'No external DEA library available. Install Pyfrontier (pip install Pyfrontier) or pytdea/pyDEA from GitHub.'

        except Exception as e:
            verification['status'] = 'error'
            verification['message'] = f'Verification error: {str(e)}'

        return verification

    def verify_bcc(self, our_results: pd.DataFrame, method: str = 'envelopment') -> Dict:
        """
        Verify BCC model results
        """
        verification = {
            'model': 'BCC',
            'method': method,
            'status': 'not_verified',
            'message': '',
            'comparison': None,
            'library': None
        }

        try:
            try:
                from Pyfrontier.frontier_model import EnvelopDEA

                orient = 'in'
                frontier = 'VRS'

                dea_model = EnvelopDEA(frontier=frontier, orient=orient)
                dea_model.fit(self.inputs, self.outputs)
                results = dea_model.results

                pyfrontier_results = np.array([r.score for r in results])

                if 'Efficiency' in our_results.columns:
                    diff = np.abs(our_results['Efficiency'].values - pyfrontier_results)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)

                    verification['status'] = 'verified' if max_diff < 1e-4 else 'warning'
                    verification['message'] = f'Pyfrontier comparison: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}'
                    verification['comparison'] = {
                        'our_efficiency': our_results['Efficiency'].values,
                        'external_efficiency': pyfrontier_results,
                        'difference': diff,
                        'max_difference': max_diff,
                        'mean_difference': mean_diff
                    }
                    verification['library'] = 'Pyfrontier'
                    return verification

            except ImportError:
                pass
            except Exception as e:
                verification['message'] = f'Pyfrontier verification failed: {str(e)}'

            if verification['status'] == 'not_verified':
                verification['message'] = 'No external DEA library available. Install Pyfrontier (pip install Pyfrontier) or pyDEA from GitHub (pip install git+https://github.com/araith/pyDEA.git).'

        except Exception as e:
            verification['status'] = 'error'
            verification['message'] = f'Verification error: {str(e)}'

        return verification

    def verify_sbm(self, our_results: pd.DataFrame) -> Dict:
        """
        Verify SBM model results
        """
        verification = {
            'model': 'SBM',
            'status': 'not_verified',
            'message': '',
            'comparison': None,
            'library': None
        }

        try:
            try:
                import pytdea
                from pytdea import DEA

                dea = DEA(self.inputs, self.outputs, orientation='input', rts='vrs', method='sbm')
                pytdea_results = dea.efficiency()

                if 'SBM_Efficiency' in our_results.columns:
                    diff = np.abs(our_results['SBM_Efficiency'].values - pytdea_results)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)

                    verification['status'] = 'verified' if max_diff < 1e-4 else 'warning'
                    verification['message'] = f'pytdea comparison: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}'
                    verification['comparison'] = {
                        'our_efficiency': our_results['SBM_Efficiency'].values,
                        'external_efficiency': pytdea_results,
                        'difference': diff,
                        'max_difference': max_diff,
                        'mean_difference': mean_diff
                    }
                    verification['library'] = 'pytdea'
                    return verification

            except ImportError:
                pass
            except Exception as e:
                verification['message'] = f'pytdea verification failed: {str(e)}'

            if verification['status'] == 'not_verified':
                verification['message'] = 'SBM verification not available. External libraries do not support SBM verification.'

        except Exception as e:
            verification['status'] = 'error'
            verification['message'] = f'Verification error: {str(e)}'

        return verification

    def verify_ap(self, our_results: pd.DataFrame) -> Dict:
        """
        Verify AP (Super-Efficiency) model results
        """
        verification = {
            'model': 'AP',
            'status': 'not_verified',
            'message': '',
            'comparison': None,
            'library': None
        }

        try:
            try:
                from Pyfrontier.frontier_model import EnvelopDEA

                orient = 'in'
                frontier = 'CRS'

                dea_model = EnvelopDEA(frontier=frontier, orient=orient, super_efficiency=True)
                dea_model.fit(self.inputs, self.outputs)
                results = dea_model.results

                pyfrontier_results = np.array([r.score for r in results])

                if 'Super_Efficiency' in our_results.columns:
                    eff_col = 'Super_Efficiency'
                elif 'Efficiency' in our_results.columns:
                    eff_col = 'Efficiency'
                else:
                    verification['message'] = 'No efficiency column found in results'
                    return verification

                diff = np.abs(our_results[eff_col].values - pyfrontier_results)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)

                verification['status'] = 'verified' if max_diff < 1e-4 else 'warning'
                verification['message'] = f'Pyfrontier comparison: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}'
                verification['comparison'] = {
                    'our_efficiency': our_results[eff_col].values,
                    'external_efficiency': pyfrontier_results,
                    'difference': diff,
                    'max_difference': max_diff,
                    'mean_difference': mean_diff
                }
                verification['library'] = 'Pyfrontier'
                return verification

            except ImportError:
                pass
            except Exception as e:
                verification['message'] = f'Pyfrontier verification failed: {str(e)}'

            if verification['status'] == 'not_verified':
                verification['message'] = 'No external DEA library available. Install Pyfrontier (pip install Pyfrontier).'

        except Exception as e:
            verification['status'] = 'error'
            verification['message'] = f'Verification error: {str(e)}'

        return verification

    def verify_cross_efficiency(self, our_results: pd.DataFrame) -> Dict:
        """
        Verify Cross-Efficiency model results
        """
        verification = {
            'model': 'Cross_Efficiency',
            'status': 'not_verified',
            'message': '',
            'comparison': None,
            'library': None
        }

        try:
            try:
                from Pyfrontier.frontier_model import MultipleDEA

                orient = 'in'
                frontier = 'CRS'

                dea_model = MultipleDEA(frontier=frontier, orient=orient)
                dea_model.fit(self.inputs, self.outputs)

                pyfrontier_cross_eff = np.array(dea_model.cross_efficiency)

                if 'Cross_Efficiency' in our_results.columns:
                    diff = np.abs(our_results['Cross_Efficiency'].values - pyfrontier_cross_eff)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)

                    verification['status'] = 'verified' if max_diff < 1e-4 else 'warning'
                    if max_diff > 0.1:
                        verification['message'] = f'Pyfrontier comparison: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}. Note: Cross-efficiency calculations may differ due to different methods or weight selection strategies.'
                    else:
                        verification['message'] = f'Pyfrontier comparison: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}'
                    verification['comparison'] = {
                        'our_efficiency': our_results['Cross_Efficiency'].values,
                        'external_efficiency': pyfrontier_cross_eff,
                        'difference': diff,
                        'max_difference': max_diff,
                        'mean_difference': mean_diff
                    }
                    verification['library'] = 'Pyfrontier'
                    return verification
                else:
                    verification['message'] = 'No Cross_Efficiency column found in results'
                    return verification

            except ImportError:
                pass
            except Exception as e:
                verification['message'] = f'Pyfrontier verification failed: {str(e)}'

            if verification['status'] == 'not_verified':
                verification['message'] = 'No external DEA library available. Install Pyfrontier (pip install Pyfrontier).'

        except Exception as e:
            verification['status'] = 'error'
            verification['message'] = f'Verification error: {str(e)}'

        return verification

    def verify_generic(self, our_results: pd.DataFrame, model_name: str) -> Dict:
        """
        Generic verification for any model
        """
        verification = {
            'model': model_name,
            'status': 'not_verified',
            'message': f'Verification not available for {model_name} model',
            'comparison': None,
            'library': None
        }

        return verification
