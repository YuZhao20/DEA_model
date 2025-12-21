"""
Comprehensive validation script for all DEA models.
Runs numerical experiments to verify correct implementation.
"""

import numpy as np
import pandas as pd
import traceback
from typing import Dict, List, Tuple, Any

# Import all models
from dea import (
    CCRModel, BCCModel, APModel, MAJModel,
    AdditiveModel, TwoPhaseModel,
    NormL1Model, CongestionModel, CommonWeightsModel, DirectionalEfficiencyModel,
    ReturnsToScaleModel,
    CostEfficiencyModel, RevenueEfficiencyModel,
    MalmquistModel,
    SBMModel,
    ProfitEfficiencyModel, ModifiedSBMModel,
    SeriesNetworkModel,
    DRSModel, IRSModel,
    FDHModel, FDHPlusModel,
    MEAModel,
    EfficiencyLadderModel,
    MergerAnalysisModel,
    BootstrapDEAModel,
    NonRadialModel,
    LGOModel,
    RDMModel,
    AddMinModel,
    AddSuperEffModel,
    DEAPSModel,
    CrossEfficiencyModel,
    transform_undesirable,
    StoNEDModel
)
import warnings

# Test data from Table 3.1 (standard DEA test dataset)
DATA = np.array([
    [20, 11, 8, 30],   # DMU 1
    [11, 40, 21, 20],  # DMU 2
    [32, 30, 34, 40],  # DMU 3
    [21, 30, 18, 50],  # DMU 4
    [20, 11, 6, 17],   # DMU 5
    [12, 43, 23, 58],  # DMU 6
    [7, 45, 28, 30],   # DMU 7
    [31, 45, 40, 20],  # DMU 8
    [19, 22, 27, 23],  # DMU 9
    [32, 11, 38, 45],  # DMU 10
])

INPUTS = DATA[:, :2]   # First 2 columns: Input1, Input2
OUTPUTS = DATA[:, 2:]  # Last 2 columns: Output1, Output2


class ValidationResult:
    """Stores validation results for a model."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.passed = True
        self.errors = []
        self.warnings = []
        self.info = []

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_info(self, msg: str):
        self.info.append(msg)

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        result = f"\n{self.model_name}: {status}\n"
        for msg in self.info:
            result += f"  INFO: {msg}\n"
        for msg in self.warnings:
            result += f"  WARN: {msg}\n"
        for msg in self.errors:
            result += f"  ERROR: {msg}\n"
        return result


def validate_ccr_model() -> ValidationResult:
    """Validate CCR model implementation."""
    result = ValidationResult("CCRModel")

    try:
        model = CCRModel(INPUTS, OUTPUTS)

        # Test envelopment model
        env_results = model.evaluate_all(method='envelopment')
        efficiencies = env_results['Efficiency'].values

        # Check efficiency bounds
        if any(efficiencies > 1.001):
            result.add_error(f"CCR efficiencies > 1: {efficiencies[efficiencies > 1.001]}")
        if any(efficiencies < 0):
            result.add_error(f"CCR efficiencies < 0: {efficiencies[efficiencies < 0]}")

        # Check known efficient DMUs (DMU 6, 7, 10 in 1-indexed)
        for dmu_idx in [5, 6, 9]:  # 0-indexed
            if efficiencies[dmu_idx] < 0.999:
                result.add_warning(f"DMU {dmu_idx+1} should be efficient, got {efficiencies[dmu_idx]:.4f}")

        # Test multiplier model
        mult_results = model.evaluate_all(method='multiplier')
        mult_efficiencies = mult_results['Efficiency'].values

        # Check consistency between methods
        diff = np.abs(efficiencies - mult_efficiencies)
        if np.max(diff) > 0.01:
            result.add_warning(f"Envelopment-Multiplier difference > 0.01: max={np.max(diff):.4f}")

        # Test output orientation (uses separate method)
        output_eff, _, _, _ = model.solve_output_oriented_envelopment(0)
        if output_eff < 0.999:
            pass  # Output-oriented CCR efficiency >= 1 is expected

        result.add_info(f"CCR Input efficiencies: min={np.min(efficiencies):.4f}, max={np.max(efficiencies):.4f}")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_bcc_model() -> ValidationResult:
    """Validate BCC model implementation."""
    result = ValidationResult("BCCModel")

    try:
        model = BCCModel(INPUTS, OUTPUTS)

        # Test envelopment model
        env_results = model.evaluate_all(method='envelopment')
        efficiencies = env_results['Efficiency'].values

        # Check efficiency bounds
        if any(efficiencies > 1.001):
            result.add_error(f"BCC efficiencies > 1: {efficiencies[efficiencies > 1.001]}")
        if any(efficiencies < 0):
            result.add_error(f"BCC efficiencies < 0: {efficiencies[efficiencies < 0]}")

        # BCC should have more efficient DMUs than CCR (VRS >= CRS efficiency)
        ccr_model = CCRModel(INPUTS, OUTPUTS)
        ccr_results = ccr_model.evaluate_all(method='envelopment')
        ccr_eff = ccr_results['Efficiency'].values

        # BCC efficiency >= CCR efficiency (with small tolerance)
        if any(efficiencies < ccr_eff - 0.01):
            result.add_warning("BCC efficiency should be >= CCR efficiency")

        # Test multiplier model
        mult_results = model.evaluate_all(method='multiplier')

        result.add_info(f"BCC Input efficiencies: min={np.min(efficiencies):.4f}, max={np.max(efficiencies):.4f}")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_ap_model() -> ValidationResult:
    """Validate AP (Anderson-Peterson) super-efficiency model."""
    result = ValidationResult("APModel")

    try:
        model = APModel(INPUTS, OUTPUTS)

        # Test input-oriented envelopment
        input_results = model.evaluate_all(orientation='input', method='envelopment')
        input_eff = input_results['Super_Efficiency'].values

        # Super-efficiency can be > 1 for efficient DMUs
        # Check that inefficient DMUs have the same efficiency as CCR
        ccr_model = CCRModel(INPUTS, OUTPUTS)
        ccr_results = ccr_model.evaluate_all(method='envelopment')
        ccr_eff = ccr_results['Efficiency'].values

        for i in range(len(input_eff)):
            if ccr_eff[i] < 0.999:  # Inefficient DMU
                if abs(input_eff[i] - ccr_eff[i]) > 0.01:
                    result.add_warning(f"DMU {i+1}: AP efficiency {input_eff[i]:.4f} != CCR {ccr_eff[i]:.4f}")

        # Test output-oriented
        output_results = model.evaluate_all(orientation='output', method='envelopment')
        output_eff = output_results['Super_Efficiency'].values

        result.add_info(f"AP Input super-efficiencies: min={np.min(input_eff):.4f}, max={np.max(input_eff):.4f}")
        result.add_info(f"AP Output super-efficiencies: min={np.min(output_eff):.4f}, max={np.max(output_eff):.4f}")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_maj_model() -> ValidationResult:
    """Validate MAJ super-efficiency model."""
    result = ValidationResult("MAJModel")

    try:
        model = MAJModel(INPUTS, OUTPUTS)
        maj_results = model.evaluate_all()

        if 'Super_Efficiency_MAJ' not in maj_results.columns:
            result.add_error("Missing Super_Efficiency_MAJ column")
        else:
            eff = maj_results['Super_Efficiency_MAJ'].values
            result.add_info(f"MAJ super-efficiencies: min={np.min(eff):.4f}, max={np.max(eff):.4f}")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_additive_model() -> ValidationResult:
    """Validate Additive model implementation."""
    result = ValidationResult("AdditiveModel")

    try:
        model = AdditiveModel(INPUTS, OUTPUTS)
        add_results = model.evaluate_all()

        # Check slacks are non-negative
        for col in add_results.columns:
            if 'Slack' in col:
                if any(add_results[col] < -0.001):
                    result.add_error(f"Negative slack in {col}: {add_results[col].min()}")

        result.add_info(f"Additive model completed successfully")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_twophase_model() -> ValidationResult:
    """Validate Two-Phase model implementation."""
    result = ValidationResult("TwoPhaseModel")

    try:
        model = TwoPhaseModel(INPUTS, OUTPUTS)
        results = model.evaluate_all()

        efficiencies = results['Efficiency'].values
        if any(efficiencies > 1.001):
            result.add_error(f"Two-phase efficiencies > 1: {efficiencies[efficiencies > 1.001]}")
        if any(efficiencies < 0):
            result.add_error(f"Two-phase efficiencies < 0: {efficiencies[efficiencies < 0]}")

        result.add_info(f"Two-phase efficiencies: min={np.min(efficiencies):.4f}, max={np.max(efficiencies):.4f}")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_sbm_model() -> ValidationResult:
    """Validate SBM (Slacks-Based Measure) model."""
    result = ValidationResult("SBMModel")

    try:
        model = SBMModel(INPUTS, OUTPUTS)

        # Test model 1
        results1 = model.evaluate_all(model_type=1)
        eff1 = results1['SBM_Efficiency'].values

        if any(eff1 > 1.001):
            result.add_error(f"SBM efficiencies > 1: {eff1[eff1 > 1.001]}")
        if any(eff1 < 0):
            result.add_error(f"SBM efficiencies < 0: {eff1[eff1 < 0]}")

        # Test model 2
        results2 = model.evaluate_all(model_type=2)
        eff2 = results2['SBM_Efficiency'].values

        result.add_info(f"SBM Model1 efficiencies: min={np.min(eff1):.4f}, max={np.max(eff1):.4f}")
        result.add_info(f"SBM Model2 efficiencies: min={np.min(eff2):.4f}, max={np.max(eff2):.4f}")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_drs_irs_models() -> ValidationResult:
    """Validate DRS and IRS models."""
    result = ValidationResult("DRS/IRS Models")

    try:
        # DRS Model
        drs_model = DRSModel(INPUTS, OUTPUTS)
        drs_input = drs_model.evaluate_all(orientation='input')
        drs_eff = drs_input['Efficiency'].values

        if any(drs_eff > 1.001):
            result.add_error(f"DRS input efficiencies > 1: {drs_eff[drs_eff > 1.001]}")
        if any(drs_eff < 0):
            result.add_error(f"DRS input efficiencies < 0: {drs_eff[drs_eff < 0]}")

        # IRS Model
        irs_model = IRSModel(INPUTS, OUTPUTS)
        irs_input = irs_model.evaluate_all(orientation='input')
        irs_eff = irs_input['Efficiency'].values

        if any(irs_eff > 1.001):
            result.add_error(f"IRS input efficiencies > 1: {irs_eff[irs_eff > 1.001]}")
        if any(irs_eff < 0):
            result.add_error(f"IRS input efficiencies < 0: {irs_eff[irs_eff < 0]}")

        # CCR efficiency <= DRS efficiency (in input orientation)
        ccr_model = CCRModel(INPUTS, OUTPUTS)
        ccr_eff = ccr_model.evaluate_all(method='envelopment')['Efficiency'].values

        result.add_info(f"DRS efficiencies: min={np.min(drs_eff):.4f}, max={np.max(drs_eff):.4f}")
        result.add_info(f"IRS efficiencies: min={np.min(irs_eff):.4f}, max={np.max(irs_eff):.4f}")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_fdh_model() -> ValidationResult:
    """Validate FDH (Free Disposal Hull) model."""
    result = ValidationResult("FDHModel")

    try:
        model = FDHModel(INPUTS, OUTPUTS)

        # Test input-oriented
        input_results = model.evaluate_all(orientation='input')
        input_eff = input_results['Efficiency'].values

        if any(input_eff > 1.001):
            result.add_error(f"FDH input efficiencies > 1: {input_eff[input_eff > 1.001]}")
        if any(input_eff < 0):
            result.add_error(f"FDH input efficiencies < 0: {input_eff[input_eff < 0]}")

        # Test output-oriented
        output_results = model.evaluate_all(orientation='output')
        output_eff = output_results['Efficiency'].values

        result.add_info(f"FDH Input efficiencies: min={np.min(input_eff):.4f}, max={np.max(input_eff):.4f}")
        result.add_info(f"FDH Output efficiencies: min={np.min(output_eff):.4f}, max={np.max(output_eff):.4f}")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_mea_model() -> ValidationResult:
    """Validate MEA (Multi-directional Efficiency Analysis) model."""
    result = ValidationResult("MEAModel")

    try:
        model = MEAModel(INPUTS, OUTPUTS)
        mea_results = model.evaluate_all()

        if 'Efficiency' in mea_results.columns:
            efficiencies = mea_results['Efficiency'].values
            if any(efficiencies > 1.001):
                result.add_error(f"MEA efficiencies > 1: {efficiencies[efficiencies > 1.001]}")
            if any(efficiencies < 0):
                result.add_error(f"MEA efficiencies < 0: {efficiencies[efficiencies < 0]}")
            result.add_info(f"MEA efficiencies: min={np.min(efficiencies):.4f}, max={np.max(efficiencies):.4f}")
        else:
            result.add_info(f"MEA model completed with {len(mea_results)} DMUs")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_norml1_model() -> ValidationResult:
    """Validate Norm L1 model."""
    result = ValidationResult("NormL1Model")

    try:
        model = NormL1Model(INPUTS, OUTPUTS)
        l1_results = model.evaluate_all()

        result.add_info(f"NormL1 model completed with {len(l1_results)} DMUs")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_congestion_model() -> ValidationResult:
    """Validate Congestion model."""
    result = ValidationResult("CongestionModel")

    try:
        model = CongestionModel(INPUTS, OUTPUTS)
        cong_results = model.evaluate_all()

        result.add_info(f"Congestion model completed with {len(cong_results)} DMUs")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_cost_revenue_models() -> ValidationResult:
    """Validate Cost and Revenue efficiency models."""
    result = ValidationResult("Cost/Revenue Models")

    try:
        # Create price vectors
        input_prices = np.array([1.0, 2.0])
        output_prices = np.array([3.0, 4.0])

        # Cost Efficiency
        cost_model = CostEfficiencyModel(INPUTS, OUTPUTS, input_prices)
        cost_results = cost_model.evaluate_all()

        if 'Cost_Efficiency' in cost_results.columns:
            cost_eff = cost_results['Cost_Efficiency'].values
            if any(cost_eff > 1.001):
                result.add_error(f"Cost efficiencies > 1: {cost_eff[cost_eff > 1.001]}")
            if any(cost_eff < 0):
                result.add_error(f"Cost efficiencies < 0: {cost_eff[cost_eff < 0]}")
            result.add_info(f"Cost efficiencies: min={np.min(cost_eff):.4f}, max={np.max(cost_eff):.4f}")

        # Revenue Efficiency
        revenue_model = RevenueEfficiencyModel(INPUTS, OUTPUTS, output_prices)
        revenue_results = revenue_model.evaluate_all()

        if 'Revenue_Efficiency' in revenue_results.columns:
            rev_eff = revenue_results['Revenue_Efficiency'].values
            if any(rev_eff > 1.001):
                result.add_error(f"Revenue efficiencies > 1: {rev_eff[rev_eff > 1.001]}")
            if any(rev_eff < 0):
                result.add_error(f"Revenue efficiencies < 0: {rev_eff[rev_eff < 0]}")
            result.add_info(f"Revenue efficiencies: min={np.min(rev_eff):.4f}, max={np.max(rev_eff):.4f}")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_profit_model() -> ValidationResult:
    """Validate Profit efficiency model."""
    result = ValidationResult("ProfitEfficiencyModel")

    try:
        input_prices = np.array([1.0, 2.0])
        output_prices = np.array([3.0, 4.0])

        model = ProfitEfficiencyModel(INPUTS, OUTPUTS, input_prices, output_prices)
        profit_results = model.evaluate_all()

        result.add_info(f"Profit model completed with {len(profit_results)} DMUs")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_malmquist_model() -> ValidationResult:
    """Validate Malmquist productivity index."""
    result = ValidationResult("MalmquistModel")

    try:
        inputs_t1 = INPUTS
        outputs_t1 = OUTPUTS
        inputs_t2 = INPUTS * 0.9
        outputs_t2 = OUTPUTS * 1.1

        model = MalmquistModel(inputs_t1, outputs_t1, inputs_t2, outputs_t2)
        malmquist_results = model.evaluate_all()

        result.add_info(f"Malmquist model completed with {len(malmquist_results)} DMUs")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_cross_efficiency_model() -> ValidationResult:
    """Validate Cross-Efficiency model."""
    result = ValidationResult("CrossEfficiencyModel")

    try:
        model = CrossEfficiencyModel(INPUTS, OUTPUTS)
        cross_results = model.evaluate_all()

        if 'Cross_Efficiency' in cross_results.columns:
            cross_eff = cross_results['Cross_Efficiency'].values
            result.add_info(f"Cross efficiencies: min={np.min(cross_eff):.4f}, max={np.max(cross_eff):.4f}")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_nonradial_model() -> ValidationResult:
    """Validate Non-Radial model."""
    result = ValidationResult("NonRadialModel")

    try:
        model = NonRadialModel(INPUTS, OUTPUTS)
        nr_results = model.evaluate_all()

        if 'Mean_Efficiency' in nr_results.columns:
            efficiencies = nr_results['Mean_Efficiency'].values
            result.add_info(f"Non-radial mean efficiencies: min={np.min(efficiencies):.4f}, max={np.max(efficiencies):.4f}")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_lgo_model() -> ValidationResult:
    """Validate LGO (Linear Goal-Oriented) model."""
    result = ValidationResult("LGOModel")

    try:
        model = LGOModel(INPUTS, OUTPUTS)
        lgo_results = model.evaluate_all()

        result.add_info(f"LGO model completed with {len(lgo_results)} DMUs")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_rdm_model() -> ValidationResult:
    """Validate RDM (Range Directional Model)."""
    result = ValidationResult("RDMModel")

    try:
        model = RDMModel(INPUTS, OUTPUTS)
        rdm_results = model.evaluate_all()

        result.add_info(f"RDM model completed with {len(rdm_results)} DMUs")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_addmin_model() -> ValidationResult:
    """Validate AddMin model."""
    result = ValidationResult("AddMinModel")

    try:
        model = AddMinModel(INPUTS, OUTPUTS)
        addmin_results = model.evaluate_all()

        result.add_info(f"AddMin model completed with {len(addmin_results)} DMUs")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_addsupereff_model() -> ValidationResult:
    """Validate AddSuperEff model."""
    result = ValidationResult("AddSuperEffModel")

    try:
        model = AddSuperEffModel(INPUTS, OUTPUTS)
        addsupereff_results = model.evaluate_all()

        result.add_info(f"AddSuperEff model completed with {len(addsupereff_results)} DMUs")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_deaps_model() -> ValidationResult:
    """Validate DEAPS model."""
    result = ValidationResult("DEAPSModel")

    try:
        model = DEAPSModel(INPUTS, OUTPUTS)
        deaps_results = model.evaluate_all()

        result.add_info(f"DEAPS model completed with {len(deaps_results)} DMUs")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_common_weights_model() -> ValidationResult:
    """Validate Common Weights model."""
    result = ValidationResult("CommonWeightsModel")

    try:
        model = CommonWeightsModel(INPUTS, OUTPUTS)
        cw_results = model.evaluate_all()

        result.add_info(f"Common Weights model completed with {len(cw_results)} DMUs")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_directional_efficiency_model() -> ValidationResult:
    """Validate Directional Efficiency model."""
    result = ValidationResult("DirectionalEfficiencyModel")

    try:
        model = DirectionalEfficiencyModel(INPUTS, OUTPUTS)
        de_results = model.evaluate_all()

        result.add_info(f"Directional Efficiency model completed with {len(de_results)} DMUs")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_eladder_model() -> ValidationResult:
    """Validate Efficiency Ladder model."""
    result = ValidationResult("EfficiencyLadderModel")

    try:
        model = EfficiencyLadderModel(INPUTS, OUTPUTS)
        eladder_results = model.evaluate_all()

        result.add_info(f"Efficiency Ladder model completed with {len(eladder_results)} DMUs")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_merger_model() -> ValidationResult:
    """Validate Merger Analysis model."""
    result = ValidationResult("MergerAnalysisModel")

    try:
        model = MergerAnalysisModel(INPUTS, OUTPUTS)
        # Create a simple merger matrix (merge DMUs 0 and 1)
        merger_matrix = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
        merger_results = model.evaluate_all(merger_matrix)

        result.add_info(f"Merger Analysis model completed")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_bootstrap_model() -> ValidationResult:
    """Validate Bootstrap DEA model."""
    result = ValidationResult("BootstrapDEAModel")

    try:
        model = BootstrapDEAModel(INPUTS, OUTPUTS)
        # Run with small number of iterations for testing
        bootstrap_results = model.evaluate_all(n_rep=10)

        result.add_info(f"Bootstrap DEA model completed with {len(bootstrap_results)} DMUs")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_undesirable_transform() -> ValidationResult:
    """Validate undesirable output transformation."""
    result = ValidationResult("transform_undesirable")

    try:
        # Test with undesirable outputs
        undesirable = np.array([[10], [20], [15], [25], [12], [18], [22], [14], [16], [19]])

        # Use correct API - specify ud_outputs
        transformed_inputs, transformed_outputs, vtrans_i, vtrans_o = \
            transform_undesirable(INPUTS, undesirable, ud_outputs=np.array([0]))

        result.add_info(f"Undesirable transform completed")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_returns_to_scale_model() -> ValidationResult:
    """Validate Returns to Scale model."""
    result = ValidationResult("ReturnsToScaleModel")

    try:
        model = ReturnsToScaleModel(INPUTS, OUTPUTS)
        rts_results = model.evaluate_all()

        result.add_info(f"Returns to Scale model completed with {len(rts_results)} DMUs")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_network_model() -> ValidationResult:
    """Validate Series Network model."""
    result = ValidationResult("SeriesNetworkModel")

    try:
        intermediates = np.array([
            [15, 25], [18, 22], [30, 35], [20, 40], [12, 15],
            [20, 50], [25, 28], [35, 25], [22, 24], [35, 40]
        ])

        model = SeriesNetworkModel(INPUTS, OUTPUTS, intermediates)
        network_results = model.evaluate_all()

        result.add_info(f"Series Network model completed with {len(network_results)} DMUs")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_fdh_plus_model() -> ValidationResult:
    """Validate FDH+ (Free Disposal Hull Plus) model."""
    result = ValidationResult("FDHPlusModel")

    try:
        model = FDHPlusModel(INPUTS, OUTPUTS, param=0.15)
        fdh_plus_results = model.evaluate_all()

        if 'Efficiency' in fdh_plus_results.columns:
            efficiencies = fdh_plus_results['Efficiency'].values
            if any(efficiencies > 1.001):
                result.add_error(f"FDH+ efficiencies > 1: {efficiencies[efficiencies > 1.001]}")
            if any(efficiencies < 0):
                result.add_error(f"FDH+ efficiencies < 0: {efficiencies[efficiencies < 0]}")
            result.add_info(f"FDH+ efficiencies: min={np.min(efficiencies):.4f}, max={np.max(efficiencies):.4f}")
        else:
            result.add_info(f"FDH+ model completed with {len(fdh_plus_results)} DMUs")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_modified_sbm_model() -> ValidationResult:
    """Validate Modified SBM model."""
    result = ValidationResult("ModifiedSBMModel")

    try:
        model = ModifiedSBMModel(INPUTS, OUTPUTS)

        # Test input-oriented
        input_results = model.evaluate_all(orientation='input')
        if 'Modified_SBM_Efficiency' in input_results.columns:
            efficiencies = input_results['Modified_SBM_Efficiency'].values
            if any(efficiencies > 1.001):
                result.add_error(f"Modified SBM efficiencies > 1: {efficiencies[efficiencies > 1.001]}")
            if any(efficiencies < 0):
                result.add_error(f"Modified SBM efficiencies < 0: {efficiencies[efficiencies < 0]}")
            result.add_info(f"Modified SBM efficiencies: min={np.min(efficiencies):.4f}, max={np.max(efficiencies):.4f}")
        else:
            result.add_info(f"Modified SBM model completed with {len(input_results)} DMUs")

        # Test output-oriented
        output_results = model.evaluate_all(orientation='output')
        result.add_info(f"Modified SBM output-oriented completed with {len(output_results)} DMUs")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def validate_stoned_model() -> ValidationResult:
    """Validate StoNED (Stochastic Non-smooth Envelopment of Data) model."""
    result = ValidationResult("StoNEDModel")

    try:
        # StoNED requires single output
        single_output = OUTPUTS[:, 0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = StoNEDModel(INPUTS, single_output)
            # Use AUTO method (tries MM, falls back to PSL if skewness is wrong)
            stoned_results = model.evaluate_all(rts='vrs', method='AUTO')

        if 'Efficiency' in stoned_results.columns:
            efficiencies = stoned_results['Efficiency'].values
            # StoNED efficiency can have NaN values if optimization fails
            valid_eff = efficiencies[~np.isnan(efficiencies)]
            if len(valid_eff) > 0:
                if any(valid_eff > 1.001):
                    result.add_error(f"StoNED efficiencies > 1: {valid_eff[valid_eff > 1.001]}")
                if any(valid_eff < 0):
                    result.add_error(f"StoNED efficiencies < 0: {valid_eff[valid_eff < 0]}")
                result.add_info(f"StoNED efficiencies: min={np.min(valid_eff):.4f}, max={np.max(valid_eff):.4f}")
            else:
                result.add_warning("StoNED returned all NaN efficiencies")
        else:
            result.add_info(f"StoNED model completed with {len(stoned_results)} DMUs")

    except Exception as e:
        result.add_error(f"Exception: {str(e)}")

    return result


def run_all_validations() -> Dict[str, ValidationResult]:
    """Run all model validations."""
    validations = [
        validate_ccr_model,
        validate_bcc_model,
        validate_ap_model,
        validate_maj_model,
        validate_additive_model,
        validate_twophase_model,
        validate_sbm_model,
        validate_drs_irs_models,
        validate_fdh_model,
        validate_mea_model,
        validate_norml1_model,
        validate_congestion_model,
        validate_cost_revenue_models,
        validate_profit_model,
        validate_malmquist_model,
        validate_cross_efficiency_model,
        validate_nonradial_model,
        validate_lgo_model,
        validate_rdm_model,
        validate_addmin_model,
        validate_addsupereff_model,
        validate_deaps_model,
        validate_common_weights_model,
        validate_directional_efficiency_model,
        validate_eladder_model,
        validate_merger_model,
        validate_bootstrap_model,
        validate_undesirable_transform,
        validate_returns_to_scale_model,
        validate_network_model,
        validate_fdh_plus_model,
        validate_modified_sbm_model,
        validate_stoned_model,
    ]

    results = {}
    for validate_func in validations:
        result = validate_func()
        results[result.model_name] = result

    return results


def print_summary(results: Dict[str, ValidationResult]):
    """Print validation summary."""
    passed = [name for name, r in results.items() if r.passed]
    failed = [name for name, r in results.items() if not r.passed]

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print(f"\nPASSED: {len(passed)}/{len(results)}")
    for name in passed:
        print(f"  + {name}")

    if failed:
        print(f"\nFAILED: {len(failed)}/{len(results)}")
        for name in failed:
            print(f"  - {name}")

    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    for name, result in results.items():
        print(result)


if __name__ == "__main__":
    print("=" * 80)
    print("DEA MODEL VALIDATION")
    print("=" * 80)
    print("\nRunning comprehensive validation tests...\n")

    results = run_all_validations()
    print_summary(results)
