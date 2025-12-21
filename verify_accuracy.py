"""
DEA Model Accuracy Verification Script
Compares calculated results with known literature values and theoretical properties.
"""

import numpy as np
import pandas as pd
from dea import (
    CCRModel, BCCModel, APModel, MAJModel, SBMModel,
    DRSModel, IRSModel, FDHModel, AdditiveModel, TwoPhaseModel,
    CostEfficiencyModel, RevenueEfficiencyModel
)

# Standard test data from Table 3.1 (Hosseinzadeh Lotfi et al., 2020)
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

INPUTS = DATA[:, :2]
OUTPUTS = DATA[:, 2:]

print("=" * 80)
print("DEA MODEL ACCURACY VERIFICATION")
print("=" * 80)

# ============================================================================
# 1. CCR Model Verification
# ============================================================================
print("\n" + "=" * 40)
print("1. CCR MODEL VERIFICATION")
print("=" * 40)

ccr = CCRModel(INPUTS, OUTPUTS)
ccr_results = ccr.evaluate_all(method='envelopment')
ccr_eff = ccr_results['Efficiency'].values

print("\nCCR Efficiencies (Input-Oriented):")
for i, eff in enumerate(ccr_eff):
    status = "Efficient" if eff >= 0.999 else "Inefficient"
    print(f"  DMU {i+1}: {eff:.4f} ({status})")

# Theoretical check: CCR efficient DMUs
ccr_efficient = np.sum(ccr_eff >= 0.999)
print(f"\nNumber of CCR-efficient DMUs: {ccr_efficient}")
print("Expected efficient DMUs: 6, 7, 10 (based on standard DEA literature)")

# Verify CCR multiplier vs envelopment consistency
ccr_mult = ccr.evaluate_all(method='multiplier')
max_diff = np.max(np.abs(ccr_eff - ccr_mult['Efficiency'].values))
print(f"Max difference (Envelopment vs Multiplier): {max_diff:.6f}")
if max_diff < 0.01:
    print("  ✓ PASS: Envelopment and Multiplier results are consistent")
else:
    print("  ✗ FAIL: Large discrepancy between methods")

# ============================================================================
# 2. BCC Model Verification
# ============================================================================
print("\n" + "=" * 40)
print("2. BCC MODEL VERIFICATION")
print("=" * 40)

bcc = BCCModel(INPUTS, OUTPUTS)
bcc_results = bcc.evaluate_all(method='envelopment')
bcc_eff = bcc_results['Efficiency'].values

print("\nBCC Efficiencies (Input-Oriented):")
for i, eff in enumerate(bcc_eff):
    status = "Efficient" if eff >= 0.999 else "Inefficient"
    print(f"  DMU {i+1}: {eff:.4f} ({status})")

# Theoretical check: BCC >= CCR
bcc_ge_ccr = np.all(bcc_eff >= ccr_eff - 0.001)
print(f"\nBCC >= CCR (theoretical property): {'✓ PASS' if bcc_ge_ccr else '✗ FAIL'}")

bcc_efficient = np.sum(bcc_eff >= 0.999)
print(f"Number of BCC-efficient DMUs: {bcc_efficient}")
print(f"(Should be >= CCR-efficient: {ccr_efficient})")

# ============================================================================
# 3. DRS/IRS Model Verification
# ============================================================================
print("\n" + "=" * 40)
print("3. DRS/IRS MODEL VERIFICATION")
print("=" * 40)

drs = DRSModel(INPUTS, OUTPUTS)
irs = IRSModel(INPUTS, OUTPUTS)

drs_results = drs.evaluate_all(orientation='input')
irs_results = irs.evaluate_all(orientation='input')

drs_eff = drs_results['Efficiency'].values
irs_eff = irs_results['Efficiency'].values

print("\nDRS vs IRS vs CCR (first 5 DMUs):")
print("DMU   CCR      DRS      IRS")
print("-" * 35)
for i in range(5):
    print(f"{i+1:<5} {ccr_eff[i]:<8.4f} {drs_eff[i]:<8.4f} {irs_eff[i]:<8.4f}")

# Theoretical: CCR <= DRS (input-oriented), CCR <= IRS (input-oriented)
ccr_le_drs = np.all(ccr_eff <= drs_eff + 0.001)
ccr_le_irs = np.all(ccr_eff <= irs_eff + 0.001)
print(f"\nCCR <= DRS: {'✓ PASS' if ccr_le_drs else '✗ FAIL'}")
print(f"CCR <= IRS: {'✓ PASS' if ccr_le_irs else '✗ FAIL'}")

# ============================================================================
# 4. AP Super-Efficiency Verification
# ============================================================================
print("\n" + "=" * 40)
print("4. AP SUPER-EFFICIENCY VERIFICATION")
print("=" * 40)

ap = APModel(INPUTS, OUTPUTS)
ap_results = ap.evaluate_all(orientation='input', method='envelopment')
ap_eff = ap_results['Super_Efficiency'].values

print("\nAP Super-Efficiencies:")
for i, eff in enumerate(ap_eff):
    ccr_status = "CCR-Efficient" if ccr_eff[i] >= 0.999 else "CCR-Inefficient"
    print(f"  DMU {i+1}: {eff:.4f} ({ccr_status})")

# Theoretical: For CCR-inefficient DMUs, AP = CCR
errors = []
for i in range(len(ap_eff)):
    if ccr_eff[i] < 0.999:  # Inefficient
        if abs(ap_eff[i] - ccr_eff[i]) > 0.01:
            errors.append(f"DMU {i+1}: AP={ap_eff[i]:.4f}, CCR={ccr_eff[i]:.4f}")

if not errors:
    print("\n✓ PASS: AP = CCR for all inefficient DMUs")
else:
    print("\n✗ FAIL: AP != CCR for some inefficient DMUs:")
    for err in errors:
        print(f"  {err}")

# For efficient DMUs, AP > 1 (super-efficiency)
super_eff_check = True
for i in range(len(ap_eff)):
    if ccr_eff[i] >= 0.999 and ap_eff[i] < 1.0:
        print(f"  Warning: DMU {i+1} is CCR-efficient but AP < 1")
        super_eff_check = False

if super_eff_check:
    print("✓ PASS: All CCR-efficient DMUs have AP >= 1")

# ============================================================================
# 5. SBM Model Verification
# ============================================================================
print("\n" + "=" * 40)
print("5. SBM MODEL VERIFICATION")
print("=" * 40)

sbm = SBMModel(INPUTS, OUTPUTS)
# Use CRS for comparison with CCR (which also uses CRS)
sbm_results = sbm.evaluate_all(model_type=1, rts='crs')
sbm_eff = sbm_results['SBM_Efficiency'].values

print("\nSBM Efficiencies (CRS):")
for i, eff in enumerate(sbm_eff):
    print(f"  DMU {i+1}: {eff:.4f}")

# Theoretical: SBM-CRS <= CCR-CRS (SBM is non-radial, always <= radial)
sbm_le_ccr = np.all(sbm_eff <= ccr_eff + 0.001)
print(f"\nSBM-CRS <= CCR-CRS: {'✓ PASS' if sbm_le_ccr else '✗ FAIL'}")

# ============================================================================
# 6. FDH Model Verification
# ============================================================================
print("\n" + "=" * 40)
print("6. FDH MODEL VERIFICATION")
print("=" * 40)

fdh = FDHModel(INPUTS, OUTPUTS)
fdh_results = fdh.evaluate_all(orientation='input')
fdh_eff = fdh_results['Efficiency'].values

print("\nFDH Efficiencies:")
for i, eff in enumerate(fdh_eff):
    print(f"  DMU {i+1}: {eff:.4f}")

# Theoretical: FDH >= BCC (FDH is less restrictive)
fdh_ge_bcc = np.all(fdh_eff >= bcc_eff - 0.001)
print(f"\nFDH >= BCC: {'✓ PASS' if fdh_ge_bcc else '✗ FAIL'}")

fdh_efficient = np.sum(fdh_eff >= 0.999)
print(f"Number of FDH-efficient DMUs: {fdh_efficient}")

# ============================================================================
# 7. Cost/Revenue Efficiency Verification
# ============================================================================
print("\n" + "=" * 40)
print("7. COST/REVENUE EFFICIENCY VERIFICATION")
print("=" * 40)

input_prices = np.array([1.0, 2.0])
output_prices = np.array([3.0, 4.0])

cost_model = CostEfficiencyModel(INPUTS, OUTPUTS, input_prices)
cost_results = cost_model.evaluate_all()

revenue_model = RevenueEfficiencyModel(INPUTS, OUTPUTS, output_prices)
revenue_results = revenue_model.evaluate_all()

cost_eff = cost_results['Cost_Efficiency'].values
rev_eff = revenue_results['Revenue_Efficiency'].values

print("\nCost and Revenue Efficiencies (first 5 DMUs):")
print("DMU   Cost_Eff   Revenue_Eff")
print("-" * 30)
for i in range(5):
    print(f"{i+1:<5} {cost_eff[i]:<10.4f} {rev_eff[i]:<10.4f}")

# Theoretical: Cost/Revenue efficiency <= 1
cost_bound = np.all(cost_eff <= 1.001)
rev_bound = np.all(rev_eff <= 1.001)
print(f"\nCost Efficiency <= 1: {'✓ PASS' if cost_bound else '✗ FAIL'}")
print(f"Revenue Efficiency <= 1: {'✓ PASS' if rev_bound else '✗ FAIL'}")

# ============================================================================
# 8. Two-Phase Model Verification
# ============================================================================
print("\n" + "=" * 40)
print("8. TWO-PHASE MODEL VERIFICATION")
print("=" * 40)

twophase = TwoPhaseModel(INPUTS, OUTPUTS)
twophase_results = twophase.evaluate_all()
twophase_eff = twophase_results['Efficiency'].values

print("\nTwo-Phase Efficiencies (should equal BCC):")
for i in range(5):
    diff = abs(twophase_eff[i] - bcc_eff[i])
    status = "✓" if diff < 0.001 else "✗"
    print(f"  DMU {i+1}: TwoPhase={twophase_eff[i]:.4f}, BCC={bcc_eff[i]:.4f} {status}")

twophase_eq_bcc = np.allclose(twophase_eff, bcc_eff, atol=0.01)
print(f"\nTwo-Phase ≈ BCC: {'✓ PASS' if twophase_eq_bcc else '✗ FAIL'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ACCURACY VERIFICATION SUMMARY")
print("=" * 80)

tests = [
    ("CCR Envelopment/Multiplier Consistency", max_diff < 0.01),
    ("BCC >= CCR", bcc_ge_ccr),
    ("CCR <= DRS", ccr_le_drs),
    ("CCR <= IRS", ccr_le_irs),
    ("SBM-CRS <= CCR-CRS", sbm_le_ccr),
    ("FDH >= BCC", fdh_ge_bcc),
    ("Cost Efficiency <= 1", cost_bound),
    ("Revenue Efficiency <= 1", rev_bound),
    ("Two-Phase ≈ BCC", twophase_eq_bcc),
]

passed = sum(1 for _, result in tests if result)
total = len(tests)

print(f"\nPassed: {passed}/{total}")
print()
for name, result in tests:
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"  {status}: {name}")

if passed == total:
    print("\n✓ ALL THEORETICAL PROPERTIES VERIFIED")
else:
    print(f"\n✗ {total - passed} TEST(S) FAILED - REVIEW REQUIRED")
