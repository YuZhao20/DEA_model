"""
Test script for DEA models using the example from Table 3.1 and Table 4.1
"""

import numpy as np
import pandas as pd
from dea import CCRModel, BCCModel, APModel, MAJModel

# Data from Table 3.1
data = np.array([
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

inputs = data[:, :2]  # First 2 columns are inputs
outputs = data[:, 2:]  # Last 2 columns are outputs

print("=" * 80)
print("Input-Oriented CCR Envelopment Model Results")
print("=" * 80)
ccr_model = CCRModel(inputs, outputs)
ccr_results = ccr_model.evaluate_all(method='envelopment')
print(ccr_results.to_string(index=False))
print("\n")

print("=" * 80)
print("Input-Oriented CCR Multiplier Model Results")
print("=" * 80)
ccr_mult_results = ccr_model.evaluate_all(method='multiplier')
print(ccr_mult_results.to_string(index=False))
print("\n")

print("=" * 80)
print("Input-Oriented BCC Envelopment Model Results")
print("=" * 80)
bcc_model = BCCModel(inputs, outputs)
bcc_results = bcc_model.evaluate_all(method='envelopment')
print(bcc_results.to_string(index=False))
print("\n")

print("=" * 80)
print("Input-Oriented BCC Multiplier Model Results")
print("=" * 80)
bcc_mult_results = bcc_model.evaluate_all(method='multiplier')
print(bcc_mult_results.to_string(index=False))
print("\n")

print("=" * 80)
print("Chapter 4: Advanced DEA Models")
print("=" * 80)
print("\n")

print("=" * 80)
print("Input-Oriented AP Envelopment Model Results (Super-Efficiency)")
print("=" * 80)
ap_model = APModel(inputs, outputs)
# APModelのsolve_input_oriented_envelopmentは(efficiency, lambdas)を返す
eff, lambdas = ap_model.solve_input_oriented_envelopment(0)
print(f"DMU 1 AP Super-Efficiency: {eff:.4f}")
ap_input_env_results = ap_model.evaluate_all(orientation='input', method='envelopment')
print(ap_input_env_results.to_string(index=False))
print("\n")

print("=" * 80)
print("Output-Oriented AP Envelopment Model Results (Super-Efficiency)")
print("=" * 80)
ap_output_env_results = ap_model.evaluate_all(orientation='output', method='envelopment')
print(ap_output_env_results.to_string(index=False))
print("\n")

print("=" * 80)
print("Input-Oriented AP Multiplier Model Results (Super-Efficiency)")
print("=" * 80)
ap_input_mult_results = ap_model.evaluate_all(orientation='input', method='multiplier')
print(ap_input_mult_results.to_string(index=False))
print("\n")

print("=" * 80)
print("MAJ Super-Efficiency Model Results")
print("=" * 80)
maj_model = MAJModel(inputs, outputs)
maj_results = maj_model.evaluate_all()
print(maj_results.to_string(index=False))

