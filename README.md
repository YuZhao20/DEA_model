# DEA Model Library

Data Envelopment Analysis (DEA) implementation in Python based on Hosseinzadeh Lotfi et al. (2020), Chapter 3: Basic DEA Models.

## 概要

このライブラリは、Hosseinzadeh Lotfi et al. (2020)の第3章「Basic DEA Models」に基づいて実装されたDEA（Data Envelopment Analysis）モデルです。

## 実装されているモデル

### 第3章: Basic DEA Models

#### CCR (Charnes-Cooper-Rhodes) モデル
- **Input-Oriented CCR Envelopment Model** (3.2.1)
- **Input-Oriented CCR Multiplier Model** (3.2.2)
- **Output-Oriented CCR Envelopment Model** (3.3.1)
- **Output-Oriented CCR Multiplier Model** (3.3.2)

#### BCC (Banker-Charnes-Cooper) モデル
- **Input-Oriented BCC Envelopment Model** (3.2.4)
- **Input-Oriented BCC Multiplier Model** (3.2.3)
- **Output-Oriented BCC Envelopment Model** (3.3.4)
- **Output-Oriented BCC Multiplier Model** (3.3.3)

#### Additive Models
- **Additive CCR Model** (3.4.1)
- **Additive BCC Model** (3.4.2)

#### Epsilon-based Multiplier Models
- **Input-Oriented BCC Multiplier Model with epsilon** (3.5.1)
- **Input-Oriented CCR Multiplier Model with epsilon** (3.5.2)

#### Two-Phase Models
- **Two-Phase Input-Oriented BCC Envelopment Model** (3.6.1)
- **Two-Phase Input-Oriented CCR Envelopment Model** (3.6.2)

### 第4章: Advanced DEA Models

#### AP (Anderson-Peterson) スーパー効率モデル
- **Input-Oriented AP Envelopment Model** (4.2.1)
- **Output-Oriented AP Envelopment Model** (4.2.2)
- **Input-Oriented AP Multiplier Model** (4.2.3)
- **Output-Oriented AP Multiplier Model** (4.2.4)

#### MAJ Super-Efficiency Model
- **MAJ Super-Efficiency Model** (4.3)

#### Norm L1 Super-Efficiency Model
- **Norm L1 Super-Efficiency Model** (4.4)

#### Returns to Scale Models
- **Returns to Scale - CCR Envelopment Model** (4.5.1)
- **Returns to Scale - DEA Multiplier Model** (4.5.2)

#### Cost and Revenue Efficiency Models
- **Cost Efficiency Model** (4.6)
- **Revenue Efficiency Model** (4.7)

#### Malmquist Productivity Index
- **Malmquist Productivity Index - CCR Multiplier Model** (4.8.1)
- **Malmquist Productivity Index - CCR Envelopment Model** (4.8.2)

#### SBM (Slacks-Based Measure) Models
- **First Model of SBM** (4.9.1)
- **Second Model of SBM** (4.9.2)

#### Modified SBM Models
- **Input-Oriented Modified SBM Model** (4.12.1)
- **Output-Oriented Modified SBM Model** (4.12.2)

#### Network DEA Models
- **Series Network DEA Model** (4.10)

#### Other Advanced Models
- **Congestion DEA Model** (4.13)
- **Common Set of Weights DEA Model** (4.14)
- **Directional Efficiency DEA Model** (4.15)
- **Profit Efficiency Model** (4.11)

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本的な使用例

```python
import numpy as np
from dea import (
    CCRModel, BCCModel, APModel, MAJModel,
    AdditiveModel, TwoPhaseModel,
    NormL1Model, ReturnsToScaleModel,
    CostEfficiencyModel, RevenueEfficiencyModel,
    MalmquistModel, SBMModel,
    ProfitEfficiencyModel, ModifiedSBMModel,
    SeriesNetworkModel,
    CongestionModel, CommonWeightsModel, DirectionalEfficiencyModel
)

# データの準備（入力と出力）
inputs = np.array([
    [20, 11],  # DMU 1
    [11, 40],  # DMU 2
    [32, 30],  # DMU 3
    # ... 他のDMU
])

outputs = np.array([
    [8, 30],   # DMU 1
    [21, 20],  # DMU 2
    [34, 40],  # DMU 3
    # ... 他のDMU
])

# CCRモデルの使用
ccr_model = CCRModel(inputs, outputs)

# Envelopment形式で全DMUを評価
results = ccr_model.evaluate_all(method='envelopment')
print(results)

# 特定のDMUを評価
efficiency, lambdas, input_targets, output_targets = ccr_model.solve_envelopment(0)
print(f"Efficiency: {efficiency}")

# Multiplier形式で評価
results_mult = ccr_model.evaluate_all(method='multiplier')
print(results_mult)

# BCCモデルの使用
bcc_model = BCCModel(inputs, outputs)
bcc_results = bcc_model.evaluate_all(method='envelopment')
print(bcc_results)

# AP (Anderson-Peterson) スーパー効率モデルの使用
ap_model = APModel(inputs, outputs)

# Input-Oriented AP Envelopment Model
ap_input_results = ap_model.evaluate_all(orientation='input', method='envelopment')
print(ap_input_results)

# Output-Oriented AP Envelopment Model
ap_output_results = ap_model.evaluate_all(orientation='output', method='envelopment')
print(ap_output_results)

# MAJ Super-Efficiency Model
maj_model = MAJModel(inputs, outputs)
maj_results = maj_model.evaluate_all()
print(maj_results)
```

### テスト実行

PDFのTable 3.1のデータを使用したテストスクリプトが含まれています：

```bash
python test_dea.py
```

## データ形式

- **inputs**: 形状 `(n_dmus, n_inputs)` のNumPy配列
- **outputs**: 形状 `(n_dmus, n_outputs)` のNumPy配列

## 参考文献

Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Studies in Fuzziness and Soft Computing, Vol. 386. Springer.

## ライセンス

LICENSEファイルを参照してください。
