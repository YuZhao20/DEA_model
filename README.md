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

### データの準備

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
    [21, 30],  # DMU 4
    [20, 11],  # DMU 5
    [12, 43],  # DMU 6
    [7, 45],   # DMU 7
    [31, 45],  # DMU 8
    [19, 22],  # DMU 9
    [32, 11],  # DMU 10
])

outputs = np.array([
    [8, 30],   # DMU 1
    [21, 20],  # DMU 2
    [34, 40],  # DMU 3
    [18, 50],  # DMU 4
    [6, 17],   # DMU 5
    [23, 58],  # DMU 6
    [28, 30],  # DMU 7
    [40, 20],  # DMU 8
    [27, 23],  # DMU 9
    [38, 45],  # DMU 10
])
```

### 第3章: Basic DEA Models

#### 1. CCR (Charnes-Cooper-Rhodes) モデル

```python
# CCRモデルの初期化
ccr_model = CCRModel(inputs, outputs)

# Input-Oriented CCR Envelopment Model (3.2.1)
results = ccr_model.evaluate_all(method='envelopment')
print("CCR Envelopment Results:")
print(results)

# 特定のDMUを評価
efficiency, lambdas, input_targets, output_targets = ccr_model.solve_envelopment(0)
print(f"DMU 1 Efficiency: {efficiency:.4f}")

# Input-Oriented CCR Multiplier Model (3.2.2)
results_mult = ccr_model.evaluate_all(method='multiplier')
print("CCR Multiplier Results:")
print(results_mult)

# Output-Oriented CCR Envelopment Model (3.3.1)
eff_out, lambdas_out, input_targets_out, output_targets_out = ccr_model.solve_output_oriented_envelopment(0)
print(f"Output-Oriented Efficiency: {eff_out:.4f}")

# Output-Oriented CCR Multiplier Model (3.3.2)
eff_out_mult, v_out, u_out = ccr_model.solve_output_oriented_multiplier(0)
print(f"Output-Oriented Multiplier Efficiency: {eff_out_mult:.4f}")
```

#### 2. BCC (Banker-Charnes-Cooper) モデル

```python
# BCCモデルの初期化
bcc_model = BCCModel(inputs, outputs)

# Input-Oriented BCC Envelopment Model (3.2.4)
bcc_results = bcc_model.evaluate_all(method='envelopment')
print("BCC Envelopment Results:")
print(bcc_results)

# Input-Oriented BCC Multiplier Model (3.2.3)
bcc_mult_results = bcc_model.evaluate_all(method='multiplier')
print("BCC Multiplier Results:")
print(bcc_mult_results)

# Output-Oriented BCC Envelopment Model (3.3.4)
eff_bcc_out, lambdas_bcc_out, _, _ = bcc_model.solve_output_oriented_envelopment(0)
print(f"Output-Oriented BCC Efficiency: {eff_bcc_out:.4f}")

# Output-Oriented BCC Multiplier Model (3.3.3)
eff_bcc_out_mult, v_bcc_out, u_bcc_out, u0_bcc_out = bcc_model.solve_output_oriented_multiplier(0)
print(f"Output-Oriented BCC Multiplier Efficiency: {eff_bcc_out_mult:.4f}")
print(f"u0 value: {u0_bcc_out:.4f}")
```

#### 3. Additive Models

```python
# Additiveモデルの初期化
additive_model = AdditiveModel(inputs, outputs)

# Additive CCR Model (3.4.1)
additive_ccr_results = additive_model.evaluate_all(model_type='ccr')
print("Additive CCR Results:")
print(additive_ccr_results)

# Additive BCC Model (3.4.2)
additive_bcc_results = additive_model.evaluate_all(model_type='bcc')
print("Additive BCC Results:")
print(additive_bcc_results)

# 特定のDMUを評価
sum_slacks, lambdas, input_slacks, output_slacks = additive_model.solve_ccr(0)
print(f"Sum of Slacks: {sum_slacks:.4f}")
```

#### 4. Epsilon-based Multiplier Models

```python
# Epsilon版のMultiplierモデル（3.5.1, 3.5.2）
# BCC Multiplier Model with epsilon (3.5.1)
eff_bcc_eps, v_bcc_eps, u_bcc_eps, u0_bcc_eps = bcc_model.solve_multiplier(0, epsilon=1e-4)
print(f"BCC Multiplier with epsilon: {eff_bcc_eps:.4f}")

# CCR Multiplier Model with epsilon (3.5.2)
eff_ccr_eps, v_ccr_eps, u_ccr_eps, _ = ccr_model.solve_multiplier(0, epsilon=1e-4)
print(f"CCR Multiplier with epsilon: {eff_ccr_eps:.4f}")
```

#### 5. Two-Phase Models

```python
# Two-Phaseモデルの初期化
twophase_model = TwoPhaseModel(inputs, outputs)

# Two-Phase Input-Oriented BCC Envelopment Model (3.6.1)
twophase_bcc_results = twophase_model.evaluate_all()
print("Two-Phase BCC Results:")
print(twophase_bcc_results)

# Two-Phase Input-Oriented CCR Envelopment Model (3.6.2)
twophase_ccr_results = twophase_model.evaluate_all_ccr()
print("Two-Phase CCR Results:")
print(twophase_ccr_results)

# 特定のDMUを評価
eff_2p, lambdas_2p, input_slacks_2p, output_slacks_2p, sum_slacks_2p = twophase_model.solve(0)
print(f"Two-Phase Efficiency: {eff_2p:.4f}, Sum of Slacks: {sum_slacks_2p:.4f}")
```

### 第4章: Advanced DEA Models

#### 6. AP (Anderson-Peterson) スーパー効率モデル

```python
# APモデルの初期化
ap_model = APModel(inputs, outputs)

# Input-Oriented AP Envelopment Model (4.2.1)
ap_input_env = ap_model.evaluate_all(orientation='input', method='envelopment')
print("AP Input-Oriented Envelopment Results:")
print(ap_input_env)

# Output-Oriented AP Envelopment Model (4.2.2)
ap_output_env = ap_model.evaluate_all(orientation='output', method='envelopment')
print("AP Output-Oriented Envelopment Results:")
print(ap_output_env)

# Input-Oriented AP Multiplier Model (4.2.3)
ap_input_mult = ap_model.evaluate_all(orientation='input', method='multiplier')
print("AP Input-Oriented Multiplier Results:")
print(ap_input_mult)

# Output-Oriented AP Multiplier Model (4.2.4)
ap_output_mult = ap_model.evaluate_all(orientation='output', method='multiplier')
print("AP Output-Oriented Multiplier Results:")
print(ap_output_mult)
```

#### 7. MAJ Super-Efficiency Model

```python
# MAJモデルの初期化
maj_model = MAJModel(inputs, outputs)

# MAJ Super-Efficiency Model (4.3)
maj_results = maj_model.evaluate_all()
print("MAJ Super-Efficiency Results:")
print(maj_results)

# 特定のDMUを評価
w_star, super_eff_maj = maj_model.solve(0)
print(f"W*: {w_star:.4f}, Super-Efficiency: {super_eff_maj:.4f}")
```

#### 8. Norm L1 Super-Efficiency Model

```python
# Norm L1モデルの初期化
norml1_model = NormL1Model(inputs, outputs)

# Norm L1 Super-Efficiency Model (4.4)
norml1_results = norml1_model.evaluate_all()
print("Norm L1 Super-Efficiency Results:")
print(norml1_results)
```

#### 9. Returns to Scale Models

```python
# Returns to Scaleモデルの初期化
rts_model = ReturnsToScaleModel(inputs, outputs)

# Returns to Scale - CCR Envelopment Model (4.5.1)
rts_results = rts_model.evaluate_all(method='envelopment')
print("Returns to Scale (Envelopment) Results:")
print(rts_results)

# Returns to Scale - DEA Multiplier Model (4.5.2)
rts_mult_results = rts_model.evaluate_all(method='multiplier')
print("Returns to Scale (Multiplier) Results:")
print(rts_mult_results)
```

#### 10. Cost and Revenue Efficiency Models

```python
# コストと価格のデータを準備
input_costs = np.array([1.0, 2.0])  # 各入力のコスト
output_prices = np.array([3.0, 4.0])  # 各出力の価格

# Cost Efficiency Model (4.6)
cost_model = CostEfficiencyModel(inputs, outputs, input_costs)
cost_results = cost_model.evaluate_all()
print("Cost Efficiency Results:")
print(cost_results)

# Revenue Efficiency Model (4.7)
revenue_model = RevenueEfficiencyModel(inputs, outputs, output_prices)
revenue_results = revenue_model.evaluate_all()
print("Revenue Efficiency Results:")
print(revenue_results)
```

#### 11. Malmquist Productivity Index

```python
# 2時点のデータを準備
inputs_t = inputs  # 時点tの入力
outputs_t = outputs  # 時点tの出力
inputs_t1 = inputs * 1.1  # 時点t+1の入力（例：10%増加）
outputs_t1 = outputs * 1.15  # 時点t+1の出力（例：15%増加）

# Malmquistモデルの初期化
malmquist_model = MalmquistModel(inputs_t, outputs_t, inputs_t1, outputs_t1)

# Malmquist Productivity Index (4.8.1, 4.8.2)
malmquist_results = malmquist_model.evaluate_all()
print("Malmquist Productivity Index Results:")
print(malmquist_results)
```

#### 12. SBM (Slacks-Based Measure) Models

```python
# SBMモデルの初期化
sbm_model = SBMModel(inputs, outputs)

# First Model of SBM (4.9.1)
sbm_results_1 = sbm_model.evaluate_all(model_type=1)
print("SBM Model 1 Results:")
print(sbm_results_1)

# Second Model of SBM (4.9.2)
sbm_results_2 = sbm_model.evaluate_all(model_type=2)
print("SBM Model 2 Results:")
print(sbm_results_2)
```

#### 13. Modified SBM Models

```python
# Modified SBMモデルの初期化
modified_sbm_model = ModifiedSBMModel(inputs, outputs)

# Input-Oriented Modified SBM Model (4.12.1)
modified_sbm_input = modified_sbm_model.evaluate_all(orientation='input')
print("Modified SBM Input-Oriented Results:")
print(modified_sbm_input)

# Output-Oriented Modified SBM Model (4.12.2)
modified_sbm_output = modified_sbm_model.evaluate_all(orientation='output')
print("Modified SBM Output-Oriented Results:")
print(modified_sbm_output)
```

#### 14. Series Network DEA Model

```python
# ネットワークDEA用のデータを準備
inputs_stage1 = inputs  # ステージ1への入力
intermediate = outputs * 0.8  # 中間製品（例：出力の80%）
outputs_stage2 = outputs  # ステージ2からの出力

# Series Networkモデルの初期化
network_model = SeriesNetworkModel(inputs_stage1, intermediate, outputs_stage2)

# Series Network DEA Model (4.10)
network_results = network_model.evaluate_all()
print("Series Network DEA Results:")
print(network_results)
```

#### 15. Profit Efficiency Model

```python
# Profit Efficiencyモデルの初期化
profit_model = ProfitEfficiencyModel(inputs, outputs, input_costs, output_prices)

# Profit Efficiency Model (4.11)
profit_results = profit_model.evaluate_all()
print("Profit Efficiency Results:")
print(profit_results)
```

#### 16. Congestion DEA Model

```python
# Congestionモデルの初期化
congestion_model = CongestionModel(inputs, outputs)

# Congestion DEA Model (4.13)
congestion_results = congestion_model.evaluate_all()
print("Congestion DEA Results:")
print(congestion_results)
```

#### 17. Common Set of Weights Model

```python
# Common Weightsモデルの初期化
common_weights_model = CommonWeightsModel(inputs, outputs, epsilon=1e-4)

# Common Set of Weights DEA Model (4.14)
u_weights, v_weights, obj_value = common_weights_model.solve()
print("Common Weights:")
print(f"Output weights (u): {u_weights}")
print(f"Input weights (v): {v_weights}")
print(f"Objective value: {obj_value:.4f}")
```

#### 18. Directional Efficiency Model

```python
# Directional Efficiencyモデルの初期化
directional_model = DirectionalEfficiencyModel(inputs, outputs)

# Directional Efficiency DEA Model (4.15)
# デフォルト方向: gx = -x_p, gy = y_p
directional_results = directional_model.evaluate_all()
print("Directional Efficiency Results:")
print(directional_results)

# カスタム方向ベクトルを指定
gx_custom = np.array([-1.0, -1.0])  # 入力方向
gy_custom = np.array([1.0, 1.0])    # 出力方向
eff_dir, lambdas_dir = directional_model.solve(0, gx=gx_custom, gy=gy_custom)
print(f"Directional Efficiency with custom direction: {eff_dir:.4f}")
```

### 完全な使用例

すべてのモデルを含む完全な例：

```python
import numpy as np
from dea import *

# データの準備
inputs = np.array([
    [20, 11], [11, 40], [32, 30], [21, 30], [20, 11],
    [12, 43], [7, 45], [31, 45], [19, 22], [32, 11]
])
outputs = np.array([
    [8, 30], [21, 20], [34, 40], [18, 50], [6, 17],
    [23, 58], [28, 30], [40, 20], [27, 23], [38, 45]
])

# 1. CCRモデル
ccr = CCRModel(inputs, outputs)
print("CCR Envelopment:", ccr.evaluate_all(method='envelopment'))
print("CCR Multiplier:", ccr.evaluate_all(method='multiplier'))

# 2. BCCモデル
bcc = BCCModel(inputs, outputs)
print("BCC Envelopment:", bcc.evaluate_all(method='envelopment'))

# 3. Additiveモデル
additive = AdditiveModel(inputs, outputs)
print("Additive CCR:", additive.evaluate_all(model_type='ccr'))

# 4. Two-Phaseモデル
twophase = TwoPhaseModel(inputs, outputs)
print("Two-Phase BCC:", twophase.evaluate_all())

# 5. APスーパー効率モデル
ap = APModel(inputs, outputs)
print("AP Input Envelopment:", ap.evaluate_all(orientation='input', method='envelopment'))

# 6. MAJモデル
maj = MAJModel(inputs, outputs)
print("MAJ:", maj.evaluate_all())

# 7. Norm L1モデル
norml1 = NormL1Model(inputs, outputs)
print("Norm L1:", norml1.evaluate_all())

# 8. Returns to Scale
rts = ReturnsToScaleModel(inputs, outputs)
print("Returns to Scale:", rts.evaluate_all(method='envelopment'))

# 9. Cost/Revenue Efficiency
input_costs = np.array([1.0, 2.0])
output_prices = np.array([3.0, 4.0])
cost = CostEfficiencyModel(inputs, outputs, input_costs)
revenue = RevenueEfficiencyModel(inputs, outputs, output_prices)
print("Cost Efficiency:", cost.evaluate_all())

# 10. Malmquist
inputs_t1 = inputs * 1.1
outputs_t1 = outputs * 1.15
malmquist = MalmquistModel(inputs, outputs, inputs_t1, outputs_t1)
print("Malmquist:", malmquist.evaluate_all())

# 11. SBM
sbm = SBMModel(inputs, outputs)
print("SBM Model 1:", sbm.evaluate_all(model_type=1))

# 12. Modified SBM
modified_sbm = ModifiedSBMModel(inputs, outputs)
print("Modified SBM:", modified_sbm.evaluate_all(orientation='input'))

# 13. Series Network
intermediate = outputs * 0.8
network = SeriesNetworkModel(inputs, intermediate, outputs)
print("Network DEA:", network.evaluate_all())

# 14. Profit Efficiency
profit = ProfitEfficiencyModel(inputs, outputs, input_costs, output_prices)
print("Profit Efficiency:", profit.evaluate_all())

# 15. Congestion
congestion = CongestionModel(inputs, outputs)
print("Congestion:", congestion.evaluate_all())

# 16. Common Weights
common = CommonWeightsModel(inputs, outputs)
u, v, obj = common.solve()
print(f"Common Weights - Objective: {obj:.4f}")

# 17. Directional Efficiency
directional = DirectionalEfficiencyModel(inputs, outputs)
print("Directional Efficiency:", directional.evaluate_all())
```

### テスト実行

PDFのTable 3.1のデータを使用したテストスクリプトが含まれています：

```bash
python test_dea.py
```

## データ形式

### 基本的なデータ形式

- **inputs**: 形状 `(n_dmus, n_inputs)` のNumPy配列
- **outputs**: 形状 `(n_dmus, n_outputs)` のNumPy配列

### 特殊なモデル用のデータ形式

#### Cost/Revenue Efficiency Models
- **input_costs**: 形状 `(n_inputs,)` または `(n_dmus, n_inputs)` のNumPy配列
- **output_prices**: 形状 `(n_outputs,)` または `(n_dmus, n_outputs)` のNumPy配列

#### Malmquist Productivity Index
- **inputs_t**: 時点tの入力データ
- **outputs_t**: 時点tの出力データ
- **inputs_t1**: 時点t+1の入力データ
- **outputs_t1**: 時点t+1の出力データ

#### Series Network DEA Model
- **inputs_stage1**: ステージ1への入力、形状 `(n_dmus, n_inputs_stage1)`
- **intermediate**: 中間製品、形状 `(n_dmus, n_intermediate)`
- **outputs_stage2**: ステージ2からの出力、形状 `(n_dmus, n_outputs_stage2)`

## 参考文献

Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Studies in Fuzziness and Soft Computing, Vol. 386. Springer.

## ライセンス

LICENSEファイルを参照してください。
