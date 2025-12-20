# DEA Model Library

Python実装による包括的なData Envelopment Analysis (DEA) ライブラリ。Hosseinzadeh Lotfi et al. (2020)の第3章・第4章、Benchmarkingパッケージ、deaRパッケージに基づいて実装されています。

## 目次

- [インストール](#インストール)
- [基本的な使用方法](#基本的な使用方法)
- [実装されているモデル一覧](#実装されているモデル一覧)
- [各モデルの詳細説明](#各モデルの詳細説明)
  - [基本モデル](#基本モデル)
  - [高度なモデル](#高度なモデル)
  - [追加モデル](#追加モデル)
- [データ形式](#データ形式)
- [参考文献](#参考文献)

## インストール

```bash
pip install pandas numpy scipy
```

## 基本的な使用方法

```python
import numpy as np
from dea import CCRModel

# データの準備
# inputs: (n_dmus, n_inputs) の形状のNumPy配列
# outputs: (n_dmus, n_outputs) の形状のNumPy配列
inputs = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs = np.array([[1], [2], [3], [1]])

# モデルの初期化
model = CCRModel(inputs, outputs)

# 特定のDMUの効率を計算
efficiency, lambdas, input_slacks, output_slacks = model.solve_envelopment(dmu_index=0)

# すべてのDMUを評価
results = model.evaluate_all()
print(results)
```

## 実装されているモデル一覧

### 基本モデル（第3章）
- **CCRModel**: CCR (Charnes-Cooper-Rhodes) モデル
- **BCCModel**: BCC (Banker-Charnes-Cooper) モデル
- **AdditiveModel**: Additiveモデル
- **TwoPhaseModel**: Two-Phaseモデル

### 高度なモデル（第4章）
- **APModel**: AP (Anderson-Peterson) スーパー効率モデル
- **MAJModel**: MAJスーパー効率モデル
- **NormL1Model**: Norm L1スーパー効率モデル
- **ReturnsToScaleModel**: 規模の収穫モデル
- **CostEfficiencyModel**: コスト効率モデル
- **RevenueEfficiencyModel**: 収益効率モデル
- **MalmquistModel**: Malmquist生産性指数
- **SBMModel**: SBM (Slacks-Based Measure) モデル
- **ModifiedSBMModel**: 修正SBMモデル
- **ProfitEfficiencyModel**: 利益効率モデル
- **SeriesNetworkModel**: 系列ネットワークDEAモデル
- **CongestionModel**: 混雑DEAモデル
- **CommonWeightsModel**: 共通重みモデル
- **DirectionalEfficiencyModel**: 方向性効率モデル

### 追加モデル（Benchmarking/deaRパッケージ）
- **DRSModel**: 収穫逓減モデル
- **IRSModel**: 収穫逓増モデル
- **FDHModel**: FDH (Free Disposal Hull) モデル
- **FDHPlusModel**: FDH+モデル
- **MEAModel**: MEA (Multi-directional Efficiency Analysis)
- **EfficiencyLadderModel**: 効率ラダー分析
- **MergerAnalysisModel**: 合併分析
- **BootstrapDEAModel**: ブートストラップDEA
- **NonRadialModel**: 非放射DEAモデル
- **LGOModel**: 線形目標指向DEAモデル
- **RDMModel**: Range Directional Model
- **AddMinModel**: Additive minモデル
- **AddSuperEffModel**: Additiveスーパー効率モデル
- **DEAPSModel**: DEA-PS (Preference Structure) モデル
- **CrossEfficiencyModel**: クロス効率分析
- **transform_undesirable**: 非望ましい入出力の変換
- **StoNEDModel**: StoNED (Stochastic Non-smooth Envelopment of Data) モデル

## 各モデルの詳細説明

### 基本モデル

#### 1. CCRModel - CCR (Charnes-Cooper-Rhodes) モデル

**説明**: 定規模収穫（CRS）を仮定した基本的なDEAモデル。

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列 - 入力データ
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列 - 出力データ

**主要メソッド**:
- `solve_envelopment(dmu_index)`: 入力指向包絡モデルを解く
  - 戻り値: `(efficiency, lambdas, input_slacks, output_slacks)`
- `solve_multiplier(dmu_index, epsilon=1e-6)`: 入力指向乗数モデルを解く
  - 戻り値: `(efficiency, v_weights, u_weights, u0)`
- `solve_output_oriented_envelopment(dmu_index)`: 出力指向包絡モデルを解く
- `solve_output_oriented_multiplier(dmu_index, epsilon=1e-6)`: 出力指向乗数モデルを解く
- `evaluate_all(method='envelopment')`: すべてのDMUを評価
  - 戻り値: `pd.DataFrame` - 効率スコア、ラムダ、スラックを含む

**使用例**:
```python
from dea import CCRModel
import numpy as np

inputs = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs = np.array([[1], [2], [3], [1]])

model = CCRModel(inputs, outputs)

# 入力指向包絡モデル
efficiency, lambdas, input_slacks, output_slacks = model.solve_envelopment(0)
print(f"Efficiency: {efficiency:.4f}")

# すべてのDMUを評価
results = model.evaluate_all(method='envelopment')
print(results)
```

#### 2. BCCModel - BCC (Banker-Charnes-Cooper) モデル

**説明**: 可変規模収穫（VRS）を仮定したDEAモデル。

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列

**主要メソッド**:
- `solve_envelopment(dmu_index)`: 入力指向包絡モデルを解く
- `solve_multiplier(dmu_index, epsilon=1e-6)`: 入力指向乗数モデルを解く
- `solve_output_oriented_envelopment(dmu_index)`: 出力指向包絡モデルを解く
- `solve_output_oriented_multiplier(dmu_index, epsilon=1e-6)`: 出力指向乗数モデルを解く
- `evaluate_all(method='envelopment')`: すべてのDMUを評価

**使用例**:
```python
from dea import BCCModel
import numpy as np

inputs = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs = np.array([[1], [2], [3], [1]])

model = BCCModel(inputs, outputs)
efficiency, lambdas, input_slacks, output_slacks = model.solve_envelopment(0)
print(f"BCC Efficiency: {efficiency:.4f}")
```

#### 3. AdditiveModel - Additiveモデル

**説明**: スラックの合計を最大化するAdditiveモデル。

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列

**主要メソッド**:
- `solve_ccr(dmu_index)`: CCR Additiveモデルを解く
- `solve_bcc(dmu_index)`: BCC Additiveモデルを解く
- `evaluate_all(model_type='ccr')`: すべてのDMUを評価

**使用例**:
```python
from dea import AdditiveModel
import numpy as np

inputs = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs = np.array([[1], [2], [3], [1]])

model = AdditiveModel(inputs, outputs)
total_slack, lambdas, input_slacks, output_slacks = model.solve_ccr(0)
print(f"Total Slack: {total_slack:.4f}")
```

#### 4. TwoPhaseModel - Two-Phaseモデル

**説明**: 2段階で効率を計算するモデル（第1段階：効率スコア、第2段階：スラック最大化）。

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列

**主要メソッド**:
- `solve(dmu_index)`: Two-Phaseモデルを解く
  - 戻り値: `(efficiency, lambdas, input_slacks, output_slacks, total_slack)`

**使用例**:
```python
from dea import TwoPhaseModel
import numpy as np

inputs = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs = np.array([[1], [2], [3], [1]])

model = TwoPhaseModel(inputs, outputs)
efficiency, lambdas, input_slacks, output_slacks, total_slack = model.solve(0)
print(f"Efficiency: {efficiency:.4f}, Total Slack: {total_slack:.4f}")
```

### 高度なモデル

#### 5. APModel - AP (Anderson-Peterson) スーパー効率モデル

**説明**: 効率的なDMUをランキングするためのスーパー効率モデル。

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列

**主要メソッド**:
- `solve_input_oriented_envelopment(dmu_index)`: 入力指向包絡モデル
  - 戻り値: `(efficiency, lambdas)` - スーパー効率スコアとラムダ値
- `solve_output_oriented_envelopment(dmu_index)`: 出力指向包絡モデル
  - 戻り値: `(efficiency, lambdas)`
- `solve_input_oriented_multiplier(dmu_index, epsilon=1e-6)`: 入力指向乗数モデル
  - 戻り値: `(efficiency, v_weights, u_weights)`
- `solve_output_oriented_multiplier(dmu_index, epsilon=1e-6)`: 出力指向乗数モデル
  - 戻り値: `(efficiency, v_weights, u_weights)`
- `evaluate_all(orientation='input', method='envelopment')`: すべてのDMUを評価
  - 戻り値: `pd.DataFrame` - 効率スコアとラムダ値を含む

**使用例**:
```python
from dea import APModel
import numpy as np

inputs = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs = np.array([[1], [2], [3], [1]])

model = APModel(inputs, outputs)
efficiency, lambdas = model.solve_input_oriented_envelopment(0)
print(f"AP Super-Efficiency: {efficiency:.4f}")
print(f"Lambdas: {lambdas}")
```

#### 6. CostEfficiencyModel - コスト効率モデル

**説明**: 入力コストを考慮した効率モデル。

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列
- `input_costs`: `(n_dmus, n_inputs)` または `(n_inputs,)` 形状のNumPy配列 - 入力コスト

**主要メソッド**:
- `solve(dmu_index)`: コスト効率を計算
  - 戻り値: `(cost_efficiency, optimal_inputs, lambdas)`

**使用例**:
```python
from dea import CostEfficiencyModel
import numpy as np

inputs = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs = np.array([[1], [2], [3], [1]])
input_costs = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])  # 各DMUの入力コスト

model = CostEfficiencyModel(inputs, outputs, input_costs)
cost_eff, optimal_inputs, lambdas = model.solve(0)
print(f"Cost Efficiency: {cost_eff:.4f}")
print(f"Optimal Inputs: {optimal_inputs}")
```

#### 7. RevenueEfficiencyModel - 収益効率モデル

**説明**: 出力価格を考慮した効率モデル。

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列
- `output_prices`: `(n_dmus, n_outputs)` または `(n_outputs,)` 形状のNumPy配列 - 出力価格

**主要メソッド**:
- `solve(dmu_index)`: 収益効率を計算
  - 戻り値: `(revenue_efficiency, optimal_outputs, lambdas)`

**使用例**:
```python
from dea import RevenueEfficiencyModel
import numpy as np

inputs = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs = np.array([[1], [2], [3], [1]])
output_prices = np.array([[10], [10], [10], [10]])  # 各出力の価格

model = RevenueEfficiencyModel(inputs, outputs, output_prices)
revenue_eff, optimal_outputs, lambdas = model.solve(0)
print(f"Revenue Efficiency: {revenue_eff:.4f}")
```

#### 8. ProfitEfficiencyModel - 利益効率モデル

**説明**: 入力コストと出力価格の両方を考慮した利益効率モデル。

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列
- `input_costs`: `(n_dmus, n_inputs)` または `(n_inputs,)` 形状のNumPy配列
- `output_prices`: `(n_dmus, n_outputs)` または `(n_outputs,)` 形状のNumPy配列

**主要メソッド**:
- `solve(dmu_index)`: 利益効率を計算
  - 戻り値: `(profit_efficiency, optimal_inputs, optimal_outputs, lambdas, profit)`

**使用例**:
```python
from dea import ProfitEfficiencyModel
import numpy as np

inputs = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs = np.array([[1], [2], [3], [1]])
input_costs = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
output_prices = np.array([[10], [10], [10], [10]])

model = ProfitEfficiencyModel(inputs, outputs, input_costs, output_prices)
profit_eff, opt_inputs, opt_outputs, lambdas, profit = model.solve(0)
print(f"Profit Efficiency: {profit_eff:.4f}")
```

#### 9. MalmquistModel - Malmquist生産性指数

**説明**: 時系列データを用いた生産性変化の測定。

**入力**:
- `inputs_t`: `(n_dmus, n_inputs)` 形状のNumPy配列 - 時点tの入力
- `outputs_t`: `(n_dmus, n_outputs)` 形状のNumPy配列 - 時点tの出力
- `inputs_t1`: `(n_dmus, n_inputs)` 形状のNumPy配列 - 時点t+1の入力
- `outputs_t1`: `(n_dmus, n_outputs)` 形状のNumPy配列 - 時点t+1の出力

**主要メソッド**:
- `calculate_malmquist_index(dmu_index)`: Malmquist指数を計算
  - 戻り値: `(d_t_t, d_t1_t1, d_t_t1, d_t1_t, mi)`
- `evaluate_all()`: すべてのDMUを評価

**使用例**:
```python
from dea import MalmquistModel
import numpy as np

# 時点tのデータ
inputs_t = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs_t = np.array([[1], [2], [3], [1]])

# 時点t+1のデータ
inputs_t1 = np.array([[2.1, 3.1], [3.1, 2.1], [4.1, 1.1], [1.1, 4.1]])
outputs_t1 = np.array([[1.1], [2.1], [3.1], [1.1]])

model = MalmquistModel(inputs_t, outputs_t, inputs_t1, outputs_t1)
d_t_t, d_t1_t1, d_t_t1, d_t1_t, mi = model.calculate_malmquist_index(0)
print(f"Malmquist Index: {mi:.4f}")
```

#### 10. SBMModel - SBM (Slacks-Based Measure) モデル

**説明**: スラックに基づく非放射的効率測定モデル。

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列

**主要メソッド**:
- `solve_model1(dmu_index)`: 第1モデルを解く
- `solve_model2(dmu_index)`: 第2モデルを解く
- `evaluate_all(model_type=1)`: すべてのDMUを評価

**使用例**:
```python
from dea import SBMModel
import numpy as np

inputs = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs = np.array([[1], [2], [3], [1]])

model = SBMModel(inputs, outputs)
sbm_eff, lambdas, input_slacks, output_slacks = model.solve_model1(0)
print(f"SBM Efficiency: {sbm_eff:.4f}")
```

#### 11. DirectionalEfficiencyModel - 方向性効率モデル

**説明**: 指定された方向への効率を測定するモデル。

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列

**主要メソッド**:
- `solve(dmu_index, g_inputs, g_outputs, rts='vrs')`: 方向性効率を計算
  - `g_inputs`: 入力方向ベクトル `(n_inputs,)`
  - `g_outputs`: 出力方向ベクトル `(n_outputs,)`
  - 戻り値: `(efficiency, lambdas, input_slacks, output_slacks)`

**使用例**:
```python
from dea import DirectionalEfficiencyModel
import numpy as np

inputs = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs = np.array([[1], [2], [3], [1]])

model = DirectionalEfficiencyModel(inputs, outputs)
g_inputs = np.array([1, 1])  # 入力方向ベクトル
g_outputs = np.array([1])     # 出力方向ベクトル
efficiency, lambdas, input_slacks, output_slacks = model.solve(0, g_inputs, g_outputs)
print(f"Directional Efficiency: {efficiency:.4f}")
```

### 追加モデル

#### 12. DRSModel / IRSModel - 収穫逓減/逓増モデル

**説明**: 収穫逓減（DRS）または収穫逓増（IRS）を仮定したモデル。

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列

**主要メソッド**:
- `solve_input_oriented_envelopment(dmu_index)`: 入力指向包絡モデル
- `solve_output_oriented_envelopment(dmu_index)`: 出力指向包絡モデル
- `evaluate_all(orientation='input')`: すべてのDMUを評価

**使用例**:
```python
from dea import DRSModel, IRSModel
import numpy as np

inputs = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs = np.array([[1], [2], [3], [1]])

# 収穫逓減モデル
drs_model = DRSModel(inputs, outputs)
efficiency, lambdas, input_slacks, output_slacks = drs_model.solve_input_oriented_envelopment(0)
print(f"DRS Efficiency: {efficiency:.4f}")

# 収穫逓増モデル
irs_model = IRSModel(inputs, outputs)
efficiency, lambdas, input_slacks, output_slacks = irs_model.solve_input_oriented_envelopment(0)
print(f"IRS Efficiency: {efficiency:.4f}")
```

#### 13. FDHModel / FDHPlusModel - FDHモデル

**説明**: 凸性を仮定しないFree Disposal Hullモデル。

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列

**主要メソッド**:
- `solve_input_oriented_envelopment(dmu_index)`: 入力指向包絡モデル
- `solve_output_oriented_envelopment(dmu_index)`: 出力指向包絡モデル
- `evaluate_all(orientation='input')`: すべてのDMUを評価

**使用例**:
```python
from dea import FDHModel
import numpy as np

inputs = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs = np.array([[1], [2], [3], [1]])

model = FDHModel(inputs, outputs)
efficiency, lambdas, input_slacks, output_slacks = model.solve_input_oriented_envelopment(0)
print(f"FDH Efficiency: {efficiency:.4f}")
```

#### 14. MEAModel - MEA (Multi-directional Efficiency Analysis)

**説明**: 各入力/出力方向の潜在的な改善を計算する多方向効率分析。

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列

**主要メソッド**:
- `solve(dmu_index, orientation='input', rts='vrs')`: MEAを計算
  - 戻り値: `(efficiency, lambdas, input_slacks, output_slacks, directions)`

**使用例**:
```python
from dea import MEAModel
import numpy as np

inputs = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs = np.array([[1], [2], [3], [1]])

model = MEAModel(inputs, outputs)
efficiency, lambdas, input_slacks, output_slacks, directions = model.solve(0)
print(f"MEA Efficiency: {efficiency:.4f}")
```

#### 15. CrossEfficiencyModel - クロス効率分析

**説明**: 各DMUの重みを使用して他のDMUの効率を評価するクロス効率分析。

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列

**主要メソッド**:
- `solve(orientation='io', rts='crs', epsilon=0.0, selfapp=True, correction=False, M2=True, M3=True)`: クロス効率を計算
  - 戻り値: 辞書（`cross_efficiency_matrix`, `average_scores`, `self_efficiency`など）

**使用例**:
```python
from dea import CrossEfficiencyModel
import numpy as np

inputs = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs = np.array([[1], [2], [3], [1]])

model = CrossEfficiencyModel(inputs, outputs)
results = model.solve(orientation='io', rts='crs')
print(f"Cross-Efficiency Matrix:\n{results['cross_efficiency_matrix']}")
print(f"Average Scores: {results['average_scores']}")
```

#### 16. StoNEDModel - StoNED (Stochastic Non-smooth Envelopment of Data)

**説明**: DEAとSFAを組み合わせた確率的非パラメトリック包絡モデル。

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `output`: `(n_dmus,)` 形状のNumPy配列 - **単一出力のみ**

**主要メソッド**:
- `solve(rts='vrs', cost=False, mult=False, method='MM')`: StoNEDを計算
  - `rts`: 規模の収穫 ('vrs', 'drs', 'crs', 'irs')
  - `cost`: Trueならコスト関数、Falseなら生産関数
  - `mult`: Trueなら乗法的誤差項、Falseなら加法的誤差項
  - `method`: 'MM' (Method of Moments) または 'PSL' (Pseudo Likelihood)
  - 戻り値: 辞書（`eff`, `sigma_u`, `fit`, `front`など）

**使用例**:
```python
from dea import StoNEDModel
import numpy as np

# 単一出力が必要
inputs = np.array([[2], [3], [4], [1]])  # 単一入力
output = np.array([1, 2, 3, 1])  # 単一出力

model = StoNEDModel(inputs, output)
results = model.solve(rts='vrs', cost=False, mult=True, method='MM')
print(f"StoNED Efficiency: {results['eff']}")
print(f"Sigma_u: {results['sigma_u']:.6f}")
```

#### 17. transform_undesirable - 非望ましい入出力の変換

**説明**: 非望ましい入出力を望ましい形式に変換する関数（Seiford and Zhu 2002）。

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列
- `ud_inputs`: 非望ましい入力のインデックス配列（例: `np.array([0, 2])`）
- `ud_outputs`: 非望ましい出力のインデックス配列（例: `np.array([1])`）

**戻り値**:
- `transformed_inputs`: 変換後の入力
- `transformed_outputs`: 変換後の出力
- `vtrans_i`: 入力の変換ベクトル
- `vtrans_o`: 出力の変換ベクトル

**使用例**:
```python
from dea import transform_undesirable
import numpy as np

inputs = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs = np.array([[1, 5], [2, 4], [3, 3], [1, 6]])

# 最初の入力と最初の出力が非望ましい
transformed_inputs, transformed_outputs, vtrans_i, vtrans_o = transform_undesirable(
    inputs, outputs, ud_inputs=np.array([0]), ud_outputs=np.array([0])
)
print(f"Transformed inputs shape: {transformed_inputs.shape}")
```

## データ形式

### 基本的なデータ形式

すべてのモデルは以下の形式のNumPy配列を入力として受け取ります：

- **inputs**: `(n_dmus, n_inputs)` 形状の2次元配列
  - 各行が1つのDMU（Decision Making Unit）を表す
  - 各列が1つの入力変数を表す
- **outputs**: `(n_dmus, n_outputs)` 形状の2次元配列
  - 各行が1つのDMUを表す
  - 各列が1つの出力変数を表す

### データの準備例

```python
import numpy as np

# 例: 10個のDMU、2つの入力、2つの出力
n_dmus = 10
n_inputs = 2
n_outputs = 2

# ランダムデータの生成（実際の使用時は実データを使用）
inputs = np.random.rand(n_dmus, n_inputs) * 10
outputs = np.random.rand(n_dmus, n_outputs) * 10

# または、CSVファイルから読み込む
import pandas as pd
data = pd.read_csv('data.csv')
inputs = data[['input1', 'input2']].values
outputs = data[['output1', 'output2']].values
```

### 特殊なデータ形式

一部のモデルは追加のデータを必要とします：

- **CostEfficiencyModel**: `input_costs` - `(n_dmus, n_inputs)` または `(n_inputs,)` 形状
- **RevenueEfficiencyModel**: `output_prices` - `(n_dmus, n_outputs)` または `(n_outputs,)` 形状
- **ProfitEfficiencyModel**: `input_costs` と `output_prices` の両方
- **MalmquistModel**: 2時点のデータ（`inputs_t`, `outputs_t`, `inputs_t1`, `outputs_t1`）
- **StoNEDModel**: `output` - `(n_dmus,)` 形状の1次元配列（単一出力のみ）

## 参考文献

1. Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer.

2. Kuosmanen, T., & Kortelainen, M. (2012). Stochastic non-smooth envelopment of data: semi-parametric frontier estimation subject to shape constraints. *Journal of Productivity Analysis*, 38(1), 11-28.

3. Bogetoft, P., & Otto, L. (2011). *Benchmarking with DEA, SFA, and R*. Springer-Verlag.

4. Seiford, L. M., & Zhu, J. (2002). Modeling undesirable factors in efficiency evaluation. *European Journal of Operational Research*, 142(1), 16-20.

5. Zhu, J. (1996). Data envelopment analysis with preference structure. *Journal of the Operational Research Society*, 47(1), 136-150.

6. Doyle, J., & Green, R. (1994). Efficiency and cross-efficiency in DEA: derivations, meanings and uses. *Journal of the Operational Research Society*, 45(5), 567-578.

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## バージョン

現在のバージョン: 1.0.0
