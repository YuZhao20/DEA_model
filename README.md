# DEA Model Library

Python実装による包括的なData Envelopment Analysis (DEA) ライブラリ。Hosseinzadeh Lotfi et al. (2020)の第3章・第4章、Benchmarkingパッケージ、deaRパッケージに基づいて実装されています。

## 目次

- [インストール](#インストール)
- [基本的な使用方法](#基本的な使用方法)
- [実装されているモデル一覧](#実装されているモデル一覧)
- [各モデルの詳細説明](#各モデルの詳細説明)
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

1. **CCRModel**: CCR (Charnes-Cooper-Rhodes) モデル
2. **BCCModel**: BCC (Banker-Charnes-Cooper) モデル
3. **APModel**: AP (Anderson-Peterson) スーパー効率モデル
4. **ReturnsToScaleModel**: 規模の収穫モデル
5. **CostEfficiencyModel**: コスト効率モデル
6. **RevenueEfficiencyModel**: 収益効率モデル
7. **MalmquistModel**: Malmquist生産性指数
8. **SBMModel**: SBM (Slacks-Based Measure) モデル
9. **DirectionalEfficiencyModel**: 方向性効率モデル
10. **BootstrapDEAModel**: ブートストラップDEA
11. **CrossEfficiencyModel**: クロス効率分析

## 各モデルの詳細説明

### 1. CCRModel - CCR (Charnes-Cooper-Rhodes) モデル

**解説**: CCRモデルは、定規模収穫（Constant Returns to Scale, CRS）を仮定した基本的なDEAモデルです。1978年にCharnes、Cooper、Rhodesによって提案され、DEAの基礎となるモデルです。このモデルは、各DMU（Decision Making Unit）の効率を、他のすべてのDMUの線形結合として表現できる効率的なDMUとの比較によって測定します。入力指向では、現在の出力水準を維持しながら入力の削減余地を測定し、出力指向では、現在の入力水準を維持しながら出力の増加余地を測定します。

**参考文献**:
- Charnes, A., Cooper, W. W., & Rhodes, E. (1978). Measuring the efficiency of decision making units. *European Journal of Operational Research*, 2(6), 429-444.
- Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer. (Chapter 3.2)

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

---

### 2. BCCModel - BCC (Banker-Charnes-Cooper) モデル

**解説**: BCCモデルは、可変規模収穫（Variable Returns to Scale, VRS）を仮定したDEAモデルです。1984年にBanker、Charnes、Cooperによって提案されました。CCRモデルと異なり、BCCモデルは規模の収穫が可変であることを考慮します。これにより、規模の経済性や非経済性を考慮した効率測定が可能になります。BCCモデルは、小規模なDMUと大規模なDMUをより公平に比較できるため、実務で広く使用されています。

**参考文献**:
- Banker, R. D., Charnes, A., & Cooper, W. W. (1984). Some models for estimating technical and scale inefficiencies in data envelopment analysis. *Management Science*, 30(9), 1078-1092.
- Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer. (Chapter 3.2.3)

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

---

### 3. APModel - AP (Anderson-Peterson) スーパー効率モデル

**解説**: APモデルは、効率的なDMU（効率スコアが1のDMU）をランキングするためのスーパー効率モデルです。1993年にAndersonとPetersonによって提案されました。通常のDEAモデルでは、効率的なDMUはすべて効率スコア1となり、それらを区別できません。APモデルでは、評価対象のDMUを参照集合から除外することで、効率的なDMUの効率スコアが1を超える値を取ることができ、効率的なDMU間のランキングが可能になります。スーパー効率スコアが1より大きいほど、そのDMUはより効率的であることを示します。

**参考文献**:
- Andersen, P., & Petersen, N. C. (1993). A procedure for ranking efficient units in data envelopment analysis. *Management Science*, 39(10), 1261-1264.
- Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer. (Chapter 4.2)

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

---

### 4. ReturnsToScaleModel - 規模の収穫モデル

**解説**: 規模の収穫モデルは、各DMUの規模の収穫（Returns to Scale, RTS）を判定するためのモデルです。規模の収穫には、定規模収穫（CRS）、可変規模収穫（VRS）、収穫逓減（DRS）、収穫逓増（IRS）があります。このモデルは、各DMUが最適規模にあるか、規模を拡大または縮小すべきかを判断するために使用されます。規模の収穫の判定は、効率改善のための戦略的指針を提供します。

**参考文献**:
- Banker, R. D. (1984). Estimating most productive scale size using data envelopment analysis. *European Journal of Operational Research*, 17(1), 35-44.
- Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer. (Chapter 4.5)

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列

**主要メソッド**:
- `solve_input_oriented_envelopment(dmu_index)`: 入力指向包絡モデル
- `solve_output_oriented_envelopment(dmu_index)`: 出力指向包絡モデル
- `solve_input_oriented_multiplier(dmu_index, epsilon=1e-6)`: 入力指向乗数モデル
- `solve_output_oriented_multiplier(dmu_index, epsilon=1e-6)`: 出力指向乗数モデル
- `evaluate_all(orientation='input', method='envelopment')`: すべてのDMUを評価
  - 戻り値: `pd.DataFrame` - 効率スコア、規模の収穫タイプ、ラムダ、スラックを含む

**使用例**:
```python
from dea import ReturnsToScaleModel
import numpy as np

inputs = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs = np.array([[1], [2], [3], [1]])

model = ReturnsToScaleModel(inputs, outputs)
results = model.evaluate_all(orientation='input', method='envelopment')
print(results[['DMU', 'Efficiency', 'RTS']])
```

---

### 5. CostEfficiencyModel - コスト効率モデル

**解説**: コスト効率モデルは、入力コストを考慮した効率測定モデルです。このモデルは、技術的効率だけでなく、コスト効率も測定します。コスト効率は、現在の出力水準を維持しながら、最小コストで達成可能な入力の組み合わせと、実際のコストとの比率として定義されます。コスト効率は、技術的効率と配分効率の積として分解できます。このモデルは、価格情報が利用可能な場合に、より実用的な効率評価を提供します。

**参考文献**:
- Färe, R., Grosskopf, S., & Lovell, C. A. K. (1985). *The Measurement of Efficiency of Production*. Kluwer Academic Publishers.
- Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer. (Chapter 4.6)

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

---

### 6. RevenueEfficiencyModel - 収益効率モデル

**解説**: 収益効率モデルは、出力価格を考慮した効率測定モデルです。このモデルは、現在の入力水準を維持しながら、最大収益で達成可能な出力の組み合わせと、実際の収益との比率として定義されます。収益効率は、技術的効率と配分効率の積として分解できます。このモデルは、出力の価格情報が利用可能な場合に、収益最大化の観点から効率評価を提供します。

**参考文献**:
- Färe, R., Grosskopf, S., & Lovell, C. A. K. (1985). *The Measurement of Efficiency of Production*. Kluwer Academic Publishers.
- Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer. (Chapter 4.7)

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

---

### 7. MalmquistModel - Malmquist生産性指数

**解説**: Malmquist生産性指数は、時系列データを用いて生産性の変化を測定するモデルです。1953年にMalmquistによって提案され、1994年にFäreらによってDEAに適用されました。この指数は、2つの時点間の生産性変化を、技術的効率の変化（Efficiency Change, EC）と技術進歩（Technical Change, TC）に分解します。Malmquist指数が1より大きい場合、生産性が向上したことを示し、1より小さい場合、生産性が低下したことを示します。

**参考文献**:
- Malmquist, S. (1953). Index numbers and indifference surfaces. *Trabajos de Estadística*, 4(2), 209-242.
- Färe, R., Grosskopf, S., Norris, M., & Zhang, Z. (1994). Productivity growth, technical progress, and efficiency change in industrialized countries. *American Economic Review*, 84(1), 66-83.
- Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer. (Chapter 4.8)

**入力**:
- `inputs_t`: `(n_dmus, n_inputs)` 形状のNumPy配列 - 時点tの入力
- `outputs_t`: `(n_dmus, n_outputs)` 形状のNumPy配列 - 時点tの出力
- `inputs_t1`: `(n_dmus, n_inputs)` 形状のNumPy配列 - 時点t+1の入力
- `outputs_t1`: `(n_dmus, n_outputs)` 形状のNumPy配列 - 時点t+1の出力

**主要メソッド**:
- `calculate_malmquist_index(dmu_index)`: Malmquist指数を計算
  - 戻り値: `(d_t_t, d_t1_t1, d_t_t1, d_t1_t, mi)`
- `evaluate_all()`: すべてのDMUを評価
  - 戻り値: `pd.DataFrame` - Malmquist指数、効率変化、技術進歩を含む

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
results = model.evaluate_all()
print(results[['DMU', 'Malmquist_Index', 'Efficiency_Change', 'Technical_Change']])
```

---

### 8. SBMModel - SBM (Slacks-Based Measure) モデル

**解説**: SBMモデルは、スラックに基づく非放射的効率測定モデルです。2001年にToneによって提案されました。従来の放射的DEAモデル（CCR、BCC）とは異なり、SBMモデルは入力と出力のスラックを直接考慮するため、非効率性の測定がより正確になります。SBM効率は0から1の間の値を取り、1に近いほど効率的であることを示します。このモデルは、入力と出力の両方のスラックを同時に考慮するため、より包括的な効率評価を提供します。

**参考文献**:
- Tone, K. (2001). A slacks-based measure of efficiency in data envelopment analysis. *European Journal of Operational Research*, 130(3), 498-509.
- Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer. (Chapter 4.9)

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列

**主要メソッド**:
- `solve_model1(dmu_index, rts='vrs')`: 第1モデル（入力指向）を解く
  - 戻り値: `(sbm_eff, lambdas, input_slacks, output_slacks)`
- `solve_model2(dmu_index, rts='vrs')`: 第2モデル（出力指向）を解く
  - 戻り値: `(sbm_eff, lambdas, input_slacks, output_slacks)`
- `evaluate_all(model_type=1, rts='vrs')`: すべてのDMUを評価
  - 戻り値: `pd.DataFrame` - SBM効率スコア、ラムダ、スラックを含む

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

---

### 9. DirectionalEfficiencyModel - 方向性効率モデル

**解説**: 方向性効率モデルは、指定された方向への効率を測定するモデルです。このモデルは、入力と出力の改善方向を明示的に指定できるため、より柔軟な効率測定が可能です。従来の放射的DEAモデルは、入力指向または出力指向のいずれか一方のみを考慮しますが、方向性効率モデルでは、入力と出力の両方を同時に改善する方向を指定できます。このモデルは、特定の改善戦略に基づいた効率評価を提供します。

**参考文献**:
- Chambers, R. G., Chung, Y., & Färe, R. (1996). Benefit and distance functions. *Journal of Economic Theory*, 70(2), 407-419.
- Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer. (Chapter 4.15)

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列

**主要メソッド**:
- `solve(dmu_index, gx=None, gy=None, rts='vrs')`: 方向性効率を計算
  - `gx`: 入力方向ベクトル `(n_inputs,)`（デフォルト: `-x_p`）
  - `gy`: 出力方向ベクトル `(n_outputs,)`（デフォルト: `y_p`）
  - 戻り値: `(efficiency, lambdas, input_slacks, output_slacks)`
- `evaluate_all(gx=None, gy=None, rts='vrs')`: すべてのDMUを評価
  - 戻り値: `pd.DataFrame` - 方向性効率スコア、ラムダ、スラックを含む

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

---

### 10. BootstrapDEAModel - ブートストラップDEA

**解説**: ブートストラップDEAは、DEA効率スコアの統計的推論を可能にするモデルです。1998年にSimarとWilsonによって提案されました。DEAは非パラメトリックな手法であるため、従来の統計的推論が困難でした。ブートストラップ法を用いることで、効率スコアの信頼区間やバイアス補正を提供し、効率評価の統計的有意性を評価できます。このモデルは、サンプルサイズが小さい場合や、効率スコアの不確実性を考慮したい場合に特に有用です。

**参考文献**:
- Simar, L., & Wilson, P. W. (1998). Sensitivity analysis of efficiency scores: How to bootstrap in nonparametric frontier models. *Management Science*, 44(11), 49-61.
- Simar, L., & Wilson, P. W. (2000). Statistical inference in nonparametric frontier models: The state of the art. *Journal of Productivity Analysis*, 13(1), 49-78.
- Bogetoft, P., & Otto, L. (2011). *Benchmarking with DEA, SFA, and R*. Springer-Verlag.

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列
- `rts`: 規模の収穫（デフォルト: `'vrs'`）- `'crs'`, `'vrs'`, `'drs'`, `'irs'`
- `orientation`: 指向性（デフォルト: `'in'`）- `'in'`（入力指向）または `'out'`（出力指向）

**主要メソッド**:
- `solve(dmu_index, n_bootstrap=2000, alpha=0.05)`: ブートストラップDEAを実行
  - `n_bootstrap`: ブートストラップ回数（デフォルト: 2000）
  - `alpha`: 信頼区間の有意水準（デフォルト: 0.05）
  - 戻り値: 辞書（`original_eff`, `bias_corrected_eff`, `lower_bound`, `upper_bound`, `bias`, `std_error`など）
- `evaluate_all(n_bootstrap=2000, alpha=0.05)`: すべてのDMUを評価
  - 戻り値: `pd.DataFrame` - 元の効率、バイアス補正効率、信頼区間を含む

**使用例**:
```python
from dea import BootstrapDEAModel
import numpy as np

inputs = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs = np.array([[1], [2], [3], [1]])

model = BootstrapDEAModel(inputs, outputs, rts='vrs', orientation='in')
results = model.evaluate_all(n_bootstrap=1000, alpha=0.05)
print(results[['DMU', 'Original_Efficiency', 'Bias_Corrected_Efficiency', 
               'Lower_Bound', 'Upper_Bound']])
```

---

### 11. CrossEfficiencyModel - クロス効率分析

**解説**: クロス効率分析は、各DMUの重みを使用して他のDMUの効率を評価する手法です。1994年にDoyleとGreenによって提案されました。従来のDEAでは、各DMUは自分に最も有利な重みを選択するため、自己効率スコアが過大評価される可能性があります。クロス効率分析では、各DMUの重みを使用して他のすべてのDMUの効率を評価し、平均クロス効率スコアを計算します。これにより、より公平で一貫性のある効率ランキングが得られます。

**参考文献**:
- Doyle, J., & Green, R. (1994). Efficiency and cross-efficiency in DEA: derivations, meanings and uses. *Journal of the Operational Research Society*, 45(5), 567-578.
- Sexton, T. R., Silkman, R. H., & Hogan, A. J. (1986). Data envelopment analysis: Critique and extensions. *New Directions for Program Evaluation*, 1986(32), 73-105.

**入力**:
- `inputs`: `(n_dmus, n_inputs)` 形状のNumPy配列
- `outputs`: `(n_dmus, n_outputs)` 形状のNumPy配列

**主要メソッド**:
- `solve(orientation='io', rts='crs', ...)`: 詳細なクロス効率結果を計算
  - `orientation`: `'io'`（入力指向）または `'oo'`（出力指向）
  - `rts`: 規模の収穫（`'crs'`, `'vrs'`, `'drs'`, `'irs'`）
  - 戻り値: 辞書（`Arbitrary`キー内に`cross_eff`行列、`e`平均スコアなど）
- `evaluate_all(orientation='io', rts='crs')`: すべてのDMUを評価
  - 戻り値: `pd.DataFrame` - DMU、Cross_Efficiency、Self_Efficiencyを含む

**使用例**:
```python
from dea import CrossEfficiencyModel
import numpy as np

inputs = np.array([[2, 3], [3, 2], [4, 1], [1, 4]])
outputs = np.array([[1], [2], [3], [1]])

model = CrossEfficiencyModel(inputs, outputs)

# 簡単な評価（推奨）
results = model.evaluate_all(orientation='io', rts='crs')
print(results)

# 詳細な結果
detailed = model.solve(orientation='io', rts='crs')
print(f"Cross-Efficiency Matrix:\n{detailed['Arbitrary']['cross_eff']}")
print(f"Average Scores: {detailed['Arbitrary']['e']}")
```

---

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
- **MalmquistModel**: 2時点のデータ（`inputs_t`, `outputs_t`, `inputs_t1`, `outputs_t1`）

## 参考文献

### 主要参考文献

1. Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer.

2. Charnes, A., Cooper, W. W., & Rhodes, E. (1978). Measuring the efficiency of decision making units. *European Journal of Operational Research*, 2(6), 429-444.

3. Banker, R. D., Charnes, A., & Cooper, W. W. (1984). Some models for estimating technical and scale inefficiencies in data envelopment analysis. *Management Science*, 30(9), 1078-1092.

4. Andersen, P., & Petersen, N. C. (1993). A procedure for ranking efficient units in data envelopment analysis. *Management Science*, 39(10), 1261-1264.

5. Tone, K. (2001). A slacks-based measure of efficiency in data envelopment analysis. *European Journal of Operational Research*, 130(3), 498-509.

6. Doyle, J., & Green, R. (1994). Efficiency and cross-efficiency in DEA: derivations, meanings and uses. *Journal of the Operational Research Society*, 45(5), 567-578.

7. Simar, L., & Wilson, P. W. (1998). Sensitivity analysis of efficiency scores: How to bootstrap in nonparametric frontier models. *Management Science*, 44(11), 49-61.

8. Färe, R., Grosskopf, S., Norris, M., & Zhang, Z. (1994). Productivity growth, technical progress, and efficiency change in industrialized countries. *American Economic Review*, 84(1), 66-83.

9. Chambers, R. G., Chung, Y., & Färe, R. (1996). Benefit and distance functions. *Journal of Economic Theory*, 70(2), 407-419.

10. Färe, R., Grosskopf, S., & Lovell, C. A. K. (1985). *The Measurement of Efficiency of Production*. Kluwer Academic Publishers.

11. Banker, R. D. (1984). Estimating most productive scale size using data envelopment analysis. *European Journal of Operational Research*, 17(1), 35-44.

12. Bogetoft, P., & Otto, L. (2011). *Benchmarking with DEA, SFA, and R*. Springer-Verlag.

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## バージョン

現在のバージョン: 1.0.0
