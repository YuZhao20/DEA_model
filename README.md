# DEA Model Library

Data Envelopment Analysis (DEA) implementation in Python based on Hosseinzadeh Lotfi et al. (2020), Chapter 3: Basic DEA Models.

## 概要

このライブラリは、Hosseinzadeh Lotfi et al. (2020)の第3章「Basic DEA Models」に基づいて実装されたDEA（Data Envelopment Analysis）モデルです。

## 実装されているモデル

### CCR (Charnes-Cooper-Rhodes) モデル
- **Input-Oriented CCR Envelopment Model** (3.2.1)
- **Input-Oriented CCR Multiplier Model** (3.2.2)

### BCC (Banker-Charnes-Cooper) モデル
- **Input-Oriented BCC Envelopment Model** (3.2.3)
- **Input-Oriented BCC Multiplier Model** (3.2.3)

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本的な使用例

```python
import numpy as np
from dea import CCRModel, BCCModel

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
