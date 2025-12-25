# DEA Model

Data Envelopment Analysis (DEA) モデルの実装とStreamlitアプリケーション。

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

### Streamlitアプリ

```bash
streamlit run streamlit_app.py
```

### Pythonライブラリ

```python
from dea import CCRModel, BCCModel
import numpy as np

inputs = np.array([[2, 3], [3, 2], [4, 1]])
outputs = np.array([[1, 2], [2, 3], [3, 4]])

model = CCRModel(inputs, outputs)
results = model.evaluate_all()
print(results)
```

## 実装モデル

- CCR (Charnes-Cooper-Rhodes)
- BCC (Banker-Charnes-Cooper)
- Super-Efficiency
- Returns to Scale
- Cost Efficiency
- Revenue Efficiency
- Malmquist Productivity Index
- SBM (Slacks-Based Measure)
- Directional Efficiency
- Bootstrap DEA
- Cross Efficiency

## ライセンス

MIT License
