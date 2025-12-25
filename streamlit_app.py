"""
Streamlit App for DEA Models
Interactive web interface for Data Envelopment Analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from dea import (
    CCRModel, BCCModel, APModel,
    DirectionalEfficiencyModel,
    ReturnsToScaleModel,
    CostEfficiencyModel, RevenueEfficiencyModel,
    MalmquistModel,
    SBMModel,
    BootstrapDEAModel,
    CrossEfficiencyModel
)
st.set_page_config(
    page_title="DEA Model Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("DEA Model Analyzer")
st.markdown("*Data Envelopment Analysis モデルのインタラクティブ分析ツール*")
MODEL_INFO = {
    "CCR": {
        "name": "CCR Model (Charnes-Cooper-Rhodes)",
        "explanation": """
**概要 (Overview)**

CCRモデルは、1978年にCharnes、Cooper、Rhodesによって提案された、Data Envelopment Analysis (DEA) の最も基本的なモデルです。Constant Returns to Scale (CRS) を仮定しており、すべてのDMU (Decision Making Unit) が最適規模で運営されていると想定します。

**主要な特徴 (Key Features)**

- **CRS (Constant Returns to Scale) の仮定**: 規模に関する収穫一定 - 入力を2倍にすると出力も2倍になるという線形関係を仮定
- **Technical Efficiency の測定**: 各DMUが効率的フロンティア (efficient frontier) からどれだけ離れているかを測定
- **Input-Oriented / Output-Oriented**: 
  - Input-Oriented: 現在の出力水準を維持しながら入力の削減余地を測定
  - Output-Oriented: 現在の入力水準を維持しながら出力の増加余地を測定

**主な用途 (Applications)**

- 銀行支店、病院、学校などの相対的効率性評価
- 製造業における生産効率の比較分析
- 公共サービスのパフォーマンス評価

**効率スコアの解釈 (Efficiency Score Interpretation)**

- **スコア = 1.0**: 効率的 (efficient) - フロンティア上に位置
- **スコア < 1.0**: 非効率的 (inefficient) - 改善の余地あり
- **入力指向の場合**: スコア0.8は入力を20%削減可能であることを意味
        """,
        "references": [
            "Charnes, A., Cooper, W. W., & Rhodes, E. (1978). Measuring the efficiency of decision making units. European Journal of Operational Research, 2(6), 429-444.",
            "Cooper, W. W., Seiford, L. M., & Tone, K. (2007). Data Envelopment Analysis: A Comprehensive Text with Models, Applications, References and DEA-Solver Software. Springer."
        ],
        "formulation": r"""
**Input-Oriented Envelopment Model (入力指向包絡モデル)**

**目的関数 (Objective Function):**
$$
\min \theta
$$

**制約条件 (Constraints):**
$$
\begin{align}
\sum_{j=1}^{n} \lambda_j x_{ij} &\leq \theta x_{ip}, \quad i=1,\ldots,m \\
\sum_{j=1}^{n} \lambda_j y_{rj} &\geq y_{rp}, \quad r=1,\ldots,s \\
\lambda_j &\geq 0, \quad j=1,\ldots,n
\end{align}
$$

**Output-Oriented Envelopment Model (出力指向包絡モデル)**

**目的関数 (Objective Function):**
$$
\max \phi
$$

**制約条件 (Constraints):**
$$
\begin{align}
\sum_{j=1}^{n} \lambda_j x_{ij} &\leq x_{ip}, \quad i=1,\ldots,m \\
\sum_{j=1}^{n} \lambda_j y_{rj} &\geq \phi y_{rp}, \quad r=1,\ldots,s \\
\lambda_j &\geq 0, \quad j=1,\ldots,n
\end{align}
$$

**変数の説明 (Variable Definitions):**
- $\theta$: 入力指向の効率スコア (input-oriented efficiency score)
- $\phi$: 出力指向の効率スコア (output-oriented efficiency score)
- $\lambda_j$: 強度変数 (intensity variables)
- $x_{ij}$: DMU $j$ の入力 $i$
- $y_{rj}$: DMU $j$ の出力 $r$
"""
    },
    "BCC": {
        "name": "BCC Model (Banker-Charnes-Cooper)",
        "explanation": """
**概要 (Overview)**

BCCモデルは、1984年にBanker、Charnes、Cooperによって提案された、Variable Returns to Scale (VRS) を仮定したDEAモデルです。CCRモデルの拡張版であり、規模の経済性や非経済性を考慮した効率測定が可能です。

**主要な特徴 (Key Features)**

- **VRS (Variable Returns to Scale) の仮定**: 規模の増加・減少に応じて効率が変化することを許容
- **凸性制約 (Convexity Constraint)**: $\sum \lambda_j = 1$ という制約を追加することでVRSを実現
- **Pure Technical Efficiency の測定**: 規模の影響を除いた技術効率を測定

**CCRとBCCの違い (CCR vs. BCC)**

- **CCR効率**: Overall Technical Efficiency (総合技術効率)
- **BCC効率**: Pure Technical Efficiency (純粋技術効率)
- **Scale Efficiency (規模効率)**: $SE = \\frac{\\theta_{CCR}}{\\theta_{BCC}}$

**効率の分解 (Efficiency Decomposition)**

$$
\\text{Overall Technical Efficiency} = \\text{Pure Technical Efficiency} \\times \\text{Scale Efficiency}
$$
        """,
        "references": [
            "Banker, R. D., Charnes, A., & Cooper, W. W. (1984). Some models for estimating technical and scale inefficiencies in data envelopment analysis. Management Science, 30(9), 1078-1092."
        ],
        "formulation": r"""
**Input-Oriented Envelopment Model (VRS) (入力指向包絡モデル)**

**目的関数 (Objective Function):**
$$
\min \theta
$$

**制約条件 (Constraints):**
$$
\begin{align}
\sum_{j=1}^{n} \lambda_j x_{ij} &\leq \theta x_{ip}, \quad i=1,\ldots,m \\
\sum_{j=1}^{n} \lambda_j y_{rj} &\geq y_{rp}, \quad r=1,\ldots,s \\
\sum_{j=1}^{n} \lambda_j &= 1 \quad \text{(VRS constraint / 凸性制約)} \\
\lambda_j &\geq 0, \quad j=1,\ldots,n
\end{align}
$$

**Output-Oriented Envelopment Model (VRS) (出力指向包絡モデル)**

**目的関数 (Objective Function):**
$$
\max \phi
$$

**制約条件 (Constraints):**
$$
\begin{align}
\sum_{j=1}^{n} \lambda_j x_{ij} &\leq x_{ip}, \quad i=1,\ldots,m \\
\sum_{j=1}^{n} \lambda_j y_{rj} &\geq \phi y_{rp}, \quad r=1,\ldots,s \\
\sum_{j=1}^{n} \lambda_j &= 1 \quad \text{(VRS constraint / 凸性制約)} \\
\lambda_j &\geq 0, \quad j=1,\ldots,n
\end{align}
$$

**注意 (Note):** VRS制約 ($\sum \lambda_j = 1$) により、規模の収穫が可変であることを表現します。
"""
    },
    "Super-Efficiency": {
        "name": "Super-Efficiency",
        "explanation": """
**概要 (Overview)**

Super-Efficiencyモデルは、1993年にAndersenとPetersenによって提案された、効率的なDMUをランキングするためのモデルです。通常のDEAでは効率的なDMUはすべてスコア1となりますが、Super-Efficiencyモデルではこれらを区別できます。

**Super-Efficiency スコアの解釈 (Score Interpretation)**

- **スコア > 1.0**: 効率的 (efficient) - 値が大きいほど優れている
- **スコア = 1.0**: ちょうど効率的フロンティア上
- **スコア < 1.0**: 非効率的 (inefficient)

**特徴 (Key Features)**

- 評価対象DMUを参照集合から除外することで、効率的なDMUでもスコアが1を超えることが可能
- 効率的なDMU間のランキングが可能
- 非効率的なDMUのスコアは通常のDEAと同じ
        """,
        "references": [
            "Andersen, P., & Petersen, N. C. (1993). A procedure for ranking efficient units in data envelopment analysis. Management Science, 39(10), 1261-1264."
        ],
        "formulation": r"""
**Input-Oriented Super-Efficiency Model (入力指向Super-Efficiencyモデル)**

**目的関数 (Objective Function):**
$$
\min \theta
$$

**制約条件 (Constraints):**
$$
\begin{align}
\sum_{j=1, j \neq p}^{n} \lambda_j x_{ij} &\leq \theta x_{ip}, \quad i=1,\ldots,m \\
\sum_{j=1, j \neq p}^{n} \lambda_j y_{rj} &\geq y_{rp}, \quad r=1,\ldots,s \\
\lambda_j &\geq 0, \quad j=1,\ldots,n, \quad j \neq p
\end{align}
$$

**Output-Oriented Super-Efficiency Model (出力指向Super-Efficiencyモデル)**

**目的関数 (Objective Function):**
$$
\max \phi
$$

**制約条件 (Constraints):**
$$
\begin{align}
\sum_{j=1, j \neq p}^{n} \lambda_j x_{ij} &\leq x_{ip}, \quad i=1,\ldots,m \\
\sum_{j=1, j \neq p}^{n} \lambda_j y_{rj} &\geq \phi y_{rp}, \quad r=1,\ldots,s \\
\lambda_j &\geq 0, \quad j=1,\ldots,n, \quad j \neq p
\end{align}
$$

**重要な点 (Key Point):** 評価対象DMU $p$ を参照集合から除外することで、効率的なDMUでもスコアが1を超えることが可能になります。
"""
    },
    "Returns to Scale": {
        "name": "Returns to Scale (RTS) Analysis",
        "explanation": """
**概要 (Overview)**

Returns to Scale (RTS) 分析は、各DMUの規模の収穫状態を判定するためのモデルです。規模の経済性や非経済性を評価するために使用されます。

**RTS の種類 (Types of RTS)**

- **CRS (Constant Returns to Scale)**: 規模に関する収穫一定 - 規模に関係なく効率一定
- **IRS (Increasing Returns to Scale)**: 収穫逓増 - 規模拡大により効率向上
- **DRS (Decreasing Returns to Scale)**: 収穫逓減 - 規模拡大により効率低下
- **VRS (Variable Returns to Scale)**: 可変規模収穫 - 規模に応じて効率が変化

**判定方法 (Determination Method)**

CCRモデルとBCCモデルの効率スコアを比較し、規模効率 (Scale Efficiency) を計算することで判定します。
        """,
        "references": [
            "Banker, R. D. (1984). Estimating most productive scale size using data envelopment analysis. European Journal of Operational Research, 17(1), 35-44."
        ],
        "formulation": r"""
**Scale Efficiency (規模効率) の計算:**

$$
SE = \frac{\theta_{CCR}}{\theta_{BCC}}
$$

**変数の説明 (Variable Definitions):**
- $\theta_{CCR}$: CCRモデルによる効率スコア (Overall Technical Efficiency / 総合技術効率)
- $\theta_{BCC}$: BCCモデルによる効率スコア (Pure Technical Efficiency / 純粋技術効率)
- $SE$: Scale Efficiency (規模効率)

**RTS の判定方法 (RTS Determination):**

**CRS (Constant Returns to Scale) の場合:**
- $SE = 1$ かつ $\sum \lambda_j < 1$ (CCR包絡モデルで)

**VRS (Variable Returns to Scale) の場合:**
- $SE = 1$ かつ $\sum \lambda_j = 1$ (CCR包絡モデルで)

**IRS (Increasing Returns to Scale) の場合:**
- $SE < 1$ かつ $\sum \lambda_j < 1$ (BCC包絡モデルで)

**DRS (Decreasing Returns to Scale) の場合:**
- $SE < 1$ かつ $\sum \lambda_j > 1$ (BCC包絡モデルで)
"""
    },
    "Cost Efficiency": {
        "name": "Cost Efficiency Model",
        "explanation": """
**概要 (Overview)**

Cost Efficiency モデルは、入力価格（コスト）を考慮した効率測定モデルです。技術的効率と配分効率を分離して評価することができます。

**効率の分解 (Efficiency Decomposition)**

- **Cost Efficiency (CE) / コスト効率**: 実際のコストと最小コストの比率
- **Technical Efficiency (TE) / 技術的効率**: 技術的な入力削減余地
- **Allocative Efficiency (AE) / 配分効率**: 価格を考慮した入力配分の効率

**関係式 (Relationship):**
$$
CE = TE \\times AE
$$

**用途 (Applications)**

- コスト削減の余地の評価
- 価格を考慮した効率性の分析
- 技術的効率と配分効率の分離評価
        """,
        "references": [
            "Färe, R., Grosskopf, S., & Lovell, C. A. K. (1985). The Measurement of Efficiency of Production. Kluwer Academic Publishers."
        ],
        "formulation": r"""
**Cost Minimization Model (コスト最小化モデル)**

**目的関数 (Objective Function):**
$$
\min \sum_{i=1}^{m} c_i x_i^*
$$

**制約条件 (Constraints):**
$$
\begin{align}
\sum_{j=1}^{n} \lambda_j x_{ij} &\leq x_i^*, \quad i=1,\ldots,m \\
\sum_{j=1}^{n} \lambda_j y_{rj} &\geq y_{rp}, \quad r=1,\ldots,s \\
\sum_{j=1}^{n} \lambda_j &= 1 \quad \text{(VRS constraint)} \\
\lambda_j &\geq 0, \quad j=1,\ldots,n \\
x_i^* &\geq 0, \quad i=1,\ldots,m
\end{align}
$$

**Cost Efficiency (コスト効率) の計算:**

$$
CE = \frac{C^*}{C_0} = \frac{\sum_{i=1}^{m} c_i x_i^*}{\sum_{i=1}^{m} c_i x_{ip}}
$$

**変数の説明 (Variable Definitions):**
- $C^*$: 最小コスト (minimum cost)
- $C_0$: 実際のコスト (actual cost)
- $c_i$: 入力 $i$ の価格（コスト）(price/cost of input $i$)
- $x_i^*$: 最適な入力量 (optimal input quantity)
"""
    },
    "Revenue Efficiency": {
        "name": "Revenue Efficiency Model",
        "explanation": """
**概要 (Overview)**

Revenue Efficiency モデルは、出力価格を考慮した効率測定モデルです。現在の入力水準で達成可能な最大収益と実際の収益を比較します。

**特徴 (Key Features)**

- 出力価格を考慮した効率測定
- 現在の入力水準を維持しながら、収益を最大化する出力配分を決定
- 技術的効率と配分効率の分離評価が可能

**用途 (Applications)**

- 収益最大化の余地の評価
- 出力価格を考慮した効率性の分析
- 最適な出力配分の決定
        """,
        "references": [
            "Färe, R., Grosskopf, S., & Lovell, C. A. K. (1985). The Measurement of Efficiency of Production. Kluwer Academic Publishers."
        ],
        "formulation": r"""
**Revenue Maximization Model (収益最大化モデル)**

**目的関数 (Objective Function):**
$$
\max \sum_{r=1}^{s} p_r y_r^*
$$

**制約条件 (Constraints):**
$$
\begin{align}
\sum_{j=1}^{n} \lambda_j x_{ij} &\leq x_{ip}, \quad i=1,\ldots,m \\
\sum_{j=1}^{n} \lambda_j y_{rj} &\geq y_r^*, \quad r=1,\ldots,s \\
\sum_{j=1}^{n} \lambda_j &= 1 \quad \text{(VRS constraint)} \\
\lambda_j &\geq 0, \quad j=1,\ldots,n \\
y_r^* &\geq 0, \quad r=1,\ldots,s
\end{align}
$$

**Revenue Efficiency (収益効率) の計算:**

$$
RE = \frac{R_0}{R^*} = \frac{\sum_{r=1}^{s} p_r y_{rp}}{\sum_{r=1}^{s} p_r y_r^*}
$$

**変数の説明 (Variable Definitions):**
- $R_0$: 実際の収益 (actual revenue)
- $R^*$: 最大収益 (maximum revenue)
- $p_r$: 出力 $r$ の価格 (price of output $r$)
- $y_r^*$: 最適な出力量 (optimal output quantity)
"""
    },
    "Malmquist": {
        "name": "Malmquist Productivity Index",
        "explanation": """
**概要 (Overview)**

Malmquist Productivity Index (MPI) は、時系列データを用いて生産性の変化を測定するモデルです。技術効率の変化と技術進歩を分離して評価することができます。

**指数の分解 (Index Decomposition)**

- **Malmquist Index (MI) / マルムクイスト指数**: Total Factor Productivity Change (総要素生産性変化)
- **Efficiency Change (EFFCH) / 技術効率変化**: フロンティアへの接近/離反
- **Technical Change (TECHCH) / 技術変化**: フロンティア自体のシフト（技術進歩）

**関係式 (Relationship):**
$$
MI = EFFCH \\times TECHCH
$$

**解釈 (Interpretation)**

- **MI > 1**: 生産性向上 (productivity improvement)
- **MI = 1**: 生産性不変 (no productivity change)
- **MI < 1**: 生産性低下 (productivity decline)

**用途 (Applications)**

- 時系列データによる生産性変化の分析
- 技術進歩と効率改善の分離評価
- 産業や企業の生産性トレンド分析
        """,
        "references": [
            "Färe, R., Grosskopf, S., Norris, M., & Zhang, Z. (1994). Productivity growth, technical progress, and efficiency change in industrialized countries. American Economic Review, 84(1), 66-83."
        ],
        "formulation": r"""
**Malmquist Productivity Index (マルムクイスト生産性指数)**

$$
MI = \sqrt{\frac{D^t(x^{t+1}, y^{t+1})}{D^t(x^t, y^t)} \cdot \frac{D^{t+1}(x^{t+1}, y^{t+1})}{D^{t+1}(x^t, y^t)}}
$$

**Efficiency Change (EFFCH) / 技術効率変化**

$$
EFFCH = \frac{D^{t+1}(x^{t+1}, y^{t+1})}{D^t(x^t, y^t)}
$$

**Technical Change (TECHCH) / 技術変化**

$$
TECHCH = \sqrt{\frac{D^t(x^{t+1}, y^{t+1})}{D^{t+1}(x^{t+1}, y^{t+1})} \cdot \frac{D^t(x^t, y^t)}{D^{t+1}(x^t, y^t)}}
$$

**関係式 (Relationship)**

$$
MI = EFFCH \times TECHCH
$$

**変数の説明 (Variable Definitions)**

- $D^t(x^s, y^s)$: Distance function (距離関数) - 期間 $t$ の技術を基準とした期間 $s$ の距離関数
- $x^t, y^t$: 期間 $t$ の入力・出力ベクトル (input/output vectors in period $t$)
- $x^{t+1}, y^{t+1}$: 期間 $t+1$ の入力・出力ベクトル (input/output vectors in period $t+1$)

**注意 (Note):** 幾何平均を使用することで、基準期間の選択による偏りを回避します。
"""
    },
    "SBM": {
        "name": "SBM Model (Slacks-Based Measure)",
        "explanation": """
**概要 (Overview)**

SBM (Slacks-Based Measure) モデルは、2001年にToneによって提案された非放射的 (non-radial) 効率測定モデルです。入力と出力のスラック (slacks) を直接考慮するため、より正確な非効率性の測定が可能です。

**CCR/BCCとの違い (Differences from CCR/BCC)**

- **CCR/BCC**: Radial (放射的) - 全入力を同比率で削減
- **SBM**: Non-radial (非放射的) - 各入力を個別に評価
- **SBMの利点**: より厳密な効率評価を提供、スラックを直接考慮

**特徴 (Key Features)**

- スラックベースの効率測定
- 入力と出力のスラックを同時に考慮
- 単位不変性 (units invariant)
- 0から1の範囲で効率スコアを提供

**用途 (Applications)**

- スラックが重要な場合の効率評価
- より厳密な非効率性の測定が必要な場合
- 入力・出力の個別改善余地の評価
        """,
        "references": [
            "Tone, K. (2001). A slacks-based measure of efficiency in data envelopment analysis. European Journal of Operational Research, 130(3), 498-509."
        ],
        "formulation": r"""
**SBM Model 1 (Input-Oriented) / SBMモデル（入力指向）**

**目的関数 (Objective Function):**
$$
\min \rho = 1 - \frac{1}{m} \sum_{i=1}^{m} \frac{s_i^-}{x_{ip}}
$$

**制約条件 (Constraints):**
$$
\begin{align}
x_{ip} &= \sum_{j=1}^{n} \lambda_j x_{ij} + s_i^-, \quad i=1,\ldots,m \\
y_{rp} &\leq \sum_{j=1}^{n} \lambda_j y_{rj} - s_r^+, \quad r=1,\ldots,s \\
\sum_{j=1}^{n} \lambda_j &= 1 \quad \text{(VRS constraint, optional)} \\
\lambda_j &\geq 0, \quad j=1,\ldots,n \\
s_i^- &\geq 0, \quad i=1,\ldots,m \\
s_r^+ &\geq 0, \quad r=1,\ldots,s
\end{align}
$$

**SBM Model 2 (Output-Oriented) / SBMモデル（出力指向）**

**目的関数 (Objective Function):**
$$
\max \tau = 1 + \frac{1}{s} \sum_{r=1}^{s} \frac{s_r^+}{y_{rp}}
$$

**制約条件 (Constraints):**
$$
\begin{align}
x_{ip} &\geq \sum_{j=1}^{n} \lambda_j x_{ij} - s_i^-, \quad i=1,\ldots,m \\
y_{rp} &= \sum_{j=1}^{n} \lambda_j y_{rj} - s_r^+, \quad r=1,\ldots,s \\
\sum_{j=1}^{n} \lambda_j &= 1 \quad \text{(VRS constraint, optional)} \\
\lambda_j &\geq 0, \quad j=1,\ldots,n \\
s_i^- &\geq 0, \quad i=1,\ldots,m \\
s_r^+ &\geq 0, \quad r=1,\ldots,s
\end{align}
$$

**効率スコアの変換 (Efficiency Score Conversion)**

- **入力指向 (Input-oriented)**: 効率 = $\rho$
- **出力指向 (Output-oriented)**: 効率 = $1/\tau$

**変数の説明 (Variable Definitions)**

- $s_i^-$: Input slack (入力スラック) - 入力 $i$ の削減可能量
- $s_r^+$: Output slack (出力スラック) - 出力 $r$ の増加可能量
- $\rho$: SBM効率スコア（入力指向）(SBM efficiency score, input-oriented)
- $\tau$: SBM効率スコア（出力指向）(SBM efficiency score, output-oriented)
"""
    },
    "Directional Efficiency": {
        "name": "Directional Distance Function (DDF) Model",
        "explanation": """
**概要 (Overview)**

Directional Distance Function (DDF) モデルは、指定された方向への効率を測定するモデルです。入力と出力の改善方向を明示的に指定できるため、より柔軟な効率測定が可能です。

**特徴 (Key Features)**

- 方向ベクトル $g = (g_x, g_y)$ を明示的に指定可能
- 入力削減と出力増加を同時に考慮
- より柔軟な効率測定が可能

**方向ベクトルの選択 (Direction Vector Selection)**

- $g = (x_p, y_p)$: 入力削減＆出力増加（標準的 / standard）
- $g = (x_p, 0)$: 入力削減のみ (input reduction only)
- $g = (0, y_p)$: 出力増加のみ (output increase only)
- $g = (1, 1)$: 単位方向ベクトル (unit direction vector)

**効率スコア（$\\beta$）の解釈 (Efficiency Score Interpretation)**

- **$\\beta = 0$**: 効率的 (efficient) - フロンティア上
- **$\\beta > 0$**: 非効率的 (inefficient) - $\\beta$ の割合だけ改善可能

**注意 (Note):** 標準的なDDFモデルではスラックが0になることが一般的です。これは $\\beta$ の最大化に焦点を当てているためです。
        """,
        "references": [
            "Chambers, R. G., Chung, Y., & Färe, R. (1996). Benefit and distance functions. Journal of Economic Theory, 70(2), 407-419."
        ],
        "formulation": r"""
**Directional Distance Function (DDF) Model / 方向性距離関数モデル**

**目的関数 (Objective Function):**
$$
\max \beta
$$

**制約条件 (Constraints):**
$$
\begin{align}
\sum_{j=1}^{n} \lambda_j x_{ij} &\leq x_{ip} - \beta g_{xi}, \quad i=1,\ldots,m \\
\sum_{j=1}^{n} \lambda_j y_{rj} &\geq y_{rp} + \beta g_{yr}, \quad r=1,\ldots,s \\
\sum_{j=1}^{n} \lambda_j &= 1 \quad \text{(VRS constraint, optional)} \\
\lambda_j &\geq 0, \quad j=1,\ldots,n \\
\beta &\geq 0
\end{align}
$$

**方向ベクトル $g = (g_x, g_y)$ の選択 (Direction Vector Selection)**

- $g = (x_p, y_p)$: 入力・出力に比例した方向（標準的 / proportional direction）
- $g = (x_p, 0)$: 入力削減のみ (input reduction only)
- $g = (0, y_p)$: 出力増加のみ (output increase only)
- $g = (1, 1)$: 単位方向ベクトル (unit direction vector)

**効率スコアの解釈 (Efficiency Score Interpretation)**

- **$\beta = 0$**: 効率的 (efficient) - フロンティア上
- **$\beta > 0$**: 非効率的 (inefficient) - $\beta$ の割合だけ改善可能

**変数の説明 (Variable Definitions)**

- $\beta$: Directional efficiency score (方向性効率スコア)
- $g_{xi}$: 入力方向ベクトルの要素 $i$ (element $i$ of input direction vector)
- $g_{yr}$: 出力方向ベクトルの要素 $r$ (element $r$ of output direction vector)
"""
    },
    "Bootstrap DEA": {
        "name": "Bootstrap DEA",
        "explanation": """
**概要 (Overview)**

Bootstrap DEAは、1998年にSimarとWilsonによって提案された、DEA効率スコアの統計的推論を可能にする手法です。非パラメトリックフロンティアモデルにおける効率スコアの信頼区間を構築します。

**手順 (Procedure)**

1. 元のデータでDEA効率を計算
2. Bootstrapサンプルを多数生成（通常1000-2000回）
3. 各サンプルで効率を再計算
4. 効率スコアの分布から信頼区間を構築

**出力 (Output)**

- **Original Efficiency**: 元の効率スコア
- **Bias-Corrected Efficiency**: バイアス補正効率スコア
- **Confidence Interval**: 信頼区間（下限・上限）
- **Bias**: バイアス
- **Variance**: 分散

**用途 (Applications)**

- 効率スコアの統計的有意性の評価
- 信頼区間の構築
- バイアス補正によるより正確な効率推定
        """,
        "references": [
            "Simar, L., & Wilson, P. W. (1998). Sensitivity analysis of efficiency scores: How to bootstrap in nonparametric frontier models. Management Science, 44(11), 49-61."
        ],
        "formulation": r"""
**Bootstrap DEA の手順 (Procedure)**

**1. 初期効率の計算 (Initial Efficiency Calculation):**
$$
\hat{\theta}_j = \text{DEA}(x_j, y_j), \quad j=1,\ldots,n
$$
$$

**2. Bootstrapサンプルの生成 (Bootstrap Sample Generation):**
- Smoothed bootstrap: カーネル密度推定 (kernel density estimation) を使用
- 各反復 $b=1,\ldots,B$ で新しいサンプルを生成

**3. Bootstrap効率の計算 (Bootstrap Efficiency Calculation):**
$$
\hat{\theta}_j^{*(b)} = \text{DEA}(x_j^{*(b)}, y_j^{*(b)}), \quad b=1,\ldots,B
$$

**4. バイアス補正 (Bias Correction):**
$$
\hat{\theta}_j^{bc} = 2\hat{\theta}_j - \bar{\theta}_j^*
$$

ここで、$\bar{\theta}_j^* = \frac{1}{B}\sum_{b=1}^{B} \hat{\theta}_j^{*(b)}$ はBootstrap平均

**5. 信頼区間の構築 (Confidence Interval Construction):**
$$
CI_{1-\alpha} = [\hat{\theta}_j^{*(\alpha/2)}, \hat{\theta}_j^{*(1-\alpha/2)}]
$$

ここで、$\hat{\theta}_j^{*(q)}$ はBootstrap分布の $q$ 分位数 (quantile)

**変数の説明 (Variable Definitions)**

- $\hat{\theta}_j$: 元の効率スコア (original efficiency score)
- $\hat{\theta}_j^{*(b)}$: Bootstrap反復 $b$ での効率スコア
- $\hat{\theta}_j^{bc}$: バイアス補正効率スコア (bias-corrected efficiency score)
- $B$: Bootstrap反復回数 (number of bootstrap iterations)
- $\alpha$: 有意水準 (significance level)
"""
    },
    "Cross Efficiency": {
        "name": "Cross-Efficiency Analysis",
        "explanation": """
**概要 (Overview)**

Cross-Efficiency 分析は、各DMUの最適重みを使用して他のすべてのDMUの効率を評価する手法です。より公平で一貫性のある効率ランキングを提供します。

**特徴 (Key Features)**

- 各DMUの最適重みで他のすべてのDMUを評価
- 自己評価だけでなく、他者評価も考慮
- より公平で一貫性のある効率ランキング

**計算方法 (Calculation Method)**

1. 各DMUの最適重み $(u^*, v^*)$ を計算（乗数モデル）
2. 各DMUの重みで他のすべてのDMUの効率を計算
3. 平均Cross-Efficiencyスコアを算出

**Cross-Efficiency vs. Self-Efficiency**

- **Self-Efficiency**: 各DMUが自分の最適重みで評価された効率
- **Cross-Efficiency**: すべてのDMUの重みで評価された平均効率

**用途 (Applications)**

- より公平な効率ランキング
- 重み選択の恣意性の軽減
- 一貫性のある効率評価
        """,
        "references": [
            "Doyle, J., & Green, R. (1994). Efficiency and cross-efficiency in DEA: derivations, meanings and uses. Journal of the Operational Research Society, 45(5), 567-578."
        ],
        "formulation": r"""
**Cross-Efficiency の計算 (Cross-Efficiency Calculation)**

各DMU $k$ の最適重み $(u_k^*, v_k^*)$ を使用して、他のすべてのDMU $d$ の効率を評価:

$$
E_{dk} = \frac{\sum_{r=1}^{s} u_{rk}^* y_{rd}}{\sum_{i=1}^{m} v_{ik}^* x_{id}}, \quad d,k=1,\ldots,n
$$

**Average Cross-Efficiency (平均クロス効率)**

$$
\bar{E}_d = \frac{1}{n} \sum_{k=1}^{n} E_{dk}, \quad d=1,\ldots,n
$$

**Self-Efficiency (自己効率)**

$$
E_{dd} = \frac{\sum_{r=1}^{s} u_{rd}^* y_{rd}}{\sum_{i=1}^{m} v_{id}^* x_{id}}
$$

**重みの計算（Multiplier Model / 乗数モデル）**

各DMU $k$ について、以下の乗数モデルを解く:

$$
\begin{align}
\max &\quad \sum_{r=1}^{s} u_r y_{rk} \\
\text{s.t.} &\quad \sum_{i=1}^{m} v_i x_{ik} = 1 \\
&\quad \sum_{r=1}^{s} u_r y_{rj} - \sum_{i=1}^{m} v_i x_{ij} \leq 0, \quad j=1,\ldots,n \\
&\quad u_r \geq \epsilon, \quad v_i \geq \epsilon
\end{align}
$$

**変数の説明 (Variable Definitions)**

- $E_{dk}$: DMU $k$ の重みで評価したDMU $d$ の効率
- $\bar{E}_d$: DMU $d$ の平均Cross-Efficiency
- $E_{dd}$: DMU $d$ のSelf-Efficiency
- $u_{rk}^*$: DMU $k$ の最適出力重み (optimal output weights)
- $v_{ik}^*$: DMU $k$ の最適入力重み (optimal input weights)
- $\epsilon$: 最小重み値 (minimum weight value)
"""
    }
}
st.sidebar.title("ナビゲーション")
page = st.sidebar.selectbox(
    "ページを選択",
    ["データアップロード", "モデル分析"]
)
if 'data' not in st.session_state:
    st.session_state.data = None
if 'inputs' not in st.session_state:
    st.session_state.inputs = None
if 'outputs' not in st.session_state:
    st.session_state.outputs = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'dmu_names' not in st.session_state:
    st.session_state.dmu_names = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if page == "データアップロード":
    st.header("データのアップロード")

    st.markdown("""
    ### データ形式について

    CSVファイルをアップロードしてください。以下の形式が必要です：
    - 最初の列: DMU名（オプション）
    - 次の列: 入力変数（複数可）
    - 最後の列: 出力変数（複数可）

    **基本データの例:**
    ```
    DMU,Input1,Input2,Output1,Output2
    A,2,3,1,2
    B,3,2,2,3
    C,4,1,3,4
    ```

    **Malmquist指数用の時系列データの例:**
    ```
    DMU,Period,Input1,Input2,Output1
    A,1,10,5,8
    A,2,9,5,9
    B,1,15,8,12
    B,2,14,7,13
    C,1,12,6,10
    C,2,11,6,11
    ```
    """)

    uploaded_file = st.file_uploader("CSVファイルをアップロード", type=['csv'])
if uploaded_file is not None:




        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df

            st.success(f"データが正常に読み込まれました: {len(df)} 行")
            st.dataframe(df.head(10))

            st.subheader("列の選択")
            all_columns = df.columns.tolist()

            dmu_col = st.selectbox(
                "DMU名の列（オプション）",
                ["なし"] + all_columns,
                index=0
            )
            if dmu_col != "なし":
                st.session_state.dmu_names = df[dmu_col].values
                remaining_cols = [col for col in all_columns if col != dmu_col]
            else:
                st.session_state.dmu_names = None
                remaining_cols = all_columns

            input_cols = st.multiselect(
                "入力変数を選択",
                remaining_cols,
                default=remaining_cols[:len(remaining_cols)//2] if len(remaining_cols) > 2 else remaining_cols[:1]
            )

            output_cols = st.multiselect(
                "出力変数を選択",
                [col for col in remaining_cols if col not in input_cols],
                default=[col for col in remaining_cols if col not in input_cols][:len(remaining_cols)//2]
            )

            if st.button("データを設定", type="primary"):
                if len(input_cols) > 0 and len(output_cols) > 0:
                    st.session_state.inputs = df[input_cols].values
                    st.session_state.outputs = df[output_cols].values
                    st.success(f"設定完了: {len(input_cols)} 入力, {len(output_cols)} 出力")
                else:
                    st.error("入力変数と出力変数を少なくとも1つずつ選択してください")
        except Exception as e:
            st.error(f"エラー: {str(e)}")


    st.subheader("サンプルデータの生成")



    sample_templates = {
        "基本データセット（10 DMU）": {"n_dmus": 10, "n_inputs": 2, "n_outputs": 2, "seed": 42, "time_periods": False},
        "Malmquist用時系列データ": {"n_dmus": 10, "n_inputs": 2, "n_outputs": 2, "seed": 42, "time_periods": True}
    }

    selected_template = st.selectbox("サンプルデータテンプレート", list(sample_templates.keys()))
    template = sample_templates[selected_template]

    if st.button("サンプルデータを生成"):
        np.random.seed(template["seed"])
        n_dmus = template["n_dmus"]
        n_inputs = template["n_inputs"]
        n_outputs = template["n_outputs"]

        base_efficiency = np.random.uniform(0.6, 1.0, n_dmus)

        if template.get("time_periods", False):
            sample_data_t1 = {'DMU': [f'DMU_{i+1}' for i in range(n_dmus)], 'Period': [1] * n_dmus}
            sample_data_t2 = {'DMU': [f'DMU_{i+1}' for i in range(n_dmus)], 'Period': [2] * n_dmus}

            for i in range(n_inputs):
                base_input = np.random.uniform(5, 15, n_dmus)
                sample_data_t1[f'Input_{i+1}'] = np.round(base_input / (base_efficiency + 0.1), 2)
                sample_data_t2[f'Input_{i+1}'] = np.round(base_input * np.random.uniform(0.85, 1.05, n_dmus) / (base_efficiency + 0.15), 2)

            for i in range(n_outputs):
                base_output = np.random.uniform(3, 12, n_dmus)
                sample_data_t1[f'Output_{i+1}'] = np.round(base_output * (base_efficiency + 0.2), 2)
                sample_data_t2[f'Output_{i+1}'] = np.round(base_output * np.random.uniform(1.0, 1.2, n_dmus) * (base_efficiency + 0.25), 2)

            df_t1 = pd.DataFrame(sample_data_t1)
            df_t2 = pd.DataFrame(sample_data_t2)
            df_sample = pd.concat([df_t1, df_t2], ignore_index=True)

            input_cols = [f'Input_{i+1}' for i in range(n_inputs)]
            output_cols = [f'Output_{i+1}' for i in range(n_outputs)]
            st.session_state.inputs_t = df_t1[input_cols].values
            st.session_state.outputs_t = df_t1[output_cols].values
            st.session_state.inputs_t1 = df_t2[input_cols].values
            st.session_state.outputs_t1 = df_t2[output_cols].values
            st.session_state.inputs = st.session_state.inputs_t
            st.session_state.outputs = st.session_state.outputs_t
            st.session_state.dmu_names = df_t1['DMU'].values
        else:
            sample_data = {'DMU': [f'DMU_{i+1}' for i in range(n_dmus)]}
            for i in range(n_inputs):
                base_input = np.random.uniform(5, 15, n_dmus)
                sample_data[f'Input_{i+1}'] = np.round(base_input / (base_efficiency + 0.1), 2)
            for i in range(n_outputs):
                base_output = np.random.uniform(3, 12, n_dmus)
                sample_data[f'Output_{i+1}'] = np.round(base_output * (base_efficiency + 0.2), 2)

            df_sample = pd.DataFrame(sample_data)
            input_cols = [f'Input_{i+1}' for i in range(n_inputs)]
            output_cols = [f'Output_{i+1}' for i in range(n_outputs)]
            st.session_state.inputs = df_sample[input_cols].values
            st.session_state.outputs = df_sample[output_cols].values
            st.session_state.dmu_names = df_sample['DMU'].values

        st.session_state.data = df_sample

        st.success(f"サンプルデータを生成しました: {n_dmus} DMUs")
        st.dataframe(df_sample)
elif page == "モデル分析":
    st.header("DEAモデル分析")

    if st.session_state.inputs is None or st.session_state.outputs is None:
        st.warning("まず「データアップロード」ページでデータを設定してください")
    else:
        all_models = [
            "CCR", "BCC", "Super-Efficiency", "Returns to Scale",
            "Cost Efficiency", "Revenue Efficiency", "Malmquist",
            "SBM", "Directional Efficiency", "Bootstrap DEA", "Cross Efficiency"
        ]

        model_type = st.selectbox("モデルを選択", all_models)
        st.session_state.model_type = model_type

        if model_type in MODEL_INFO:
            st.subheader(f"{MODEL_INFO[model_type]['name']}")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### モデルの詳細解説")
                st.markdown(MODEL_INFO[model_type]['explanation'])

            with col2:
                st.markdown("### 数学的定式化")
                formulation_parts = MODEL_INFO[model_type]['formulation'].strip().split('\n\n')
                for part in formulation_parts:
                    if part.strip():
                        st.markdown(part.strip())

            st.markdown("---")
            st.markdown("### 参考文献")
            for ref in MODEL_INFO[model_type]['references']:
                st.markdown(f"- {ref}")

        st.markdown("---")

        orientation = None
        method = None
        input_costs = None
        output_prices = None
        g_inputs = None
        g_outputs = None
        sbm_type = "Model 1"
        ap_orientation = "入力指向"
        rts = "vrs"
        n_bootstrap = 1000

        st.subheader("パラメータ設定")

        if model_type in ["CCR", "BCC"]:
            col1, col2 = st.columns(2)
            with col1:
                orientation = st.selectbox("方向", ["入力指向", "出力指向"])
            with col2:
                method = st.selectbox("方法", ["包絡モデル", "乗数モデル"])

        elif model_type == "Super-Efficiency":
            ap_orientation = st.selectbox("方向", ["入力指向", "出力指向"])

        elif model_type == "SBM":
            col1, col2 = st.columns(2)
            with col1:
                sbm_type = st.selectbox("タイプ", ["Model 1 (入力指向)", "Model 2 (出力指向)"])
            with col2:
                rts = st.selectbox("規模の収穫", ["vrs", "crs", "drs", "irs"])

        elif model_type == "Cost Efficiency":
            st.write("入力コスト（カンマ区切り）:")
            cost_str = st.text_input("例: 1.0, 2.0", value=", ".join(["1.0"] * st.session_state.inputs.shape[1]))
            try:
                input_costs = np.array([float(x.strip()) for x in cost_str.split(",")])
            except:
                st.error("コストの形式が正しくありません")
                input_costs = None

        elif model_type == "Revenue Efficiency":
            st.write("出力価格（カンマ区切り）:")
            price_str = st.text_input("例: 1.0, 2.0", value=", ".join(["1.0"] * st.session_state.outputs.shape[1]))
            try:
                output_prices = np.array([float(x.strip()) for x in price_str.split(",")])
            except:
                st.error("価格の形式が正しくありません")
                output_prices = None

        elif model_type == "Directional Efficiency":
            st.markdown("""
            **方向ベクトルの設定:**
            - 入力方向 (gx): 入力削減の方向（正の値）
            - 出力方向 (gy): 出力増加の方向（正の値）
            - 空欄の場合、各DMUの入力・出力値に比例した方向が使用されます
            """)

            col1, col2 = st.columns(2)
            with col1:
                gx_str = st.text_input(
                    f"入力方向ベクトル gx ({st.session_state.inputs.shape[1]}個)",
                    value="",
                    placeholder="例: 1, 1（空欄で自動設定）"
                )
            with col2:
                gy_str = st.text_input(
                    f"出力方向ベクトル gy ({st.session_state.outputs.shape[1]}個)",
                    value="",
                    placeholder="例: 1, 1（空欄で自動設定）"
                )

            g_inputs = None
            g_outputs = None
            if gx_str.strip():
                try:
                    g_inputs = np.array([float(x.strip()) for x in gx_str.split(",")])
                    if len(g_inputs) != st.session_state.inputs.shape[1]:
                        st.error(f"入力方向ベクトルは{st.session_state.inputs.shape[1]}個の値が必要です")
                        g_inputs = None
                except:
                    st.error("入力方向ベクトルの形式が正しくありません")

            if gy_str.strip():
                try:
                    g_outputs = np.array([float(x.strip()) for x in gy_str.split(",")])
                    if len(g_outputs) != st.session_state.outputs.shape[1]:
                        st.error(f"出力方向ベクトルは{st.session_state.outputs.shape[1]}個の値が必要です")
                        g_outputs = None
                except:
                    st.error("出力方向ベクトルの形式が正しくありません")

            rts = st.selectbox("規模の収穫", ["vrs", "crs", "drs", "irs"])

        elif model_type == "Malmquist":
            if not hasattr(st.session_state, 'inputs_t') or st.session_state.inputs_t is None:
                st.warning("Malmquistモデルには時系列データが必要です。「Malmquist用時系列データ」テンプレートを使用してください。")

        elif model_type == "Bootstrap DEA":
            col1, col2 = st.columns(2)
            with col1:
                n_bootstrap = st.number_input("ブートストラップ回数", min_value=100, max_value=2000, value=500, step=100)
            with col2:
                rts = st.selectbox("規模の収穫", ["vrs", "crs"])
            orientation = st.selectbox("方向", ["入力指向", "出力指向"])

        elif model_type == "Cross Efficiency":
            col1, col2 = st.columns(2)
            with col1:
                orientation = st.selectbox("方向", ["入力指向", "出力指向"])
            with col2:
                rts = st.selectbox("規模の収穫", ["vrs", "crs"])

        st.markdown("---")

        if st.button("分析を実行", type="primary"):
            try:
                with st.spinner("計算中..."):
                    results = None

                    if model_type == "CCR":
                        model = CCRModel(st.session_state.inputs, st.session_state.outputs)
                        if method == "包絡モデル":
                            if orientation == "入力指向":
                                results = model.evaluate_all(method='envelopment')
                            else:
                                results_list = []
                                for i in range(len(st.session_state.inputs)):
                                    eff, lambdas, _, _ = model.solve_output_oriented_envelopment(i)
                                    results_list.append({'DMU': i+1, 'Efficiency': eff})
                                results = pd.DataFrame(results_list)
                        else:

                            results = model.evaluate_all(method='multiplier')

                    elif model_type == "BCC":
                        model = BCCModel(st.session_state.inputs, st.session_state.outputs)
                        if method == "包絡モデル":
                            results = model.evaluate_all(method='envelopment')
else:
                            results = model.evaluate_all(method='multiplier')

                    elif model_type == "Super-Efficiency":
                        model = APModel(st.session_state.inputs, st.session_state.outputs)
                        orient = 'input' if ap_orientation == "入力指向" else 'output'
                        results = model.evaluate_all(orientation=orient, method='envelopment')

                    elif model_type == "Returns to Scale":
                        model = ReturnsToScaleModel(st.session_state.inputs, st.session_state.outputs)
                        results = model.evaluate_all()

                    elif model_type == "Cost Efficiency":
                        if input_costs is not None:
                            model = CostEfficiencyModel(st.session_state.inputs, st.session_state.outputs, input_costs)
                            results = model.evaluate_all()

                    elif model_type == "Revenue Efficiency":
                        if output_prices is not None:
                            model = RevenueEfficiencyModel(st.session_state.inputs, st.session_state.outputs, output_prices)
                            results = model.evaluate_all()

                    elif model_type == "Malmquist":
                        if hasattr(st.session_state, 'inputs_t') and st.session_state.inputs_t is not None:
                            model = MalmquistModel(
                                st.session_state.inputs_t, st.session_state.outputs_t,
                                st.session_state.inputs_t1, st.session_state.outputs_t1
                            )
                            results = model.evaluate_all()

                    elif model_type == "SBM":
                        model = SBMModel(st.session_state.inputs, st.session_state.outputs)
                        model_type_num = 1 if "Model 1" in sbm_type else 2
                        results = model.evaluate_all(model_type=model_type_num, rts=rts)

                    elif model_type == "Directional Efficiency":
                        model = DirectionalEfficiencyModel(st.session_state.inputs, st.session_state.outputs)
                        results = model.evaluate_all(gx=g_inputs, gy=g_outputs, rts=rts)

                    elif model_type == "Bootstrap DEA":
                        orient = 'in' if orientation == "入力指向" else 'out'
                        model = BootstrapDEAModel(st.session_state.inputs, st.session_state.outputs, rts=rts, orientation=orient)
                        results = model.evaluate_all(n_rep=n_bootstrap)

                    elif model_type == "Cross Efficiency":
                        model = CrossEfficiencyModel(st.session_state.inputs, st.session_state.outputs)
                        orient = 'io' if orientation == "入力指向" else 'oo'
                        results = model.evaluate_all(orientation=orient, rts=rts)

                    if results is not None:
                        st.session_state.results = results
                        st.success("分析が完了しました")

            except Exception as e:
                st.error(f"エラー: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

        if st.session_state.results is not None:
            st.markdown("---")
            st.subheader("分析結果")

            results = st.session_state.results
            st.dataframe(results, use_container_width=True)

            csv = results.to_csv(index=False)
            st.download_button(
                label="結果をCSVでダウンロード",
                data=csv,
                file_name=f"{model_type.replace(' ', '_')}_results.csv",
                mime="text/csv"
            )
st.sidebar.markdown("---")
st.sidebar.markdown("### 情報")
st.sidebar.info("""
対応モデル:
- CCR, BCC, Super-Efficiency
- Returns to Scale, Cost/Revenue Efficiency
- Malmquist, SBM, Directional Efficiency
- Bootstrap DEA, Cross Efficiency
""")
