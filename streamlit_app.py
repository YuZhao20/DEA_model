"""
Streamlit App for DEA Models
Interactive web interface for Data Envelopment Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Import all DEA models
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

# Page configuration
st.set_page_config(
    page_title="DEA Model Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📊 DEA Model Analyzer")
st.markdown("Data Envelopment Analysis (DEA) モデルのインタラクティブ分析ツール")

# Model explanations and references
MODEL_INFO = {
    "CCR": {
        "name": "CCR (Charnes-Cooper-Rhodes) モデル",
        "explanation": "CCRモデルは、定規模収穫（Constant Returns to Scale, CRS）を仮定した基本的なDEAモデルです。1978年にCharnes、Cooper、Rhodesによって提案され、DEAの基礎となるモデルです。このモデルは、各DMU（Decision Making Unit）の効率を、他のすべてのDMUの線形結合として表現できる効率的なDMUとの比較によって測定します。入力指向では、現在の出力水準を維持しながら入力の削減余地を測定し、出力指向では、現在の入力水準を維持しながら出力の増加余地を測定します。",
        "references": [
            "Charnes, A., Cooper, W. W., & Rhodes, E. (1978). Measuring the efficiency of decision making units. *European Journal of Operational Research*, 2(6), 429-444.",
            "Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer. (Chapter 3.2)"
        ]
    },
    "BCC": {
        "name": "BCC (Banker-Charnes-Cooper) モデル",
        "explanation": "BCCモデルは、可変規模収穫（Variable Returns to Scale, VRS）を仮定したDEAモデルです。1984年にBanker、Charnes、Cooperによって提案されました。CCRモデルと異なり、BCCモデルは規模の収穫が可変であることを考慮します。これにより、規模の経済性や非経済性を考慮した効率測定が可能になります。BCCモデルは、小規模なDMUと大規模なDMUをより公平に比較できるため、実務で広く使用されています。",
        "references": [
            "Banker, R. D., Charnes, A., & Cooper, W. W. (1984). Some models for estimating technical and scale inefficiencies in data envelopment analysis. *Management Science*, 30(9), 1078-1092.",
            "Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer. (Chapter 3.2.3)"
        ]
    },
    "AP (Super-Efficiency)": {
        "name": "AP (Anderson-Peterson) スーパー効率モデル",
        "explanation": "APモデルは、効率的なDMU（効率スコアが1のDMU）をランキングするためのスーパー効率モデルです。1993年にAndersonとPetersonによって提案されました。通常のDEAモデルでは、効率的なDMUはすべて効率スコア1となり、それらを区別できません。APモデルでは、評価対象のDMUを参照集合から除外することで、効率的なDMUの効率スコアが1を超える値を取ることができ、効率的なDMU間のランキングが可能になります。スーパー効率スコアが1より大きいほど、そのDMUはより効率的であることを示します。",
        "references": [
            "Andersen, P., & Petersen, N. C. (1993). A procedure for ranking efficient units in data envelopment analysis. *Management Science*, 39(10), 1261-1264.",
            "Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer. (Chapter 4.2)"
        ]
    },
    "Returns to Scale": {
        "name": "規模の収穫モデル",
        "explanation": "規模の収穫モデルは、各DMUの規模の収穫（Returns to Scale, RTS）を判定するためのモデルです。規模の収穫には、定規模収穫（CRS）、可変規模収穫（VRS）、収穫逓減（DRS）、収穫逓増（IRS）があります。このモデルは、各DMUが最適規模にあるか、規模を拡大または縮小すべきかを判断するために使用されます。規模の収穫の判定は、効率改善のための戦略的指針を提供します。",
        "references": [
            "Banker, R. D. (1984). Estimating most productive scale size using data envelopment analysis. *European Journal of Operational Research*, 17(1), 35-44.",
            "Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer. (Chapter 4.5)"
        ]
    },
    "Cost Efficiency": {
        "name": "コスト効率モデル",
        "explanation": "コスト効率モデルは、入力コストを考慮した効率測定モデルです。このモデルは、技術的効率だけでなく、コスト効率も測定します。コスト効率は、現在の出力水準を維持しながら、最小コストで達成可能な入力の組み合わせと、実際のコストとの比率として定義されます。コスト効率は、技術的効率と配分効率の積として分解できます。このモデルは、価格情報が利用可能な場合に、より実用的な効率評価を提供します。",
        "references": [
            "Färe, R., Grosskopf, S., & Lovell, C. A. K. (1985). *The Measurement of Efficiency of Production*. Kluwer Academic Publishers.",
            "Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer. (Chapter 4.6)"
        ]
    },
    "Revenue Efficiency": {
        "name": "収益効率モデル",
        "explanation": "収益効率モデルは、出力価格を考慮した効率測定モデルです。このモデルは、現在の入力水準を維持しながら、最大収益で達成可能な出力の組み合わせと、実際の収益との比率として定義されます。収益効率は、技術的効率と配分効率の積として分解できます。このモデルは、出力の価格情報が利用可能な場合に、収益最大化の観点から効率評価を提供します。",
        "references": [
            "Färe, R., Grosskopf, S., & Lovell, C. A. K. (1985). *The Measurement of Efficiency of Production*. Kluwer Academic Publishers.",
            "Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer. (Chapter 4.7)"
        ]
    },
    "Malmquist": {
        "name": "Malmquist生産性指数",
        "explanation": "Malmquist生産性指数は、時系列データを用いて生産性の変化を測定するモデルです。1953年にMalmquistによって提案され、1994年にFäreらによってDEAに適用されました。この指数は、2つの時点間の生産性変化を、技術的効率の変化（Efficiency Change, EC）と技術進歩（Technical Change, TC）に分解します。Malmquist指数が1より大きい場合、生産性が向上したことを示し、1より小さい場合、生産性が低下したことを示します。",
        "references": [
            "Malmquist, S. (1953). Index numbers and indifference surfaces. *Trabajos de Estadística*, 4(2), 209-242.",
            "Färe, R., Grosskopf, S., Norris, M., & Zhang, Z. (1994). Productivity growth, technical progress, and efficiency change in industrialized countries. *American Economic Review*, 84(1), 66-83.",
            "Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer. (Chapter 4.8)"
        ]
    },
    "SBM": {
        "name": "SBM (Slacks-Based Measure) モデル",
        "explanation": "SBMモデルは、スラックに基づく非放射的効率測定モデルです。2001年にToneによって提案されました。従来の放射的DEAモデル（CCR、BCC）とは異なり、SBMモデルは入力と出力のスラックを直接考慮するため、非効率性の測定がより正確になります。SBM効率は0から1の間の値を取り、1に近いほど効率的であることを示します。このモデルは、入力と出力の両方のスラックを同時に考慮するため、より包括的な効率評価を提供します。",
        "references": [
            "Tone, K. (2001). A slacks-based measure of efficiency in data envelopment analysis. *European Journal of Operational Research*, 130(3), 498-509.",
            "Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer. (Chapter 4.9)"
        ]
    },
    "Directional Efficiency": {
        "name": "方向性効率モデル",
        "explanation": "方向性効率モデルは、指定された方向への効率を測定するモデルです。このモデルは、入力と出力の改善方向を明示的に指定できるため、より柔軟な効率測定が可能です。従来の放射的DEAモデルは、入力指向または出力指向のいずれか一方のみを考慮しますが、方向性効率モデルでは、入力と出力の両方を同時に改善する方向を指定できます。このモデルは、特定の改善戦略に基づいた効率評価を提供します。",
        "references": [
            "Chambers, R. G., Chung, Y., & Färe, R. (1996). Benefit and distance functions. *Journal of Economic Theory*, 70(2), 407-419.",
            "Hosseinzadeh Lotfi, F., Hatami-Marbini, A., Agrell, P. J., Aghayi, N., & Gholami, K. (2020). *Data Envelopment Analysis with R*. Springer. (Chapter 4.15)"
        ]
    },
    "Bootstrap DEA": {
        "name": "ブートストラップDEA",
        "explanation": "ブートストラップDEAは、DEA効率スコアの統計的推論を可能にするモデルです。1998年にSimarとWilsonによって提案されました。DEAは非パラメトリックな手法であるため、従来の統計的推論が困難でした。ブートストラップ法を用いることで、効率スコアの信頼区間やバイアス補正を提供し、効率評価の統計的有意性を評価できます。このモデルは、サンプルサイズが小さい場合や、効率スコアの不確実性を考慮したい場合に特に有用です。",
        "references": [
            "Simar, L., & Wilson, P. W. (1998). Sensitivity analysis of efficiency scores: How to bootstrap in nonparametric frontier models. *Management Science*, 44(11), 49-61.",
            "Simar, L., & Wilson, P. W. (2000). Statistical inference in nonparametric frontier models: The state of the art. *Journal of Productivity Analysis*, 13(1), 49-78.",
            "Bogetoft, P., & Otto, L. (2011). *Benchmarking with DEA, SFA, and R*. Springer-Verlag."
        ]
    },
    "Cross Efficiency": {
        "name": "クロス効率分析",
        "explanation": "クロス効率分析は、各DMUの重みを使用して他のDMUの効率を評価する手法です。1994年にDoyleとGreenによって提案されました。従来のDEAでは、各DMUは自分に最も有利な重みを選択するため、自己効率スコアが過大評価される可能性があります。クロス効率分析では、各DMUの重みを使用して他のすべてのDMUの効率を評価し、平均クロス効率スコアを計算します。これにより、より公平で一貫性のある効率ランキングが得られます。",
        "references": [
            "Doyle, J., & Green, R. (1994). Efficiency and cross-efficiency in DEA: derivations, meanings and uses. *Journal of the Operational Research Society*, 45(5), 567-578.",
            "Sexton, T. R., Silkman, R. H., & Hogan, A. J. (1986). Data envelopment analysis: Critique and extensions. *New Directions for Program Evaluation*, 1986(32), 73-105."
        ]
    }
}

# Sidebar for navigation
st.sidebar.title("ナビゲーション")
page = st.sidebar.selectbox(
    "ページを選択",
    ["データアップロード", "モデル分析", "結果の可視化"]
)

# Initialize session state
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

# Data Upload Page
if page == "データアップロード":
    st.header("📁 データのアップロード")
    
    st.markdown("""
    ### データ形式について
    
    CSVファイルをアップロードしてください。以下の形式が必要です：
    - 最初の列: DMU名（オプション）
    - 次の列: 入力変数（複数可）
    - 最後の列: 出力変数（複数可）
    
    **例:**
    ```
    DMU,Input1,Input2,Output1,Output2
    A,2,3,1,2
    B,3,2,2,3
    C,4,1,3,4
    ```
    """)
    
    uploaded_file = st.file_uploader("CSVファイルをアップロード", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            
            st.success(f"データが正常に読み込まれました: {len(df)} DMUs")
            st.dataframe(df.head(10))
            
            # Column selection
            st.subheader("列の選択")
            all_columns = df.columns.tolist()
            
            # DMU name column (optional)
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
            
            # Input columns
            input_cols = st.multiselect(
                "入力変数を選択",
                remaining_cols,
                default=remaining_cols[:len(remaining_cols)//2] if len(remaining_cols) > 2 else remaining_cols[:1]
            )
            
            # Output columns
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

    # Sample data generator (available even without file upload)
    st.subheader("サンプルデータの生成")
    
    # Sample data templates
    sample_templates = {
        "基本データセット（小規模）": {"n_dmus": 10, "n_inputs": 2, "n_outputs": 2, "seed": 42},
        "基本データセット（中規模）": {"n_dmus": 20, "n_inputs": 3, "n_outputs": 2, "seed": 42},
        "基本データセット（大規模）": {"n_dmus": 30, "n_inputs": 3, "n_outputs": 3, "seed": 42},
        "StoNED用（単一出力）": {"n_dmus": 15, "n_inputs": 2, "n_outputs": 1, "seed": 42},
        "複数入力・出力": {"n_dmus": 25, "n_inputs": 4, "n_outputs": 3, "seed": 42},
        "時系列データ（Malmquist用）": {"n_dmus": 20, "n_inputs": 2, "n_outputs": 2, "seed": 42, "time_periods": True}
    }
    
    selected_template = st.selectbox(
        "サンプルデータテンプレートを選択",
        list(sample_templates.keys())
    )
    
    template = sample_templates[selected_template]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        n_dmus = st.number_input("DMU数", min_value=5, max_value=100, value=template["n_dmus"], step=5)
    with col2:
        n_inputs = st.number_input("入力変数数", min_value=1, max_value=10, value=template["n_inputs"], step=1)
    with col3:
        n_outputs = st.number_input("出力変数数", min_value=1, max_value=10, value=template["n_outputs"], step=1)
    
    if st.button("サンプルデータを生成", type="primary"):
        np.random.seed(template["seed"])
        
        # Generate realistic DEA data
        # Create efficient frontier first
        base_efficiency = np.random.uniform(0.7, 1.0, n_dmus)
        
        sample_data = {
            'DMU': [f'DMU_{i+1}' for i in range(n_dmus)],
        }
        
        # Generate inputs (vary with efficiency)
        for i in range(n_inputs):
            # More efficient DMUs use fewer inputs
            base_input = np.random.uniform(5, 15, n_dmus)
            inputs = base_input / (base_efficiency + 0.1)  # Inverse relationship with efficiency
            sample_data[f'Input_{i+1}'] = inputs
        
        # Generate outputs (vary with efficiency)
        for i in range(n_outputs):
            # More efficient DMUs produce more outputs
            base_output = np.random.uniform(3, 12, n_dmus)
            outputs = base_output * (base_efficiency + 0.2)  # Positive relationship with efficiency
            sample_data[f'Output_{i+1}'] = outputs
        
        # Add time periods for Malmquist if needed
        if template.get("time_periods", False):
            # For Malmquist, each DMU needs data for both time periods
            # Generate data for period 1
            base_efficiency_t = np.random.uniform(0.7, 1.0, n_dmus)
            sample_data_t = {
                'DMU': [f'DMU_{i+1}' for i in range(n_dmus)],
                'Period': [1] * n_dmus
            }
            
            # Generate inputs and outputs for period 1
            for i in range(n_inputs):
                base_input = np.random.uniform(5, 15, n_dmus)
                inputs = base_input / (base_efficiency_t + 0.1)
                sample_data_t[f'Input_{i+1}'] = inputs
            
            for i in range(n_outputs):
                base_output = np.random.uniform(3, 12, n_dmus)
                outputs = base_output * (base_efficiency_t + 0.2)
                sample_data_t[f'Output_{i+1}'] = outputs
            
            # Generate data for period 2 (with some improvement/degradation)
            base_efficiency_t1 = base_efficiency_t + np.random.uniform(-0.1, 0.15, n_dmus)
            base_efficiency_t1 = np.clip(base_efficiency_t1, 0.6, 1.0)
            
            sample_data_t1 = {
                'DMU': [f'DMU_{i+1}' for i in range(n_dmus)],
                'Period': [2] * n_dmus
            }
            
            # Generate inputs and outputs for period 2
            for i in range(n_inputs):
                base_input = np.random.uniform(5, 15, n_dmus)
                inputs = base_input / (base_efficiency_t1 + 0.1)
                sample_data_t1[f'Input_{i+1}'] = inputs
            
            for i in range(n_outputs):
                base_output = np.random.uniform(3, 12, n_dmus)
                outputs = base_output * (base_efficiency_t1 + 0.2)
                sample_data_t1[f'Output_{i+1}'] = outputs
            
            # Combine both periods
            df_t = pd.DataFrame(sample_data_t)
            df_t1 = pd.DataFrame(sample_data_t1)
            df_sample = pd.concat([df_t, df_t1], ignore_index=True)
            
            # Sort by DMU and Period
            df_sample = df_sample.sort_values(['DMU', 'Period']).reset_index(drop=True)
        else:
            df_sample = pd.DataFrame(sample_data)
        
        st.session_state.data = df_sample
        
        # For Malmquist, we need to separate data by period
        if template.get("time_periods", False):
            # Store period 1 and period 2 data separately for Malmquist
            df_t = df_sample[df_sample['Period'] == 1].copy()
            df_t1 = df_sample[df_sample['Period'] == 2].copy()
            
            # Ensure both periods have the same DMUs
            common_dmus = set(df_t['DMU'].unique()) & set(df_t1['DMU'].unique())
            df_t = df_t[df_t['DMU'].isin(common_dmus)].sort_values('DMU').reset_index(drop=True)
            df_t1 = df_t1[df_t1['DMU'].isin(common_dmus)].sort_values('DMU').reset_index(drop=True)
            
            st.session_state.inputs_t = df_t[[f'Input_{i+1}' for i in range(n_inputs)]].values
            st.session_state.outputs_t = df_t[[f'Output_{i+1}' for i in range(n_outputs)]].values
            st.session_state.inputs_t1 = df_t1[[f'Input_{i+1}' for i in range(n_inputs)]].values
            st.session_state.outputs_t1 = df_t1[[f'Output_{i+1}' for i in range(n_outputs)]].values
            st.session_state.dmu_names = df_t['DMU'].values
            
            # Also set regular inputs/outputs for other models (use period 1)
            st.session_state.inputs = st.session_state.inputs_t
            st.session_state.outputs = st.session_state.outputs_t
        else:
            st.session_state.inputs = df_sample[[f'Input_{i+1}' for i in range(n_inputs)]].values
            st.session_state.outputs = df_sample[[f'Output_{i+1}' for i in range(n_outputs)]].values
            st.session_state.dmu_names = df_sample['DMU'].values
        
        st.success(f"サンプルデータを生成しました: {n_dmus} DMUs, {n_inputs} 入力, {n_outputs} 出力")
        st.dataframe(df_sample, use_container_width=True)
        
        # Download sample data
        csv_sample = df_sample.to_csv(index=False)
        st.download_button(
            label="サンプルデータをCSVでダウンロード",
            data=csv_sample,
            file_name=f"sample_data_{n_dmus}dmus_{n_inputs}inputs_{n_outputs}outputs.csv",
            mime="text/csv"
        )

# Model Analysis Page (Unified)
elif page == "モデル分析":
    st.header("📊 DEAモデル分析")
    
    if st.session_state.inputs is None or st.session_state.outputs is None:
        st.warning("⚠️ まず「データアップロード」ページでデータを設定してください")
    else:
        # All models in one list
        all_models = [
            "CCR", "BCC", "AP (Super-Efficiency)", "Returns to Scale",
            "Cost Efficiency", "Revenue Efficiency", "Malmquist",
            "SBM", "Directional Efficiency", "Bootstrap DEA", "Cross Efficiency"
        ]
        
        model_type = st.selectbox(
            "モデルを選択",
            all_models
        )
        
        # Display model explanation and references
        if model_type in MODEL_INFO:
            with st.expander("📖 モデルの解説と参考文献", expanded=False):
                st.markdown(f"### {MODEL_INFO[model_type]['name']}")
                st.markdown(f"**解説:** {MODEL_INFO[model_type]['explanation']}")
                st.markdown("**参考文献:**")
                for ref in MODEL_INFO[model_type]['references']:
                    st.markdown(f"- {ref}")
        
        # Initialize variables for all models
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
        
        # Model-specific parameters
        if model_type in ["CCR", "BCC"]:
            orientation = st.selectbox("方向", ["入力指向", "出力指向"], index=0)
            method = st.selectbox("方法", ["包絡モデル", "乗数モデル"], index=0)
            
            # 包絡型と乗数型の違いについての説明
            if method == "乗数モデル":
                st.info("""
            **包絡型と乗数型について:**
            
            包絡型と乗数型は**双対問題**の関係にあり、理論的には同じ効率値になります。
            ただし、実装上の理由で微小な差（通常10^-6以下）が生じることがあります：
            
            - **乗数型**: 非Archimedean制約（epsilon制約）を使用して重みが0になることを防ぎます
            - **数値計算の誤差**: 浮動小数点演算による微小な誤差
            - **BCCモデル**: u0変数の表現方法による影響
            
            実用上は、差が10^-6以下であれば同じ結果と見なせます。
            """)
        
        # モデル定式化の表示
        st.subheader("📐 モデル定式化")
        model_formulations = {
            "CCR": r"""
**入力指向包絡モデル:**
$$\min \theta$$
$$\text{s.t. } \sum_{j=1}^{n} \lambda_j x_{ij} \leq \theta x_{ip}, \quad i=1,\ldots,m$$
$$\sum_{j=1}^{n} \lambda_j y_{rj} \geq y_{rp}, \quad r=1,\ldots,s$$
$$\lambda_j \geq 0, \quad j=1,\ldots,n$$

**入力指向乗数モデル:**
$$\max \sum_{r=1}^{s} u_r y_{rp}$$
$$\text{s.t. } \sum_{r=1}^{s} u_r y_{rj} - \sum_{i=1}^{m} v_i x_{ij} \leq 0, \quad j=1,\ldots,n$$
$$\sum_{i=1}^{m} v_i x_{ip} = 1$$
$$u_r \geq \epsilon, \quad v_i \geq \epsilon$$
""",
            "BCC": r"""
**入力指向包絡モデル:**
$$\min \theta$$
$$\text{s.t. } \sum_{j=1}^{n} \lambda_j x_{ij} \leq \theta x_{ip}, \quad i=1,\ldots,m$$
$$\sum_{j=1}^{n} \lambda_j y_{rj} \geq y_{rp}, \quad r=1,\ldots,s$$
$$\sum_{j=1}^{n} \lambda_j = 1$$
$$\lambda_j \geq 0, \quad j=1,\ldots,n$$

**入力指向乗数モデル:**
$$\max \sum_{r=1}^{s} u_r y_{rp} + u_0$$
$$\text{s.t. } \sum_{r=1}^{s} u_r y_{rj} - \sum_{i=1}^{m} v_i x_{ij} + u_0 \leq 0, \quad j=1,\ldots,n$$
$$\sum_{i=1}^{m} v_i x_{ip} = 1$$
$$u_r \geq \epsilon, \quad v_i \geq \epsilon$$
"""
        }
        
        if model_type in model_formulations:
            # Display each line separately for better formatting
            formula_text = model_formulations[model_type]
            # Split by double newlines to preserve paragraph breaks
            paragraphs = formula_text.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    st.markdown(para.strip())
        else:
            st.info(f"{model_type}モデルの定式化は準備中です。")
        
        # Model-specific parameter settings
        if model_type in ["SBM", "Directional Efficiency", "Returns to Scale", "Bootstrap DEA", "Cross Efficiency"]:
            rts = st.selectbox("規模の収穫", ["vrs", "drs", "crs", "irs"], index=0, key="model_rts")
        
        if model_type == "Cost Efficiency":
            st.subheader("入力コストの設定")
            cost_input = st.text_input(
                "入力コスト（カンマ区切り）",
                value=",".join(["1"] * st.session_state.inputs.shape[1])
            )
            try:
                input_costs = np.array([float(x.strip()) for x in cost_input.split(",")])
                if len(input_costs) != st.session_state.inputs.shape[1]:
                    st.error(f"コストの数が入力変数の数と一致しません（{len(input_costs)} vs {st.session_state.inputs.shape[1]}）")
                    input_costs = None
                else:
                    input_costs = np.tile(input_costs, (len(st.session_state.inputs), 1))
            except:
                st.error("コストの形式が正しくありません")
                input_costs = None
        
        if model_type == "Revenue Efficiency":
            st.subheader("出力価格の設定")
            price_input = st.text_input(
                "出力価格（カンマ区切り）",
                value=",".join(["1"] * st.session_state.outputs.shape[1])
            )
            try:
                output_prices = np.array([float(x.strip()) for x in price_input.split(",")])
                if len(output_prices) != st.session_state.outputs.shape[1]:
                    st.error(f"価格の数が出力変数の数と一致しません（{len(output_prices)} vs {st.session_state.outputs.shape[1]}）")
                    output_prices = None
                else:
                    output_prices = np.tile(output_prices, (len(st.session_state.outputs), 1))
            except:
                st.error("価格の形式が正しくありません")
                output_prices = None
        
        if model_type == "Directional Efficiency":
            st.subheader("方向ベクトルの設定")
            g_inputs_str = st.text_input(
                "入力方向ベクトル（カンマ区切り）",
                value=",".join(["1"] * st.session_state.inputs.shape[1])
            )
            g_outputs_str = st.text_input(
                "出力方向ベクトル（カンマ区切り）",
                value=",".join(["1"] * st.session_state.outputs.shape[1])
            )
            try:
                g_inputs = np.array([float(x.strip()) for x in g_inputs_str.split(",")])
                g_outputs = np.array([float(x.strip()) for x in g_outputs_str.split(",")])
            except:
                g_inputs = None
                g_outputs = None
        
        if model_type == "AP (Super-Efficiency)":
            ap_orientation = st.selectbox("方向", ["入力指向", "出力指向"], index=0, key="ap_orient")
        
        if model_type == "SBM":
            sbm_type = st.selectbox("SBMタイプ", ["Model 1", "Model 2"], index=0)
        
        if model_type == "Bootstrap DEA":
            n_bootstrap = st.number_input("ブートストラップ回数", min_value=100, max_value=10000, value=1000, step=100, key="bootstrap_n")
            orientation = st.selectbox("方向", ["入力指向", "出力指向"], index=0, key="bootstrap_orient")
        
        if model_type == "Cross Efficiency":
            orientation = st.selectbox("方向", ["入力指向", "出力指向"], index=0, key="cross_orient")
        
        # モデル定式化の表示
        st.subheader("📐 モデル定式化")
        model_formulations = {
            "CCR": r"""
**入力指向包絡モデル:**
$$\min \theta$$
$$\text{s.t. } \sum_{j=1}^{n} \lambda_j x_{ij} \leq \theta x_{ip}, \quad i=1,\ldots,m$$
$$\sum_{j=1}^{n} \lambda_j y_{rj} \geq y_{rp}, \quad r=1,\ldots,s$$
$$\lambda_j \geq 0, \quad j=1,\ldots,n$$

**入力指向乗数モデル:**
$$\max \sum_{r=1}^{s} u_r y_{rp}$$
$$\text{s.t. } \sum_{r=1}^{s} u_r y_{rj} - \sum_{i=1}^{m} v_i x_{ij} \leq 0, \quad j=1,\ldots,n$$
$$\sum_{i=1}^{m} v_i x_{ip} = 1$$
$$u_r \geq \epsilon, \quad v_i \geq \epsilon$$
""",
            "BCC": r"""
**入力指向包絡モデル:**
$$\min \theta$$
$$\text{s.t. } \sum_{j=1}^{n} \lambda_j x_{ij} \leq \theta x_{ip}, \quad i=1,\ldots,m$$
$$\sum_{j=1}^{n} \lambda_j y_{rj} \geq y_{rp}, \quad r=1,\ldots,s$$
$$\sum_{j=1}^{n} \lambda_j = 1$$
$$\lambda_j \geq 0, \quad j=1,\ldots,n$$

**入力指向乗数モデル:**
$$\max \sum_{r=1}^{s} u_r y_{rp} + u_0$$
$$\text{s.t. } \sum_{r=1}^{s} u_r y_{rj} - \sum_{i=1}^{m} v_i x_{ij} + u_0 \leq 0, \quad j=1,\ldots,n$$
$$\sum_{i=1}^{m} v_i x_{ip} = 1$$
$$u_r \geq \epsilon, \quad v_i \geq \epsilon$$
""",
            "AP (Super-Efficiency)": r"""
**入力指向包絡モデル:**
$$\min \theta$$
$$\text{s.t. } \sum_{j=1}^{n} \lambda_j x_{ij} \leq \theta x_{ip}, \quad i=1,\ldots,m$$
$$\sum_{j=1}^{n} \lambda_j y_{rj} \geq y_{rp}, \quad r=1,\ldots,s$$
$$\lambda_j \geq 0, \quad j=1,\ldots,n, j \neq p$$
""",
            "SBM": r"""
**Model 1 (入力指向):**
$$\rho^* = \min \frac{1 - \frac{1}{m}\sum_{i=1}^{m} \frac{s_i^-}{x_{ip}}}{1 + \frac{1}{s}\sum_{r=1}^{s} \frac{s_r^+}{y_{rp}}}$$
$$\text{s.t. } \sum_{j=1}^{n} \lambda_j x_{ij} + s_i^- = x_{ip}, \quad i=1,\ldots,m$$
$$\sum_{j=1}^{n} \lambda_j y_{rj} - s_r^+ = y_{rp}, \quad r=1,\ldots,s$$
$$\sum_{j=1}^{n} \lambda_j = 1 \text{ (VRS)}$$
$$\lambda_j \geq 0, s_i^- \geq 0, s_r^+ \geq 0$$
""",
            "Directional Efficiency": r"""
$$\max \beta$$
$$\text{s.t. } \sum_{j=1}^{n} \lambda_j x_{ij} \leq x_{ip} - \beta g_{xi}, \quad i=1,\ldots,m$$
$$\sum_{j=1}^{n} \lambda_j y_{rj} \geq y_{rp} + \beta g_{yr}, \quad r=1,\ldots,s$$
$$\sum_{j=1}^{n} \lambda_j = 1 \text{ (VRS)}$$
$$\lambda_j \geq 0, \beta \geq 0$$
""",
            "Cost Efficiency": r"""
$$\min \sum_{i=1}^{m} c_i x_i^*$$
$$\text{s.t. } \sum_{j=1}^{n} \lambda_j x_{ij} \leq x_i^*, \quad i=1,\ldots,m$$
$$\sum_{j=1}^{n} \lambda_j y_{rj} \geq y_{rp}, \quad r=1,\ldots,s$$
$$\sum_{j=1}^{n} \lambda_j = 1 \text{ (VRS)}$$
$$\lambda_j \geq 0, x_i^* \geq 0$$
""",
            "Revenue Efficiency": r"""
$$\max \sum_{r=1}^{s} p_r y_r^*$$
$$\text{s.t. } \sum_{j=1}^{n} \lambda_j x_{ij} \leq x_{ip}, \quad i=1,\ldots,m$$
$$\sum_{j=1}^{n} \lambda_j y_{rj} \geq y_r^*, \quad r=1,\ldots,s$$
$$\sum_{j=1}^{n} \lambda_j = 1 \text{ (VRS)}$$
$$\lambda_j \geq 0, y_r^* \geq 0$$
""",
            "Malmquist": r"""
**Malmquist Productivity Index:**
$$M_{t,t+1} = \left[ \frac{D^t(x^{t+1}, y^{t+1})}{D^t(x^t, y^t)} \cdot \frac{D^{t+1}(x^{t+1}, y^{t+1})}{D^{t+1}(x^t, y^t)} \right]^{1/2}$$

技術効率変化 (EFFCH):
$$EFFCH = \frac{D^{t+1}(x^{t+1}, y^{t+1})}{D^t(x^t, y^t)}$$

技術変化 (TECHCH):
$$TECHCH = \left[ \frac{D^t(x^{t+1}, y^{t+1})}{D^{t+1}(x^{t+1}, y^{t+1})} \cdot \frac{D^t(x^t, y^t)}{D^{t+1}(x^t, y^t)} \right]^{1/2}$$
""",
            "Bootstrap DEA": r"""
**Bootstrap DEA モデル:**
1. 元のDEA効率性 $\theta_j^*$ を計算
2. $B$ 回のブートストラップサンプルを生成
3. 各サンプル $b$ について効率性 $\theta_j^{*(b)}$ を計算
4. 信頼区間を計算:

$$CI_{1-\alpha} = [\theta_j^{*(lower)}, \theta_j^{*(upper)}]$$

ここで、$\theta_j^{*(lower)}$ と $\theta_j^{*(upper)}$ は $\alpha/2$ と $1-\alpha/2$ 分位数
""",
            "Cross Efficiency": r"""
**Cross-Efficiency モデル:**
各DMU $d$ について、他のすべてのDMU $k$ の最適重み $(u_k^*, v_k^*)$ を使用:

$$E_{dk} = \frac{\sum_{r=1}^{s} u_{rk}^* y_{rd}}{\sum_{i=1}^{m} v_{ik}^* x_{id}}$$

平均Cross-Efficiency:
$$\bar{E}_d = \frac{1}{n} \sum_{k=1}^{n} E_{dk}$$
"""
        }
        
        if model_type in model_formulations:
            # Display each line separately for better formatting
            formula_text = model_formulations[model_type]
            # Split by double newlines to preserve paragraph breaks
            paragraphs = formula_text.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    st.markdown(para.strip())
        else:
            st.info(f"{model_type}モデルの定式化は準備中です。")
        
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
                                    eff, lambdas, input_slacks, output_slacks = model.solve_output_oriented_envelopment(i)
                                    results_list.append({
                                        'DMU': i+1,
                                        'Efficiency': eff,
                                        **{f'Lambda_{j+1}': lambdas[j] for j in range(len(lambdas))}
                                    })
                                results = pd.DataFrame(results_list)
                        else:
                            results = model.evaluate_all(method='multiplier')
                    
                    elif model_type == "BCC":
                        model = BCCModel(st.session_state.inputs, st.session_state.outputs)
                        if method == "包絡モデル":
                            results = model.evaluate_all(method='envelopment')
                        else:
                            results = model.evaluate_all(method='multiplier')
                    
                    elif model_type == "AP (Super-Efficiency)":
                        model = APModel(st.session_state.inputs, st.session_state.outputs)
                        if ap_orientation == "入力指向":
                            results = model.evaluate_all(orientation='input', method='envelopment')
                        else:
                            results = model.evaluate_all(orientation='output', method='envelopment')
                    
                    elif model_type == "SBM":
                        model = SBMModel(st.session_state.inputs, st.session_state.outputs)
                        results_list = []
                        for i in range(len(st.session_state.inputs)):
                            if sbm_type == "Model 1":
                                eff, lambdas, input_slacks, output_slacks = model.solve_model1(i, rts=rts)
                            else:
                                eff, lambdas, input_slacks, output_slacks = model.solve_model2(i, rts=rts)
                            results_list.append({
                                'DMU': i+1,
                                'SBM_Efficiency': eff,
                                **{f'Lambda_{j+1}': lambdas[j] for j in range(len(lambdas))}
                            })
                        results = pd.DataFrame(results_list)
                    
                    elif model_type == "Cost Efficiency":
                        if input_costs is not None:
                            model = CostEfficiencyModel(st.session_state.inputs, st.session_state.outputs, input_costs)
                            results = model.evaluate_all()
                        else:
                            st.error("入力コストを正しく設定してください")
                            results = None
                    
                    elif model_type == "Revenue Efficiency":
                        if output_prices is not None:
                            model = RevenueEfficiencyModel(st.session_state.inputs, st.session_state.outputs, output_prices)
                            results = model.evaluate_all()
                        else:
                            st.error("出力価格を正しく設定してください")
                            results = None
                    
                    elif model_type == "Directional Efficiency":
                        if g_inputs is not None and g_outputs is not None:
                            model = DirectionalEfficiencyModel(st.session_state.inputs, st.session_state.outputs)
                            results_list = []
                            for i in range(len(st.session_state.inputs)):
                                eff, lambdas, input_slacks, output_slacks = model.solve(i, g_inputs, g_outputs, rts=rts)
                                results_list.append({
                                    'DMU': i+1,
                                    'Directional_Efficiency': eff,
                                    **{f'Lambda_{j+1}': lambdas[j] for j in range(len(lambdas))}
                                })
                            results = pd.DataFrame(results_list)
                        else:
                            st.error("方向ベクトルを正しく設定してください")
                            results = None
                    
                    elif model_type == "Returns to Scale":
                        model = ReturnsToScaleModel(st.session_state.inputs, st.session_state.outputs)
                        results = model.evaluate_all()
                    
                    elif model_type == "Malmquist":
                        if hasattr(st.session_state, 'inputs_t') and hasattr(st.session_state, 'inputs_t1'):
                            model = MalmquistModel(
                                st.session_state.inputs_t, st.session_state.outputs_t,
                                st.session_state.inputs_t1, st.session_state.outputs_t1
                            )
                            results = model.evaluate_all()
                        else:
                            st.error("Malmquistモデルには時系列データが必要です。「データアップロード」ページで「時系列データ（Malmquist用）」テンプレートを使用してデータを生成してください。")
                            results = None
                    
                    elif model_type == "Bootstrap DEA":
                        model = BootstrapDEAModel(st.session_state.inputs, st.session_state.outputs, rts=rts, orientation='in' if orientation == "入力指向" else 'out')
                        results = model.evaluate_all(n_rep=n_bootstrap)
                    
                    elif model_type == "Cross Efficiency":
                        model = CrossEfficiencyModel(st.session_state.inputs, st.session_state.outputs)
                        results = model.evaluate_all(
                            orientation='io' if orientation == "入力指向" else 'oo',
                            rts=rts
                        )
                    
                    if results is not None:
                        st.session_state.results = results
                        st.success("分析が完了しました！")
            
            except Exception as e:
                st.error(f"エラー: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        if st.session_state.results is not None:
            st.subheader("結果")
            st.dataframe(st.session_state.results, use_container_width=True)
            
            csv = st.session_state.results.to_csv(index=False)
            st.download_button(
                label="結果をCSVでダウンロード",
                data=csv,
                file_name=f"{model_type.replace(' ', '_')}_results.csv",
                mime="text/csv"
            )

# Visualization Page
elif page == "結果の可視化":
    st.header("📈 結果の可視化")
    
    if st.session_state.results is None:
        st.warning("⚠️ まず他のページで分析を実行してください")
    else:
        results = st.session_state.results
        
        # Check if results is a DataFrame
        if not isinstance(results, pd.DataFrame):
            st.error("結果がDataFrame形式ではありません。分析を再実行してください。")
else:
            # Efficiency score visualization
            eff_cols = [col for col in results.columns if 'Efficiency' in col or 'efficiency' in col.lower()]
            eff_col = None
            if eff_cols:
                eff_col = eff_cols[0]
                
                st.subheader("効率スコアの分布")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart
                    fig_bar = px.bar(
                        results,
                        x='DMU',
                        y=eff_col,
                        title="効率スコア（バーチャート）",
                        labels={eff_col: '効率スコア', 'DMU': 'DMU'}
                    )
                    fig_bar.update_layout(height=400)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    # Histogram
                    fig_hist = px.histogram(
                        results,
                        x=eff_col,
                        title="効率スコアの分布",
                        labels={eff_col: '効率スコア', 'count': '頻度'},
                        nbins=20
                    )
                    fig_hist.update_layout(height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Summary statistics
                st.subheader("統計サマリー")
                summary_stats = results[eff_col].describe()
                st.dataframe(summary_stats)
            
            # If we have input/output data, show scatter plots
            if st.session_state.inputs is not None and st.session_state.outputs is not None:
                st.subheader("入力・出力の関係")
                
                if st.session_state.inputs.shape[1] >= 1 and st.session_state.outputs.shape[1] >= 1:
                    # Create a combined dataframe
                    plot_df = pd.DataFrame({
                        'Input1': st.session_state.inputs[:, 0],
                        'Output1': st.session_state.outputs[:, 0],
                        'DMU': range(1, len(st.session_state.inputs) + 1)
                    })
                    
                    if st.session_state.inputs.shape[1] > 1:
                        plot_df['Input2'] = st.session_state.inputs[:, 1]
                    
                    if eff_cols and len(eff_cols) > 0 and eff_col in results.columns:
                        plot_df['Efficiency'] = results[eff_col].values
                    
                    # Scatter plot
                    if 'Efficiency' in plot_df.columns:
                        fig_scatter = px.scatter(
                            plot_df,
                            x='Input1',
                            y='Output1',
                            size='Efficiency',
                            color='Efficiency',
                            hover_data=['DMU'],
                            title="入力と出力の関係（効率スコアで色分け）",
                            labels={'Input1': '入力1', 'Output1': '出力1', 'Efficiency': '効率'}
                        )
                    else:
                        fig_scatter = px.scatter(
                            plot_df,
                            x='Input1',
                            y='Output1',
                            hover_data=['DMU'],
                            title="入力と出力の関係",
                            labels={'Input1': '入力1', 'Output1': '出力1'}
                        )
                    st.plotly_chart(fig_scatter, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 情報")
st.sidebar.info("""
このアプリはDEAモデルを簡単に使用できるようにするためのツールです。

**対応モデル:**
- CCR, BCC, AP (Super-Efficiency), Returns to Scale
- Cost Efficiency, Revenue Efficiency, Malmquist
- SBM, Directional Efficiency, Bootstrap DEA, Cross Efficiency
""")
