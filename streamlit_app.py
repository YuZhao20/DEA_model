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
    CCRModel, BCCModel, APModel, MAJModel,
    AdditiveModel, TwoPhaseModel,
    NormL1Model, CongestionModel, CommonWeightsModel, DirectionalEfficiencyModel,
    ReturnsToScaleModel,
    CostEfficiencyModel, RevenueEfficiencyModel,
    MalmquistModel,
    SBMModel,
    ProfitEfficiencyModel, ModifiedSBMModel,
    SeriesNetworkModel,
    DRSModel, IRSModel,
    FDHModel, FDHPlusModel,
    MEAModel,
    EfficiencyLadderModel,
    MergerAnalysisModel,
    BootstrapDEAModel,
    NonRadialModel, LGOModel, RDMModel,
    AddMinModel, AddSuperEffModel, DEAPSModel,
    CrossEfficiencyModel,
    transform_undesirable,
    StoNEDModel
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

# Sidebar for navigation
st.sidebar.title("ナビゲーション")
page = st.sidebar.selectbox(
    "ページを選択",
    ["データアップロード", "基本モデル", "高度なモデル", "追加モデル", "特殊モデル", "結果の可視化"]
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

    # Sample data generator
    st.subheader("サンプルデータの生成")
    if st.button("サンプルデータを生成"):
        np.random.seed(42)
        n_dmus = 10
        n_inputs = 2
        n_outputs = 2
        
        sample_data = {
            'DMU': [f'DMU_{i+1}' for i in range(n_dmus)],
        }
        for i in range(n_inputs):
            sample_data[f'Input_{i+1}'] = np.random.uniform(1, 10, n_dmus)
        for i in range(n_outputs):
            sample_data[f'Output_{i+1}'] = np.random.uniform(1, 10, n_dmus)
        
        df_sample = pd.DataFrame(sample_data)
        st.session_state.data = df_sample
        st.session_state.inputs = df_sample[[f'Input_{i+1}' for i in range(n_inputs)]].values
        st.session_state.outputs = df_sample[[f'Output_{i+1}' for i in range(n_outputs)]].values
        st.session_state.dmu_names = df_sample['DMU'].values
        
        st.success("サンプルデータを生成しました")
        st.dataframe(df_sample)

# Basic Models Page
elif page == "基本モデル":
    st.header("🔷 基本DEAモデル")
    
    if st.session_state.inputs is None or st.session_state.outputs is None:
        st.warning("⚠️ まず「データアップロード」ページでデータを設定してください")
    else:
        model_type = st.selectbox(
            "モデルを選択",
            ["CCR", "BCC", "Additive", "Two-Phase"]
        )
        
        orientation = st.selectbox("方向", ["入力指向", "出力指向"], index=0)
        method = st.selectbox("方法", ["包絡モデル", "乗数モデル"], index=0)
        
        # Additive model type selection
        model_type_add = "CCR"
        if model_type == "Additive":
            model_type_add = st.selectbox("Additiveタイプ", ["CCR", "BCC"], index=0)
        
        if st.button("分析を実行", type="primary"):
            try:
                with st.spinner("計算中..."):
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
                    
                    elif model_type == "Additive":
                        model = AdditiveModel(st.session_state.inputs, st.session_state.outputs)
                        results_list = []
                        for i in range(len(st.session_state.inputs)):
                            if model_type_add == "CCR":
                                slack, lambdas, input_slacks, output_slacks = model.solve_ccr(i)
                            else:
                                slack, lambdas, input_slacks, output_slacks = model.solve_bcc(i)
                            results_list.append({
                                'DMU': i+1,
                                'Total_Slack': slack,
                                **{f'Lambda_{j+1}': lambdas[j] for j in range(len(lambdas))}
                            })
                        results = pd.DataFrame(results_list)
                    
                    elif model_type == "Two-Phase":
                        model = TwoPhaseModel(st.session_state.inputs, st.session_state.outputs)
                        results_list = []
                        for i in range(len(st.session_state.inputs)):
                            eff, lambdas, input_slacks, output_slacks, total_slack = model.solve(i)
                            results_list.append({
                                'DMU': i+1,
                                'Efficiency': eff,
                                'Total_Slack': total_slack,
                                **{f'Lambda_{j+1}': lambdas[j] for j in range(len(lambdas))}
                            })
                        results = pd.DataFrame(results_list)
                    
                    st.session_state.results = results
                    st.success("分析が完了しました！")
            
            except Exception as e:
                st.error(f"エラー: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        if st.session_state.results is not None:
            st.subheader("結果")
            st.dataframe(st.session_state.results, use_container_width=True)
            
            # Download button
            csv = st.session_state.results.to_csv(index=False)
            st.download_button(
                label="結果をCSVでダウンロード",
                data=csv,
                file_name=f"{model_type}_results.csv",
                mime="text/csv"
            )

# Advanced Models Page
elif page == "高度なモデル":
    st.header("🔶 高度なDEAモデル")
    
    if st.session_state.inputs is None or st.session_state.outputs is None:
        st.warning("⚠️ まず「データアップロード」ページでデータを設定してください")
    else:
        model_type = st.selectbox(
            "モデルを選択",
            ["AP (Super-Efficiency)", "MAJ (Super-Efficiency)", "SBM", "Cost Efficiency", 
             "Revenue Efficiency", "Directional Efficiency", "Norm L1", "Congestion",
             "Common Weights", "Returns to Scale"]
        )
        
        # Initialize variables
        input_costs = None
        output_prices = None
        g_inputs = None
        g_outputs = None
        sbm_type = "Model 1"
        ap_orientation = "入力指向"
        
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
        
        if st.button("分析を実行", type="primary"):
            try:
                with st.spinner("計算中..."):
                    if model_type == "AP (Super-Efficiency)":
                        model = APModel(st.session_state.inputs, st.session_state.outputs)
                        if ap_orientation == "入力指向":
                            results = model.evaluate_all(orientation='input', method='envelopment')
                        else:
                            results = model.evaluate_all(orientation='output', method='envelopment')
                    
                    elif model_type == "MAJ (Super-Efficiency)":
                        model = MAJModel(st.session_state.inputs, st.session_state.outputs)
                        results = model.evaluate_all()
                    
                    elif model_type == "SBM":
                        model = SBMModel(st.session_state.inputs, st.session_state.outputs)
                        results_list = []
                        for i in range(len(st.session_state.inputs)):
                            if sbm_type == "Model 1":
                                eff, lambdas, input_slacks, output_slacks = model.solve_model1(i)
                            else:
                                eff, lambdas, input_slacks, output_slacks = model.solve_model2(i)
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
                                eff, lambdas, input_slacks, output_slacks = model.solve(i, g_inputs, g_outputs)
                                results_list.append({
                                    'DMU': i+1,
                                    'Directional_Efficiency': eff,
                                    **{f'Lambda_{j+1}': lambdas[j] for j in range(len(lambdas))}
                                })
                            results = pd.DataFrame(results_list)
                        else:
                            st.error("方向ベクトルを正しく設定してください")
                            results = None
                    
                    elif model_type == "Norm L1":
                        model = NormL1Model(st.session_state.inputs, st.session_state.outputs)
                        results = model.evaluate_all()
                    
                    elif model_type == "Congestion":
                        model = CongestionModel(st.session_state.inputs, st.session_state.outputs)
                        results_list = []
                        for i in range(len(st.session_state.inputs)):
                            congestion, lambdas, input_slacks, output_slacks = model.solve(i)
                            results_list.append({
                                'DMU': i+1,
                                'Congestion': congestion,
                                **{f'Lambda_{j+1}': lambdas[j] for j in range(len(lambdas))}
                            })
                        results = pd.DataFrame(results_list)
                    
                    elif model_type == "Common Weights":
                        model = CommonWeightsModel(st.session_state.inputs, st.session_state.outputs)
                        results = model.evaluate_all()
                    
                    elif model_type == "Returns to Scale":
                        model = ReturnsToScaleModel(st.session_state.inputs, st.session_state.outputs)
                        results = model.evaluate_all()
                    
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

# Additional Models Page
elif page == "追加モデル":
    st.header("🔸 追加DEAモデル")
    
    if st.session_state.inputs is None or st.session_state.outputs is None:
        st.warning("⚠️ まず「データアップロード」ページでデータを設定してください")
    else:
        model_type = st.selectbox(
            "モデルを選択",
            ["DRS", "IRS", "FDH", "FDH+", "MEA", "Cross Efficiency", "Non-Radial", "LGO", "RDM"]
        )
        
        rts = st.selectbox("規模の収穫", ["vrs", "drs", "crs", "irs"], index=0)
        orientation = st.selectbox("方向", ["入力指向", "出力指向"], index=0)
        
        if st.button("分析を実行", type="primary"):
            try:
                with st.spinner("計算中..."):
                    if model_type == "DRS":
                        model = DRSModel(st.session_state.inputs, st.session_state.outputs)
                        if orientation == "入力指向":
                            results = model.evaluate_all(orientation='input')
                        else:
                            results = model.evaluate_all(orientation='output')
                    
                    elif model_type == "IRS":
                        model = IRSModel(st.session_state.inputs, st.session_state.outputs)
                        if orientation == "入力指向":
                            results = model.evaluate_all(orientation='input')
                        else:
                            results = model.evaluate_all(orientation='output')
                    
                    elif model_type == "FDH":
                        model = FDHModel(st.session_state.inputs, st.session_state.outputs)
                        if orientation == "入力指向":
                            results = model.evaluate_all(orientation='input')
                        else:
                            results = model.evaluate_all(orientation='output')
                    
                    elif model_type == "FDH+":
                        model = FDHPlusModel(st.session_state.inputs, st.session_state.outputs)
                        if orientation == "入力指向":
                            results = model.evaluate_all(orientation='input')
                        else:
                            results = model.evaluate_all(orientation='output')
                    
                    elif model_type == "MEA":
                        model = MEAModel(st.session_state.inputs, st.session_state.outputs)
                        results_list = []
                        for i in range(len(st.session_state.inputs)):
                            eff, lambdas, input_slacks, output_slacks, directions = model.solve(
                                i, orientation='input' if orientation == "入力指向" else 'output', rts=rts
                            )
                            results_list.append({
                                'DMU': i+1,
                                'MEA_Efficiency': eff,
                                **{f'Lambda_{j+1}': lambdas[j] for j in range(len(lambdas))}
                            })
                        results = pd.DataFrame(results_list)
                    
                    elif model_type == "Cross Efficiency":
                        model = CrossEfficiencyModel(st.session_state.inputs, st.session_state.outputs)
                        cross_results = model.solve(
                            orientation='io' if orientation == "入力指向" else 'oo',
                            rts=rts
                        )
                        results = pd.DataFrame({
                            'DMU': range(1, len(st.session_state.inputs) + 1),
                            'Cross_Efficiency': cross_results['average_scores']
                        })
                    
                    elif model_type == "Non-Radial":
                        model = NonRadialModel(st.session_state.inputs, st.session_state.outputs)
                        results_list = []
                        for i in range(len(st.session_state.inputs)):
                            eff, lambdas, input_slacks, output_slacks = model.solve(
                                i, orientation='input' if orientation == "入力指向" else 'output', rts=rts
                            )
                            results_list.append({
                                'DMU': i+1,
                                'NonRadial_Efficiency': eff,
                                **{f'Lambda_{j+1}': lambdas[j] for j in range(len(lambdas))}
                            })
                        results = pd.DataFrame(results_list)
                    
                    elif model_type == "LGO":
                        model = LGOModel(st.session_state.inputs, st.session_state.outputs)
                        results_list = []
                        for i in range(len(st.session_state.inputs)):
                            eff, lambdas, input_slacks, output_slacks = model.solve(
                                i, orientation='input' if orientation == "入力指向" else 'output', rts=rts
                            )
                            results_list.append({
                                'DMU': i+1,
                                'LGO_Efficiency': eff,
                                **{f'Lambda_{j+1}': lambdas[j] for j in range(len(lambdas))}
                            })
                        results = pd.DataFrame(results_list)
                    
                    elif model_type == "RDM":
                        model = RDMModel(st.session_state.inputs, st.session_state.outputs)
                        results_list = []
                        for i in range(len(st.session_state.inputs)):
                            eff, lambdas, input_slacks, output_slacks = model.solve(
                                i, orientation='input' if orientation == "入力指向" else 'output', rts=rts
                            )
                            results_list.append({
                                'DMU': i+1,
                                'RDM_Efficiency': eff,
                                **{f'Lambda_{j+1}': lambdas[j] for j in range(len(lambdas))}
                            })
                        results = pd.DataFrame(results_list)
                    
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
                file_name=f"{model_type}_results.csv",
                mime="text/csv"
            )

# Special Models Page
elif page == "特殊モデル":
    st.header("🔹 特殊DEAモデル")
    
    if st.session_state.inputs is None or st.session_state.outputs is None:
        st.warning("⚠️ まず「データアップロード」ページでデータを設定してください")
    else:
        model_type = st.selectbox(
            "モデルを選択",
            ["Profit Efficiency", "Modified SBM", "Series Network", "Malmquist",
             "Efficiency Ladder", "Merger Analysis", "Bootstrap DEA",
             "Add Min", "Add Super-Eff", "DEA-PS", "StoNED"]
        )
        
        rts = st.selectbox("規模の収穫", ["vrs", "drs", "crs", "irs"], index=0)
        orientation = st.selectbox("方向", ["入力指向", "出力指向"], index=0)
        
        # Special parameters
        input_prices = None
        output_prices = None
        network_stages = 2
        
        if model_type == "Profit Efficiency":
            st.subheader("価格の設定")
            input_price_str = st.text_input(
                "入力価格（カンマ区切り）",
                value=",".join(["1"] * st.session_state.inputs.shape[1])
            )
            output_price_str = st.text_input(
                "出力価格（カンマ区切り）",
                value=",".join(["1"] * st.session_state.outputs.shape[1])
            )
            try:
                input_prices = np.array([float(x.strip()) for x in input_price_str.split(",")])
                output_prices = np.array([float(x.strip()) for x in output_price_str.split(",")])
                if len(input_prices) != st.session_state.inputs.shape[1] or len(output_prices) != st.session_state.outputs.shape[1]:
                    st.error("価格の数が変数の数と一致しません")
                    input_prices = None
                    output_prices = None
            except:
                st.error("価格の形式が正しくありません")
                input_prices = None
                output_prices = None
        
        if model_type == "Series Network":
            network_stages = st.number_input("ネットワーク段階数", min_value=2, max_value=10, value=2, step=1)
        
        if st.button("分析を実行", type="primary"):
            try:
                with st.spinner("計算中..."):
                    if model_type == "Profit Efficiency":
                        if input_prices is not None and output_prices is not None:
                            model = ProfitEfficiencyModel(
                                st.session_state.inputs, st.session_state.outputs,
                                input_prices, output_prices
                            )
                            results = model.evaluate_all()
                        else:
                            st.error("価格を正しく設定してください")
                            results = None
                    
                    elif model_type == "Modified SBM":
                        model = ModifiedSBMModel(st.session_state.inputs, st.session_state.outputs)
                        results_list = []
                        for i in range(len(st.session_state.inputs)):
                            eff, lambdas, input_slacks, output_slacks = model.solve(i)
                            results_list.append({
                                'DMU': i+1,
                                'Modified_SBM_Efficiency': eff,
                                **{f'Lambda_{j+1}': lambdas[j] for j in range(len(lambdas))}
                            })
                        results = pd.DataFrame(results_list)
                    
                    elif model_type == "Series Network":
                        model = SeriesNetworkModel(st.session_state.inputs, st.session_state.outputs)
                        results_list = []
                        for i in range(len(st.session_state.inputs)):
                            eff, lambdas = model.solve(i, n_stages=network_stages)
                            results_list.append({
                                'DMU': i+1,
                                'Network_Efficiency': eff,
                                **{f'Lambda_{j+1}': lambdas[j] for j in range(len(lambdas))}
                            })
                        results = pd.DataFrame(results_list)
                    
                    elif model_type == "Malmquist":
                        st.warning("Malmquistモデルには時系列データが必要です。現在の実装では基本的な分析のみサポートします。")
                        model = MalmquistModel(st.session_state.inputs, st.session_state.outputs)
                        results = model.evaluate_all()
                    
                    elif model_type == "Efficiency Ladder":
                        model = EfficiencyLadderModel(st.session_state.inputs, st.session_state.outputs)
                        results = model.evaluate_all()
                    
                    elif model_type == "Merger Analysis":
                        st.info("マージ分析には複数のDMUグループが必要です。デフォルトでは全DMUを1グループとして分析します。")
                        model = MergerAnalysisModel(st.session_state.inputs, st.session_state.outputs)
                        # Simple analysis with all DMUs as one group
                        groups = [[i for i in range(len(st.session_state.inputs))]]
                        results = model.analyze_merger(groups)
                    
                    elif model_type == "Bootstrap DEA":
                        n_bootstrap = st.number_input("ブートストラップ回数", min_value=100, max_value=10000, value=1000, step=100)
                        model = BootstrapDEAModel(st.session_state.inputs, st.session_state.outputs)
                        results = model.bootstrap(n_replications=n_bootstrap, rts=rts)
                    
                    elif model_type == "Add Min":
                        model = AddMinModel(st.session_state.inputs, st.session_state.outputs)
                        results_list = []
                        for i in range(len(st.session_state.inputs)):
                            eff, lambdas, input_slacks, output_slacks = model.solve(i, rts=rts)
                            results_list.append({
                                'DMU': i+1,
                                'AddMin_Efficiency': eff,
                                **{f'Lambda_{j+1}': lambdas[j] for j in range(len(lambdas))}
                            })
                        results = pd.DataFrame(results_list)
                    
                    elif model_type == "Add Super-Eff":
                        model = AddSuperEffModel(st.session_state.inputs, st.session_state.outputs)
                        results_list = []
                        for i in range(len(st.session_state.inputs)):
                            eff, lambdas, input_slacks, output_slacks = model.solve(i, rts=rts)
                            results_list.append({
                                'DMU': i+1,
                                'AddSuperEff_Efficiency': eff,
                                **{f'Lambda_{j+1}': lambdas[j] for j in range(len(lambdas))}
                            })
                        results = pd.DataFrame(results_list)
                    
                    elif model_type == "DEA-PS":
                        model = DEAPSModel(st.session_state.inputs, st.session_state.outputs)
                        results_list = []
                        for i in range(len(st.session_state.inputs)):
                            eff, lambdas, input_slacks, output_slacks, u, v = model.solve(
                                i, orientation='io' if orientation == "入力指向" else 'oo', rts=rts
                            )
                            results_list.append({
                                'DMU': i+1,
                                'DEAPS_Efficiency': eff,
                                **{f'Lambda_{j+1}': lambdas[j] for j in range(len(lambdas))}
                            })
                        results = pd.DataFrame(results_list)
                    
                    elif model_type == "StoNED":
                        st.info("StoNEDモデルは単一出力のみをサポートします。")
                        if st.session_state.outputs.shape[1] == 1:
                            model = StoNEDModel(st.session_state.inputs, st.session_state.outputs.flatten())
                            stoned_results = model.solve(rts=rts, method='MM')
                            results = pd.DataFrame({
                                'DMU': range(1, len(st.session_state.inputs) + 1),
                                'Efficiency': stoned_results.get('efficiency', np.ones(len(st.session_state.inputs))),
                                'Inefficiency': stoned_results.get('inefficiency', np.zeros(len(st.session_state.inputs)))
                            })
        else:
                            st.error("StoNEDモデルは単一出力のみをサポートします")
                            results = None
                    
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
        
        # Efficiency score visualization
        eff_cols = [col for col in results.columns if 'Efficiency' in col or 'efficiency' in col.lower()]
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
                
                if eff_cols and eff_col in results.columns:
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
- 基本モデル: CCR, BCC, Additive, Two-Phase
- 高度なモデル: AP, MAJ, SBM, Cost/Revenue Efficiency, Norm L1, Congestion, Common Weights
- 追加モデル: DRS, IRS, FDH, FDH+, MEA, Cross Efficiency, Non-Radial, LGO, RDM
- 特殊モデル: Profit Efficiency, Modified SBM, Series Network, Malmquist, Efficiency Ladder, Merger Analysis, Bootstrap DEA, Add Min, Add Super-Eff, DEA-PS, StoNED
""")
