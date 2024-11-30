import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# タイトル
st.title("回帰分析アプリ")

# データアップロード
st.sidebar.header("データをアップロードしてください")
uploaded_file = st.sidebar.file_uploader("CSVファイルを選択", type=["csv"])

if uploaded_file is not None:
    # データ読み込み
    data = pd.read_csv(uploaded_file)
    st.write("データプレビュー:")
    st.write(data.head())

    # ユーザーがターゲット変数を選択
    target_column = st.sidebar.selectbox("ターゲット変数を選択してください", data.columns)
    feature_columns = st.sidebar.multiselect("説明変数を選択してください", [col for col in data.columns if col != target_column])

    if feature_columns and target_column:
        # データの分割
        X = data[feature_columns]
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 回帰モデルの構築
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 予測と評価
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("## モデル評価")
        st.write(f"平均二乗誤差 (MSE): {mse:.2f}")
        st.write(f"決定係数 (R²): {r2:.2f}")

        # 回帰係数
        st.write("## 回帰係数")
        coefficients = pd.DataFrame({
            "説明変数": feature_columns,
            "回帰係数": model.coef_
        })
        st.write(coefficients)

        # 可視化
        st.write("## 可視化")
        if len(feature_columns) == 1:
            # 単変量回帰の場合、散布図と回帰直線をプロット
            plt.figure(figsize=(8, 6))
            plt.scatter(X_test, y_test, label="実データ", color="blue")
            plt.plot(X_test, y_pred, label="回帰直線", color="red")
            plt.xlabel(feature_columns[0])
            plt.ylabel(target_column)
            plt.title("回帰直線")
            plt.legend()
            st.pyplot(plt)
        else:
            st.write("複数の説明変数が選択されているため、可視化は省略されます。")

else:
    st.sidebar.write("CSVファイルをアップロードしてください。")

st.write("### 注意事項")
st.write("""
- データのアップロードはCSV形式で行ってください。
- 説明変数（X）とターゲット変数（y）を正しく選択してください。
- 複数の説明変数を選択した場合、可視化は対応していません。
""")
