import pandas as pd
import plotly.express as px
import streamlit as st
from src.logger import configure_logger

logger = configure_logger(__name__)


def main():
    logger.info("now loading...")
    logger.info("start fun time")
    st.header("予測結果")

    # CSVファイルを選択する
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

    if uploaded_file is not None:
        # 解析ボタンを表示する
        analyze_button = st.button("解析")

        if analyze_button:
            # CSVファイルをデータフレームに読み込む
            df = pd.read_csv(uploaded_file)

            # ヒストグラムを描画する
            fig = px.histogram(df, x="prediction")

            # ヒストグラムを表示する
            st.plotly_chart(fig)


if __name__ == "__main__":
    main()
