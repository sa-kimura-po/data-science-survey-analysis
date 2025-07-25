import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from survey_analyzer import SurveyAnalyzer
from bert_analyzer import BertSurveyAnalyzer
import tempfile
import os

# ページ設定
st.set_page_config(
    page_title="学生アンケート理解度分析システム",
    page_icon="📊",
    layout="wide"
)

# タイトル
st.title("📊 学生アンケート理解度分析システム")
st.markdown("学生のアンケート回答を分析して理解度を可視化します。")

# サイドバー
st.sidebar.header("📁 ファイルアップロード")

# ファイルアップローダー
uploaded_file = st.sidebar.file_uploader(
    "CSVファイルをアップロードしてください",
    type=['csv'],
    help="学生ID、氏名、回答の列が含まれるCSVファイルをアップロードしてください"
)

# サンプルデータを使用するオプション
use_sample = st.sidebar.checkbox("サンプルデータを使用", value=True)

# 分析方法の選択
analysis_method = st.sidebar.selectbox(
    "分析方法を選択",
    ["ルールベース分析", "BERT分析（高精度）"],
    help="BERT分析は初回実行時にモデルダウンロードが必要です"
)

# 分析実行
if uploaded_file is not None or use_sample:
    
    # 分析方法に応じてアナライザーを初期化
    if analysis_method == "BERT分析（高精度）":
        analyzer = BertSurveyAnalyzer()
        st.info("🤖 BERT分析を使用します（初回実行時はモデルダウンロードに時間がかかります）")
    else:
        analyzer = SurveyAnalyzer()
        st.info("📊 ルールベース分析を使用します")
    
    if use_sample:
        # サンプルデータを使用
        if analyzer.load_csv('survey_data.csv'):
            st.success("✅ サンプルデータを読み込みました")
        else:
            st.error("❌ サンプルデータの読み込みに失敗しました")
            st.stop()
    else:
        # アップロードされたファイルを使用
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        if analyzer.load_csv(tmp_file_path):
            st.success(f"✅ ファイルを読み込みました（{len(analyzer.df)}行）")
        else:
            st.error("❌ ファイルの読み込みに失敗しました")
            st.stop()
    
    # データ表示
    st.header("📋 データ確認")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("データプレビュー")
        st.dataframe(analyzer.df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("データ概要")
        st.metric("総レコード数", len(analyzer.df))
        st.metric("カラム数", len(analyzer.df.columns))
        
        # カラム一覧
        st.write("**カラム一覧:**")
        for col in analyzer.df.columns:
            st.write(f"• {col}")
    
    # 理解度分析の実行
    st.header("🧠 理解度分析")
    
    if st.button("分析実行", type="primary"):
        with st.spinner("分析中..."):
            # 理解度分析（分析方法に応じて実行）
            if analysis_method == "BERT分析（高精度）":
                analyzer.analyze_understanding_with_bert()
            else:
                analyzer.analyze_understanding()
            
            # 結果表示
            st.success("✅ 分析が完了しました！")
            
            # 統計サマリー
            scores = analyzer.understanding_scores
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("平均理解度", f"{np.mean(scores):.1f}")
            with col2:
                st.metric("最高スコア", f"{np.max(scores):.1f}")
            with col3:
                st.metric("最低スコア", f"{np.min(scores):.1f}")
            with col4:
                st.metric("標準偏差", f"{np.std(scores):.1f}")
            
            # 可視化
            st.header("📈 理解度可視化")
            
            # グラフを2列に分割
            col1, col2 = st.columns(2)
            
            with col1:
                # ヒストグラム
                st.subheader("理解度スコア分布")
                fig_hist = px.histogram(
                    x=scores, 
                    nbins=20,
                    title="理解度スコアの分布",
                    labels={'x': '理解度スコア', 'y': '人数'}
                )
                fig_hist.update_layout(showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # 散布図
                st.subheader("学生別理解度")
                student_ids = list(range(1, len(scores) + 1))
                fig_scatter = px.scatter(
                    x=student_ids, 
                    y=scores,
                    title="学生別理解度スコア",
                    labels={'x': '学生ID', 'y': '理解度スコア'}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                # 箱ひげ図
                st.subheader("理解度スコア分布（箱ひげ図）")
                fig_box = px.box(
                    y=scores,
                    title="理解度スコアの箱ひげ図",
                    labels={'y': '理解度スコア'}
                )
                st.plotly_chart(fig_box, use_container_width=True)
                
                # 円グラフ（理解度レベル）
                st.subheader("理解度レベル分布")
                levels = ['低い(0-40)', '普通(41-70)', '高い(71-100)']
                level_counts = [
                    sum(1 for score in scores if 0 <= score <= 40),
                    sum(1 for score in scores if 41 <= score <= 70),
                    sum(1 for score in scores if 71 <= score <= 100)
                ]
                
                fig_pie = px.pie(
                    values=level_counts,
                    names=levels,
                    title="理解度レベル分布"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # 詳細結果テーブル
            st.header("📊 詳細結果")
            
            # 理解度でソート
            result_df = analyzer.df.copy()
            result_df = result_df.sort_values('理解度スコア', ascending=False)
            
            st.subheader("理解度スコア付きデータ")
            st.dataframe(result_df, use_container_width=True)
            
            # 理解度レベル別統計
            st.subheader("理解度レベル別統計")
            level_stats = []
            for i, level in enumerate(levels):
                if i == 0:  # 低い
                    count = sum(1 for score in scores if 0 <= score <= 40)
                    avg_score = np.mean([score for score in scores if 0 <= score <= 40]) if count > 0 else 0
                elif i == 1:  # 普通
                    count = sum(1 for score in scores if 41 <= score <= 70)
                    avg_score = np.mean([score for score in scores if 41 <= score <= 70]) if count > 0 else 0
                else:  # 高い
                    count = sum(1 for score in scores if 71 <= score <= 100)
                    avg_score = np.mean([score for score in scores if 71 <= score <= 100]) if count > 0 else 0
                
                percentage = (count / len(scores)) * 100
                level_stats.append({
                    'レベル': level,
                    '人数': count,
                    '割合(%)': f"{percentage:.1f}%",
                    '平均スコア': f"{avg_score:.1f}"
                })
            
            level_df = pd.DataFrame(level_stats)
            st.dataframe(level_df, use_container_width=True)
            
            # BERT分析の場合は追加インサイトを表示
            if analysis_method == "BERT分析（高精度）" and hasattr(analyzer, 'get_bert_insights'):
                st.subheader("🤖 BERT分析インサイト")
                insights = analyzer.get_bert_insights()
                
                if isinstance(insights, list) and len(insights) > 0:
                    insight_data = []
                    for insight in insights:
                        insight_data.append({
                            'クラスター': f"グループ{insight['cluster']}",
                            '人数': insight['count'],
                            '平均スコア': f"{insight['avg_score']:.1f}",
                            'テーマ': insight['theme']
                        })
                    
                    insight_df = pd.DataFrame(insight_data)
                    st.dataframe(insight_df, use_container_width=True)
                    
                    st.info("💡 BERTによる意味解析で学生を理解パターン別にグループ化しました")
                else:
                    st.info("BERTインサイト分析が利用できません")
            
            # ダウンロードボタン
            st.header("💾 結果ダウンロード")
            
            csv = result_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 分析結果をCSVでダウンロード",
                data=csv,
                file_name='analysis_results.csv',
                mime='text/csv'
            )

else:
    # 初期画面
    st.info("👆 サイドバーからCSVファイルをアップロードするか、サンプルデータを使用してください。")
    
    # サンプルファイル形式の説明
    st.header("📝 CSVファイル形式")
    st.markdown("""
    アップロードするCSVファイルは以下の形式である必要があります：
    
    | 学生ID | 氏名 | 回答 |
    |--------|------|------|
    | 1 | 田中太郎 | データサイエンスについての回答文... |
    | 2 | 佐藤花子 | 機械学習についての回答文... |
    
    - **学生ID**: 学生を識別するID
    - **氏名**: 学生の氏名
    - **回答**: アンケートの回答文章
    """)
    
    # 機能説明
    st.header("🚀 システム機能")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 ルールベース分析**
        - 文章の長さ分析
        - 専門用語使用度
        - 具体例の言及
        - ポジティブ/ネガティブ表現
        
        **🤖 BERT分析**
        - 意味ベクトル解析
        - 文脈理解度評価
        - 深層学習特徴抽出
        - クラスター分析
        """)
    
    with col2:
        st.markdown("""
        **📈 可視化機能**
        - ヒストグラム
        - 箱ひげ図
        - 散布図
        - 円グラフ
        """)
    
    with col3:
        st.markdown("""
        **💾 出力機能**
        - 理解度スコア算出
        - 詳細統計情報
        - CSV形式ダウンロード
        - レベル別分析
        """)

# フッター
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit | データサイエンス授業用アンケート分析システム")