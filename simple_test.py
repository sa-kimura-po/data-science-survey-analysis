from survey_analyzer import SurveyAnalyzer

# テスト実行
analyzer = SurveyAnalyzer()

# CSVファイル読み込み
print("CSVファイルを読み込み中...")
if analyzer.load_csv('survey_data.csv'):
    print("読み込み成功！")
    
    # 理解度分析
    print("理解度分析を実行中...")
    analyzer.analyze_understanding()
    print("分析完了！")
    
    # 結果表示
    print("\n=== 分析結果サンプル ===")
    print(analyzer.df[['学生ID', '氏名', '理解度スコア']].head())
    
    # 統計情報
    import numpy as np
    scores = analyzer.understanding_scores
    print(f"\n平均理解度: {np.mean(scores):.2f}")
    print(f"最高スコア: {np.max(scores):.2f}")
    print(f"最低スコア: {np.min(scores):.2f}")
    
    print("\n可視化をスキップして実行完了！")
else:
    print("ファイル読み込みに失敗しました")