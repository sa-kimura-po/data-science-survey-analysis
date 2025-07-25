import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

class SurveyAnalyzer:
    def __init__(self):
        self.df = None
        self.understanding_scores = None
        
    def load_csv(self, filepath):
        """CSVファイルを読み込む"""
        try:
            self.df = pd.read_csv(filepath, encoding='utf-8')
            print(f"データを正常に読み込みました。レコード数: {len(self.df)}")
            print(f"カラム: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"ファイル読み込みエラー: {e}")
            return False
    
    def analyze_understanding(self, text_column='回答'):
        """文章から理解度を分析する"""
        if self.df is None:
            print("データが読み込まれていません")
            return
        
        understanding_scores = []
        
        for text in self.df[text_column]:
            score = self._calculate_understanding_score(str(text))
            understanding_scores.append(score)
        
        self.df['理解度スコア'] = understanding_scores
        self.understanding_scores = understanding_scores
        
        print("理解度分析が完了しました")
        return understanding_scores
    
    def _calculate_understanding_score(self, text):
        """個別の文章から理解度スコアを計算"""
        score = 0
        
        # 文章の長さ（詳細度の指標）
        length_score = min(len(text) / 100, 3) * 10
        
        # 専門用語の使用
        technical_terms = [
            'データ', 'アルゴリズム', '機械学習', '統計', '分析', '予測',
            '回帰', '分類', 'クラスタリング', 'ニューラルネットワーク',
            'Python', 'R', 'SQL', 'ビッグデータ', '可視化'
        ]
        technical_score = sum(1 for term in technical_terms if term in text) * 5
        
        # 具体例や数値の言及
        example_patterns = [r'\d+', r'例えば', r'具体的に', r'実際に', r'つまり']
        example_score = sum(1 for pattern in example_patterns if re.search(pattern, text)) * 3
        
        # ネガティブ表現の検出
        negative_patterns = [r'わからない', r'難しい', r'理解できない', r'不明']
        negative_score = sum(1 for pattern in negative_patterns if re.search(pattern, text)) * -5
        
        # ポジティブ表現の検出
        positive_patterns = [r'理解できた', r'面白い', r'勉強になった', r'よくわかった']
        positive_score = sum(1 for pattern in positive_patterns if re.search(pattern, text)) * 8
        
        total_score = length_score + technical_score + example_score + negative_score + positive_score
        
        # 0-100の範囲に正規化
        normalized_score = max(0, min(100, total_score))
        
        return round(normalized_score, 2)
    
    def visualize_understanding(self):
        """理解度を可視化する"""
        if self.understanding_scores is None:
            print("理解度分析が実行されていません")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ヒストグラム
        axes[0, 0].hist(self.understanding_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('理解度スコアの分布')
        axes[0, 0].set_xlabel('理解度スコア')
        axes[0, 0].set_ylabel('人数')
        
        # 箱ひげ図
        axes[0, 1].boxplot(self.understanding_scores)
        axes[0, 1].set_title('理解度スコアの箱ひげ図')
        axes[0, 1].set_ylabel('理解度スコア')
        
        # 散布図（学生ID vs 理解度）
        student_ids = range(1, len(self.understanding_scores) + 1)
        axes[1, 0].scatter(student_ids, self.understanding_scores, alpha=0.6, color='orange')
        axes[1, 0].set_title('学生別理解度スコア')
        axes[1, 0].set_xlabel('学生ID')
        axes[1, 0].set_ylabel('理解度スコア')
        
        # 理解度レベル分類
        levels = ['低い(0-40)', '普通(41-70)', '高い(71-100)']
        level_counts = [
            sum(1 for score in self.understanding_scores if 0 <= score <= 40),
            sum(1 for score in self.understanding_scores if 41 <= score <= 70),
            sum(1 for score in self.understanding_scores if 71 <= score <= 100)
        ]
        
        axes[1, 1].pie(level_counts, labels=levels, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('理解度レベル分布')
        
        plt.tight_layout()
        plt.savefig('understanding_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 統計サマリー
        print("\n=== 理解度分析結果 ===")
        print(f"平均理解度: {np.mean(self.understanding_scores):.2f}")
        print(f"標準偏差: {np.std(self.understanding_scores):.2f}")
        print(f"最高スコア: {np.max(self.understanding_scores):.2f}")
        print(f"最低スコア: {np.min(self.understanding_scores):.2f}")
        print(f"中央値: {np.median(self.understanding_scores):.2f}")
        
        print(f"\n理解度レベル分布:")
        print(f"低い(0-40): {level_counts[0]}人 ({level_counts[0]/len(self.understanding_scores)*100:.1f}%)")
        print(f"普通(41-70): {level_counts[1]}人 ({level_counts[1]/len(self.understanding_scores)*100:.1f}%)")
        print(f"高い(71-100): {level_counts[2]}人 ({level_counts[2]/len(self.understanding_scores)*100:.1f}%)")
    
    def save_results(self, output_path='analysis_results.csv'):
        """分析結果をCSVファイルに保存"""
        if self.df is None:
            print("分析結果がありません")
            return
        
        self.df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"分析結果を {output_path} に保存しました")

# 使用例
if __name__ == "__main__":
    analyzer = SurveyAnalyzer()
    
    # CSVファイル読み込み
    if analyzer.load_csv('survey_data.csv'):
        # 理解度分析
        analyzer.analyze_understanding()
        
        # 可視化
        analyzer.visualize_understanding()
        
        # 結果保存
        analyzer.save_results()