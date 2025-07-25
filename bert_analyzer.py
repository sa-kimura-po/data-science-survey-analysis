import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

# BERT関連のインポート
try:
    from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: BERT libraries not available. Using fallback analysis.")

class BertSurveyAnalyzer:
    def __init__(self):
        self.df = None
        self.understanding_scores = None
        self.bert_model = None
        self.tokenizer = None
        self.sentence_model = None
        
        if BERT_AVAILABLE:
            self._load_bert_models()
    
    def _load_bert_models(self):
        """BERTモデルを読み込む"""
        try:
            # 日本語BERT（軽量版）
            model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            
            # Sentence Transformer（文章の意味ベクトル化用）
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            print("✅ BERTモデルの読み込み完了")
        except Exception as e:
            print(f"❌ BERTモデル読み込みエラー: {e}")
            self.bert_model = None
            self.tokenizer = None
            self.sentence_model = None
    
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
    
    def analyze_understanding_with_bert(self, text_column='回答'):
        """BERTを使った高度な理解度分析"""
        if self.df is None:
            print("データが読み込まれていません")
            return
        
        understanding_scores = []
        
        for i, text in enumerate(self.df[text_column]):
            print(f"分析中... {i+1}/{len(self.df)}")
            
            if BERT_AVAILABLE and self.bert_model is not None:
                score = self._bert_understanding_score(str(text))
            else:
                # フォールバック: ルールベース分析
                score = self._fallback_understanding_score(str(text))
            
            understanding_scores.append(score)
        
        self.df['理解度スコア'] = understanding_scores
        self.understanding_scores = understanding_scores
        
        print("BERT理解度分析が完了しました")
        return understanding_scores
    
    def _bert_understanding_score(self, text):
        """BERTを使った理解度スコア計算"""
        try:
            # 基本スコア
            base_score = 0
            
            # 1. 文章の意味ベクトル化
            if self.sentence_model:
                embedding = self.sentence_model.encode([text])[0]
                # ベクトルの情報密度（標準偏差）を理解度の指標とする
                semantic_density = np.std(embedding) * 100
                base_score += min(semantic_density, 30)
            
            # 2. BERT特徴量抽出
            if self.tokenizer and self.bert_model:
                # トークン化
                inputs = self.tokenizer(text, return_tensors="pt", 
                                      truncation=True, max_length=512, padding=True)
                
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    # 最後の隠れ層の平均を取得
                    hidden_states = outputs.last_hidden_state
                    sentence_embedding = torch.mean(hidden_states, dim=1).squeeze()
                
                # 埋め込みベクトルの活性化度合い
                activation_score = torch.norm(sentence_embedding).item() * 5
                base_score += min(activation_score, 25)
            
            # 3. 専門用語分析（BERT強化版）
            technical_terms = [
                'データサイエンス', 'アルゴリズム', '機械学習', '統計', '分析', '予測',
                '回帰', '分類', 'クラスタリング', 'ニューラルネットワーク', 'ディープラーニング',
                'Python', 'R', 'SQL', 'ビッグデータ', '可視化', 'モデル', '精度',
                '仮説検定', '相関', '分散', '標準偏差', 'データベース', 'API'
            ]
            
            # BERTトークナイザーでの専門用語検出
            if self.tokenizer:
                tokens = self.tokenizer.tokenize(text)
                technical_matches = 0
                for term in technical_terms:
                    term_tokens = self.tokenizer.tokenize(term)
                    if any(tt in tokens for tt in term_tokens):
                        technical_matches += 1
                
                technical_score = technical_matches * 3
                base_score += min(technical_score, 20)
            
            # 4. 文章構造分析
            sentences = re.split(r'[。！？]', text)
            valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
            
            # 文章の複雑性
            complexity_score = len(valid_sentences) * 2
            base_score += min(complexity_score, 15)
            
            # 5. 感情・理解表現の検出（日本語特化）
            positive_patterns = [
                r'理解(でき|し)', r'わかっ', r'面白', r'興味深', r'勉強になっ',
                r'よく理解', r'具体的', r'詳しく', r'実際', r'応用'
            ]
            negative_patterns = [
                r'わからな', r'難し', r'理解できな', r'不明', r'混乱',
                r'よくわからな', r'ピンとこな', r'理解不足'
            ]
            
            positive_score = sum(3 for pattern in positive_patterns if re.search(pattern, text))
            negative_score = sum(-2 for pattern in negative_patterns if re.search(pattern, text))
            
            emotion_score = positive_score + negative_score
            base_score += emotion_score
            
            # 最終スコアの正規化 (0-100)
            final_score = max(0, min(100, base_score))
            
            return round(final_score, 2)
            
        except Exception as e:
            print(f"BERT分析エラー: {e}")
            return self._fallback_understanding_score(text)
    
    def _fallback_understanding_score(self, text):
        """BERTが使えない場合のフォールバック分析"""
        score = 0
        
        # 文章の長さ
        length_score = min(len(text) / 100, 3) * 10
        
        # 専門用語
        technical_terms = [
            'データ', 'アルゴリズム', '機械学習', '統計', '分析', '予測',
            'Python', 'R', 'SQL', '可視化', 'モデル'
        ]
        technical_score = sum(1 for term in technical_terms if term in text) * 5
        
        # 具体例
        example_patterns = [r'\d+', r'例えば', r'具体的に', r'実際に']
        example_score = sum(1 for pattern in example_patterns if re.search(pattern, text)) * 3
        
        # 感情表現
        positive_patterns = [r'理解できた', r'面白い', r'勉強になった']
        negative_patterns = [r'わからない', r'難しい', r'理解できない']
        
        positive_score = sum(1 for pattern in positive_patterns if re.search(pattern, text)) * 8
        negative_score = sum(1 for pattern in negative_patterns if re.search(pattern, text)) * -5
        
        total_score = length_score + technical_score + example_score + positive_score + negative_score
        
        return max(0, min(100, round(total_score, 2)))
    
    def get_bert_insights(self, text_column='回答'):
        """BERT分析による詳細インサイト"""
        if not BERT_AVAILABLE or self.sentence_model is None:
            return "BERT分析が利用できません"
        
        insights = []
        
        try:
            # 全文章の意味ベクトル取得
            texts = [str(text) for text in self.df[text_column]]
            embeddings = self.sentence_model.encode(texts)
            
            # クラスタリング分析
            from sklearn.cluster import KMeans
            n_clusters = min(3, len(texts))
            
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(embeddings)
                
                # クラスター分析結果
                for i in range(n_clusters):
                    cluster_texts = [texts[j] for j in range(len(texts)) if clusters[j] == i]
                    cluster_scores = [self.understanding_scores[j] for j in range(len(texts)) if clusters[j] == i]
                    
                    insights.append({
                        'cluster': i + 1,
                        'count': len(cluster_texts),
                        'avg_score': np.mean(cluster_scores),
                        'theme': f"理解パターン{i+1}"
                    })
            
            return insights
            
        except Exception as e:
            return f"インサイト分析エラー: {e}"
    
    def visualize_understanding(self):
        """理解度を可視化する"""
        if self.understanding_scores is None:
            print("理解度分析が実行されていません")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 日本語フォント設定
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Yu Gothic', 'Meiryo']
        
        # ヒストグラム
        axes[0, 0].hist(self.understanding_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('BERT理解度スコアの分布')
        axes[0, 0].set_xlabel('理解度スコア')
        axes[0, 0].set_ylabel('人数')
        
        # 箱ひげ図
        axes[0, 1].boxplot(self.understanding_scores)
        axes[0, 1].set_title('BERT理解度スコアの箱ひげ図')
        axes[0, 1].set_ylabel('理解度スコア')
        
        # 散布図
        student_ids = range(1, len(self.understanding_scores) + 1)
        axes[1, 0].scatter(student_ids, self.understanding_scores, alpha=0.6, color='orange')
        axes[1, 0].set_title('学生別BERT理解度スコア')
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
        axes[1, 1].set_title('BERT理解度レベル分布')
        
        plt.tight_layout()
        plt.savefig('bert_understanding_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 統計サマリー
        print("\n=== BERT理解度分析結果 ===")
        print(f"平均理解度: {np.mean(self.understanding_scores):.2f}")
        print(f"標準偏差: {np.std(self.understanding_scores):.2f}")
        print(f"最高スコア: {np.max(self.understanding_scores):.2f}")
        print(f"最低スコア: {np.min(self.understanding_scores):.2f}")
        print(f"中央値: {np.median(self.understanding_scores):.2f}")
        
        # BERTインサイト表示
        insights = self.get_bert_insights()
        if isinstance(insights, list):
            print(f"\n=== BERT分析インサイト ===")
            for insight in insights:
                print(f"クラスター{insight['cluster']}: {insight['count']}人, 平均スコア: {insight['avg_score']:.2f}")
    
    def save_results(self, output_path='bert_analysis_results.csv'):
        """分析結果をCSVファイルに保存"""
        if self.df is None:
            print("分析結果がありません")
            return
        
        self.df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"BERT分析結果を {output_path} に保存しました")

# 使用例
if __name__ == "__main__":
    analyzer = BertSurveyAnalyzer()
    
    # CSVファイル読み込み
    if analyzer.load_csv('survey_data.csv'):
        # BERT理解度分析
        analyzer.analyze_understanding_with_bert()
        
        # 可視化
        analyzer.visualize_understanding()
        
        # 結果保存
        analyzer.save_results()