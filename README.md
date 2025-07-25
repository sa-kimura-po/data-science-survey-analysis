# 📊 データサイエンス授業アンケート分析システム

学生のアンケート回答を**AI**で分析して理解度を可視化するPythonシステムです。  
ルールベース分析と**BERT深層学習**による高精度分析の両方に対応しています。

## ✨ 特徴

### 🤖 AI分析エンジン
- **ルールベース分析**: 高速で軽量な分析
- **BERT分析**: 日本語BERTによる高精度な意味理解分析
- **クラスター分析**: 学生を理解パターン別にグループ化

### 📈 可視化機能  
- インタラクティブなWebUI（Streamlit）
- 4種類のグラフ（ヒストグラム、箱ひげ図、散布図、円グラフ）
- リアルタイム分析結果表示

### 💾 データ処理
- CSVファイルアップロード対応
- 分析結果のCSVエクスポート
- サンプルデータ内蔵

## 🚀 クイックスタート

### 1. 環境セットアップ
```bash
# リポジトリをクローン
git clone https://github.com/[your-username]/data-science-survey-analysis.git
cd data-science-survey-analysis

# 依存関係をインストール
pip install -r requirements.txt
```

### 2. WebUIを起動
```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` にアクセス

### 3. 分析実行
1. サイドバーで分析方法を選択
   - **ルールベース分析**: 高速分析
   - **BERT分析（高精度）**: AI深層学習分析
2. CSVファイルをアップロード（またはサンプルデータ使用）
3. 「分析実行」ボタンをクリック
4. 結果をWebUIで確認・ダウンロード

## 📝 CSVファイル形式

```csv
学生ID,氏名,回答
1,田中太郎,データサイエンスの授業を通じて機械学習の基本概念を理解できました...
2,佐藤花子,今回の授業内容は少し難しく感じました...
```

## 🧠 分析手法

### ルールベース分析
- 文章の詳細度分析
- 専門用語使用度評価  
- 具体例・数値言及検出
- 感情表現分析

### BERT分析（高精度）
- **意味ベクトル解析**: 文章の深層的意味理解
- **文脈理解**: 前後の文脈を考慮した分析
- **特徴抽出**: 深層学習による高次元特徴抽出
- **クラスタリング**: 理解パターンによるグループ化

## 📊 分析結果

### 基本統計
- 平均理解度、標準偏差、最高・最低スコア
- 理解度レベル分布（低い・普通・高い）

### BERT分析限定
- 学生の理解パターン別クラスター分析
- 意味的類似性による分類
- 深層学習インサイト

## 🛠️ 技術スタック

- **Backend**: Python 3.13+
- **AI/ML**: 
  - Transformers (Hugging Face)
  - PyTorch
  - Sentence Transformers
  - scikit-learn
- **Frontend**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn
- **NLP**: 日本語BERT (cl-tohoku/bert-base-japanese)

## 📁 プロジェクト構造

```
data-science-survey-analysis/
├── app.py                 # Streamlit WebUI
├── survey_analyzer.py     # ルールベース分析
├── bert_analyzer.py       # BERT分析エンジン
├── requirements.txt       # 依存関係
├── survey_data.csv       # サンプルデータ
├── simple_test.py        # テスト用スクリプト
└── README.md
```

## 🎯 使用例

### 教育現場での活用
- 授業理解度の定量的評価
- 学習困難な学生の早期発見
- カリキュラム改善のためのデータ分析

### 研究活動での活用  
- 教育効果測定
- 学習分析研究
- テキストマイニング研究

## 🔧 カスタマイズ

### 専門用語辞書の拡張
`bert_analyzer.py`の`technical_terms`リストを編集

### 分析パラメータの調整
各分析関数内のスコア重み付けを変更可能

## 📜 ライセンス

MIT License

## 🤝 コントリビューション

プルリクエスト、イシュー報告を歓迎します！

## 📞 サポート

質問やバグ報告は[Issues](https://github.com/[your-username]/data-science-survey-analysis/issues)まで

---

**Developed with ❤️ for Education Technology**