# DRL Stock Trading System

日本株の株式トレーディングを行うDeep Q-Network（DQN）強化学習システム

## 概要

このプロジェクトは、深層強化学習（DQN）を使用して日本株の自動トレーディングを行うシステムです。テクニカル指標を用いて市場状態を分析し、最適な売買タイミングを学習します。

## 機能

- **データ取得**: yfinanceを使用した日本株の過去データ取得
- **テクニカル分析**: 各種テクニカル指標の計算
  - 移動平均（SMA, EMA）
  - RSI（Relative Strength Index）
  - MACD（Moving Average Convergence Divergence）
  - ボリンジャーバンド
- **強化学習**: DQNアルゴリズムによる売買戦略の学習

## ファイル構成

```
.
├── symbols.txt                # 取引対象銘柄リスト（東証銘柄コード）
├── 1_get_data.py              # 株価データ取得スクリプト
├── 2_calc_tech.py             # テクニカル指標計算スクリプト
├── 3_dqn_train.py             # DQNモデル学習スクリプト
├── yfinance_data/             # 取得した株価データ（CSV形式）
├── technical_indicators/      # 計算されたテクニカル指標（CSV形式）
└── result/                    # 学習結果
```

## 使用方法

### 1. データ取得

```bash
python 1_get_data.py
```

`symbols.txt`に記載された銘柄の株価データを取得し、`yfinance_data/`フォルダにCSV形式で保存します。

**設定可能なパラメータ（1_get_data.py内）:**
- `DEFAULT_PERIOD`: データ取得期間（例: "1y", "2y", "3y", "5y"）
- `DEFAULT_DAYS_BACK`: 過去何日分のデータを取得するか
- `FORCE_UPDATE_DEFAULT`: 既存データを強制更新するかどうか

**出力:** `yfinance_data/` フォルダに各銘柄のCSVファイルが作成されます

### 2. テクニカル指標の計算

```bash
python 2_calc_tech.py
```

取得した株価データに対してテクニカル指標を計算し、`technical_indicators/`フォルダにCSV形式で保存します。

**出力:** `technical_indicators/` フォルダに各銘柄のテクニカル指標データが作成されます

### 3. DQNモデルの学習

```bash
python 3_dqn_train.py
```

強化学習によりトレーディング戦略を学習し、`result/`フォルダに学習結果を保存します。

**設定可能なパラメータ（3_dqn_train.py内）:**
- `RANDOM_SEED`: 再現性のためのシード値
- `USE_DAYS_BACK`: 学習に使用する過去日数
- `USE_DATE_FROM`, `USE_DATE_TO`: 学習期間の指定

**出力:** `result/` フォルダに以下のファイルが作成されます
- `{symbol}_trading_report.txt`: 各銘柄の取引レポート
- `overall_summary.txt`: 全体サマリーと翌取引日の推奨アクション

## 必要なライブラリ

```bash
pip install yfinance pandas numpy torch gymnasium
```

## 対象銘柄

`symbols.txt`に記載された銘柄を対象とします：
(記載例)
- 2502.T
- 3086.T
- 5801.T
- 6506.T
- 6645.T
- 6861.T
- 7182.T
- 9984.T

## 注意事項

- このシステムは教育・研究目的で作成されています
- 実際の投資判断に使用する際は自己責任でお願いします
- 過去のデータに基づく学習結果は将来の投資成果を保証するものではありません

## ライセンス

このプロジェクトは教育目的で作成されました。