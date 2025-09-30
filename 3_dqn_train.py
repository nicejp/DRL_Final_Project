import random
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

import torch
from torch import nn, optim
import gymnasium as gym
from gymnasium import spaces

# ========================
# 再現性のためのシード設定
# ========================
RANDOM_SEED = 24

# シード値の設定
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# PyTorchの決定的実行の設定
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ========================
# 日付範囲設定
# ========================
# データ使用期間の設定 (どちらか一つを設定してください)
#
# オプション1: 過去日数を指定する場合
USE_DAYS_BACK = 730           # 過去何日分のデータを使用するか (例: 730, None)
#
# オプション2: 開始日と終了日を指定する場合
USE_DATE_FROM = None           # 開始日 (例: "2020-01-01", None)
USE_DATE_TO = None             # 終了日 (例: "2021-12-31", None)
#
# 設定例:
# - 過去2年分: USE_DAYS_BACK = 730, USE_DATE_FROM = None, USE_DATE_TO = None
# - 過去1年分: USE_DAYS_BACK = 365, USE_DATE_FROM = None, USE_DATE_TO = None
# - 特定期間: USE_DAYS_BACK = None, USE_DATE_FROM = "2020-01-01", USE_DATE_TO = "2021-12-31"
# - 2020年のみ: USE_DAYS_BACK = None, USE_DATE_FROM = "2020-01-01", USE_DATE_TO = "2020-12-31"
# - 全期間: USE_DAYS_BACK = None, USE_DATE_FROM = None, USE_DATE_TO = None


class StockTradingEnv(gym.Env):
    """
    株式取引のためのカスタム環境クラス（CSV専用）
    """
    def __init__(self, symbol="", lookback=30, use_technical_indicators=True):
        super(StockTradingEnv, self).__init__()

        self.symbol = symbol
        self.lookback = lookback
        self.initial_cash = 1000000  # 初期資金100万円
        self.use_technical_indicators = use_technical_indicators

        # CSVファイルからデータを読み込み
        self.stock_data = self._load_data_from_csv(symbol)
        self.data_length = len(self.stock_data)

        # テクニカル指標データを読み込み
        self.technical_indicators = self._load_technical_indicators(symbol) if use_technical_indicators else None

        # 日付データも取得（取引記録用）
        self.dates = self._get_dates_from_csv(symbol)

        # 観測空間のサイズを計算
        base_features = lookback + 2  # 価格履歴 + ポジション情報
        technical_features = self._get_technical_features_count() if use_technical_indicators else 0
        total_features = base_features + technical_features

        # 観測空間: 直近lookback日の終値 + テクニカル指標 + ポジション情報（株数、現金）
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32
        )

        # 行動空間: 0=ホールド, 1=買い, 2=売り
        self.action_space = spaces.Discrete(3)

        # 取引記録用
        self.trading_records = []

        self.reset()

    def _load_data_from_csv(self, symbol):
        """CSVファイルから株式データを読み込み"""
        csv_file = f"yfinance_data/{symbol.replace('.', '_')}.csv"

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}. Please run fetch_stock_data.py first.")

        try:
            data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            if data.empty:
                raise ValueError(f"CSV file is empty: {csv_file}")

            # NaN値のチェックと処理
            nan_count = data['Close'].isna().sum()
            if nan_count > 0:
                print(f"Warning: {symbol} has {nan_count} NaN values in Close price. Filling with forward fill method.")
                # 前方補完（ffill）と後方補完（bfill）を使用
                data['Close'] = data['Close'].fillna(method='ffill').fillna(method='bfill')
                # それでもNaNが残る場合は平均値で補完
                if data['Close'].isna().any():
                    mean_value = data['Close'].mean()
                    data['Close'] = data['Close'].fillna(mean_value)
                    print(f"  Remaining NaN values filled with mean value: {mean_value:.2f}")

            # 日付範囲でフィルタリング
            original_length = len(data)
            data = self._filter_data_by_date_range(data, symbol)

            print(f"Loaded {len(data)} days of data for {symbol} from {csv_file} (original: {original_length} days)")
            return data['Close'].values

        except Exception as e:
            raise ValueError(f"Error loading CSV data for {symbol}: {e}")

    def _filter_data_by_date_range(self, data, symbol=None):
        """設定された日付範囲でデータをフィルタリング"""
        if USE_DAYS_BACK is not None:
            # 過去日数指定の場合
            end_date = data.index[-1]  # 最新日
            start_date = end_date - timedelta(days=USE_DAYS_BACK)
            filtered_data = data[data.index >= start_date]
            print(f"  Filtered by days back ({USE_DAYS_BACK} days): {start_date.date()} to {end_date.date()}")
            return filtered_data

        elif USE_DATE_FROM is not None or USE_DATE_TO is not None:
            # 開始日/終了日指定の場合
            filtered_data = data.copy()

            if USE_DATE_FROM is not None:
                start_date = pd.to_datetime(USE_DATE_FROM)
                # タイムゾーンを合わせる
                if data.index.tz is not None:
                    start_date = start_date.tz_localize(data.index.tz)
                filtered_data = filtered_data[filtered_data.index >= start_date]
                print(f"  Filtered from: {start_date.date()}")

            if USE_DATE_TO is not None:
                end_date = pd.to_datetime(USE_DATE_TO)
                # タイムゾーンを合わせる
                if data.index.tz is not None:
                    end_date = end_date.tz_localize(data.index.tz)
                filtered_data = filtered_data[filtered_data.index <= end_date]
                print(f"  Filtered to: {end_date.date()}")

            return filtered_data

        else:
            # フィルタリングなし（全期間使用）
            print(f"  Using all available data")
            return data

    def _get_dates_from_csv(self, symbol):
        """CSVファイルから日付データを取得"""
        csv_file = f"yfinance_data/{symbol.replace('.', '_')}.csv"

        try:
            data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            if not data.empty:
                # 同じ日付範囲フィルタリングを適用
                filtered_data = self._filter_data_by_date_range(data, symbol)
                return filtered_data.index.tolist()
        except Exception as e:
            print(f"Error reading dates from CSV: {e}")

        # フォールバック: 仮の日付を生成
        base_date = datetime(2023, 1, 1)
        return [base_date + timedelta(days=i) for i in range(len(self.stock_data))]

    def _load_technical_indicators(self, symbol):
        """テクニカル指標CSVファイルから指標データを読み込み"""
        indicators_file = f"technical_indicators/{symbol.replace('.', '_')}_indicators.csv"

        if not os.path.exists(indicators_file):
            print(f"Warning: Technical indicators file not found: {indicators_file}")
            print("Please run calculate_technical_indicators.py first.")
            return None

        try:
            indicators_data = pd.read_csv(indicators_file, index_col=0, parse_dates=True)
            if indicators_data.empty:
                print(f"Warning: Empty technical indicators file: {indicators_file}")
                return None

            # 日付範囲でフィルタリング（株価データと同じ期間）
            filtered_indicators = self._filter_data_by_date_range(indicators_data, symbol)
            print(f"Loaded {len(filtered_indicators)} technical indicator records for {symbol}")

            return filtered_indicators

        except Exception as e:
            print(f"Error loading technical indicators for {symbol}: {e}")
            return None

    def _get_technical_features_count(self):
        """テクニカル指標の特徴量数を取得"""
        if self.technical_indicators is None:
            return 0

        # 使用するテクニカル指標を選択（価格とボリュームは除外）
        exclude_columns = ['Price', 'Volume']
        technical_columns = [col for col in self.technical_indicators.columns if col not in exclude_columns]

        return len(technical_columns)

    def _get_technical_features(self, current_step):
        """現在のステップでのテクニカル指標特徴量を取得"""
        if self.technical_indicators is None:
            return np.array([])

        # インデックス範囲チェック
        if current_step >= len(self.technical_indicators):
            current_step = len(self.technical_indicators) - 1

        # 使用するテクニカル指標を選択（価格とボリュームは除外）
        exclude_columns = ['Price', 'Volume']
        technical_columns = [col for col in self.technical_indicators.columns if col not in exclude_columns]

        if not technical_columns:
            return np.array([])

        try:
            # 現在のステップの指標値を取得
            current_indicators = self.technical_indicators.iloc[current_step][technical_columns]

            # NaN値を0で置換
            current_indicators = current_indicators.fillna(0)

            # 正規化（各指標を-1から1の範囲に正規化）
            normalized_indicators = []
            for value in current_indicators.values:
                if np.isfinite(value):
                    # 簡易正規化: tanh関数を使用して-1から1の範囲に
                    normalized_value = np.tanh(value / 100.0)  # スケーリング調整
                else:
                    normalized_value = 0.0
                normalized_indicators.append(normalized_value)

            return np.array(normalized_indicators, dtype=np.float32)

        except Exception as e:
            print(f"Error getting technical features at step {current_step}: {e}")
            return np.zeros(len(technical_columns), dtype=np.float32)

    def reset(self, seed=None):
        """環境を初期化"""
        # 再現性のためのシード設定
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # データ長の最低要件をチェック
        if self.data_length < 2:
            raise ValueError(f"Data length ({self.data_length}) is too short. Need at least 2 data points.")

        # データ長をチェックしてlookbackを調整
        if self.data_length < self.lookback + 1:
            original_lookback = self.lookback
            # データ長に応じてlookbackを調整（最低1日、データがあれば最大データ長-1）
            self.lookback = max(1, min(self.data_length - 1, 5))
            print(f"Warning: Data length ({self.data_length}) is too short for lookback ({original_lookback}).")
            print(f"Automatically adjusted lookback to {self.lookback} days.")

            # 観測空間を再定義
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.lookback + 2,),
                dtype=np.float32
            )

        self.current_step = self.lookback
        self.cash = self.initial_cash
        self.shares = 0
        self.initial_portfolio_value = self.initial_cash
        self.previous_portfolio_value = self.initial_cash

        # 取引記録をリセット
        self.trading_records = []

        return self._get_observation()

    def _get_observation(self):
        """現在の観測を取得"""
        # インデックス範囲チェック
        if self.current_step >= self.data_length:
            self.current_step = self.data_length - 1

        # 直近lookback日の終値を取得
        start_idx = max(0, self.current_step - self.lookback)
        prices = self.stock_data[start_idx:self.current_step]

        # データが不足している場合はパディング
        if len(prices) < self.lookback:
            padding = np.full(self.lookback - len(prices), prices[0] if len(prices) > 0 else 100.0)
            prices = np.concatenate([padding, prices])

        # 正規化（空の配列の場合の対策）
        if len(prices) == 0:
            normalized_prices = np.zeros(self.lookback)
        elif np.std(prices) == 0 or np.isnan(np.std(prices)):
            normalized_prices = np.zeros(self.lookback)
        else:
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            if std_price == 0:
                normalized_prices = np.zeros(self.lookback)
            else:
                normalized_prices = (prices - mean_price) / (std_price + 1e-8)

        # ポジション情報（現金と株数）を正規化
        current_price = int(round(self.stock_data[self.current_step])) if self.current_step < self.data_length else int(round(self.stock_data[-1]))

        normalized_cash = self.cash / self.initial_cash
        normalized_shares = (self.shares * current_price) / self.initial_cash

        # 基本観測データ（価格履歴 + ポジション情報）
        base_observation = np.concatenate([
            normalized_prices,
            [normalized_cash, normalized_shares]
        ])

        # テクニカル指標を追加
        if self.use_technical_indicators and self.technical_indicators is not None:
            technical_features = self._get_technical_features(self.current_step)
            observation = np.concatenate([
                base_observation,
                technical_features
            ])
        else:
            observation = base_observation

        return observation.astype(np.float32)

    def step(self, action):
        """アクションを実行"""
        # 現在のステップ（観測日）の情報
        observation_date = self.dates[self.current_step] if self.current_step < len(self.dates) else self.dates[-1]

        # NaN値のチェック
        observation_price_raw = self.stock_data[self.current_step]
        if np.isnan(observation_price_raw):
            raise ValueError(f"Invalid price data (NaN) at step {self.current_step}")
        observation_price = int(round(observation_price_raw))

        # 次のステップに進む
        self.current_step += 1

        # エピソード終了判定（翌日のデータがあるかチェック）
        done = self.current_step >= self.data_length

        # 取引価格と日付（翌日のデータ）
        if not done:
            trade_price_raw = self.stock_data[self.current_step]
            if np.isnan(trade_price_raw):
                raise ValueError(f"Invalid price data (NaN) at step {self.current_step}")
            trade_price = int(round(trade_price_raw))
            trade_date = self.dates[self.current_step] if self.current_step < len(self.dates) else self.dates[-1]
        else:
            # データが終了した場合は、最終日の価格で評価のみ行う
            trade_price = observation_price
            trade_date = observation_date

        # 取引前の状態を記録
        old_cash = self.cash
        old_shares = self.shares

        # アクションの実行（翌日の価格で取引）
        if action == 1:  # 買い
            # 株を持っていない場合のみ買いを許可
            if self.shares == 0 and self.cash >= trade_price * 100:
                # 100株単位で購入可能な株数を計算
                shares_to_buy = int(self.cash // (trade_price * 100)) * 100
                if shares_to_buy > 0:
                    self.shares += shares_to_buy
                    self.cash -= shares_to_buy * trade_price

                    # 買い取引を記録
                    self.trading_records.append({
                        'date': trade_date,
                        'action': 'Buy',
                        'price': trade_price,
                        'shares': shares_to_buy,
                        'amount': shares_to_buy * trade_price,
                        'cash_before': old_cash,
                        'cash_after': self.cash,
                        'shares_before': old_shares,
                        'shares_after': self.shares
                    })
            # デバッグ: 買いアクションが拒否された理由を記録
            elif self.shares > 0:
                pass  # 既に株を保有しているため買いを拒否
            elif self.cash < trade_price * 100:
                pass  # 資金不足のため買いを拒否（100株購入に必要な資金がない）

        elif action == 2:  # 売り
            if self.shares > 0:
                sell_amount = self.shares * trade_price
                self.cash += sell_amount

                # 売り取引を記録
                self.trading_records.append({
                    'date': trade_date,
                    'action': 'Sell',
                    'price': trade_price,
                    'shares': self.shares,
                    'amount': sell_amount,
                    'cash_before': old_cash,
                    'cash_after': self.cash,
                    'shares_before': old_shares,
                    'shares_after': 0
                })

                self.shares = 0

        # 報酬計算（総資産の変動額、翌日の価格で評価）
        current_portfolio_value = self.cash + self.shares * trade_price
        reward = current_portfolio_value - self.previous_portfolio_value
        self.previous_portfolio_value = current_portfolio_value

        info = {
            'portfolio_value': current_portfolio_value,
            'cash': self.cash,
            'shares': self.shares,
            'current_price': trade_price
        }

        return self._get_observation(), reward, done, info


class ReplayBuffer:
    """通常のReplayバッファ"""
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.index = 0
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def push(self, experience):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.index] = experience

        self.index = (self.index + 1) % self.buffer_size

    def sample(self, batch_size):
        # 再現性のため、現在のnumpy randomの状態を保持
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        obs, action, reward, next_obs, done = zip(*[self.buffer[i] for i in indices])

        return (torch.stack(obs),
                torch.as_tensor(action),
                torch.as_tensor(reward, dtype=torch.float32),
                torch.stack(next_obs),
                torch.as_tensor(done, dtype=torch.uint8))


class DQNNetwork(nn.Module):
    """標準DQNネットワーク"""
    def __init__(self, state_dim, n_action):
        super(DQNNetwork, self).__init__()
        self.state_dim = state_dim
        self.n_action = n_action

        # 標準的な全結合層
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_action)
        )

    def forward(self, obs):
        return self.network(obs)

    def act(self, obs, epsilon):
        """ε-greedy行動選択"""
        if random.random() < epsilon:
            return random.randrange(self.n_action)
        else:
            with torch.no_grad():
                return torch.argmax(self.forward(obs.unsqueeze(0))).item()


class DQNAgent:
    """DQNエージェント"""
    def __init__(self, state_dim, n_action, device='cpu'):
        self.state_dim = state_dim
        self.n_action = n_action
        self.device = device

        # ネットワーク
        self.q_network = DQNNetwork(state_dim, n_action).to(device)
        self.target_network = DQNNetwork(state_dim, n_action).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 最適化
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.loss_func = nn.SmoothL1Loss(reduction='none')

        # ハイパーパラメータ
        self.gamma = 0.99
        self.target_update_interval = 1000
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 5000

        self.step_count = 0

    def get_epsilon(self):
        """現在のε値を計算"""
        epsilon = max(self.epsilon_end,
                     self.epsilon_start - (self.epsilon_start - self.epsilon_end) *
                     (self.step_count / self.epsilon_decay))
        return epsilon

    def act(self, obs):
        """行動選択"""
        epsilon = self.get_epsilon()
        return self.q_network.act(obs, epsilon)

    def update(self, replay_buffer, batch_size=32):
        """ネットワークの更新"""
        if len(replay_buffer) < batch_size:
            return None

        obs, action, reward, next_obs, done = replay_buffer.sample(batch_size)

        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        # 現在のQ値
        current_q_values = self.q_network(obs).gather(1, action.unsqueeze(1)).squeeze(1)

        # 標準DQN（ターゲットネットワークから最大Q値を取得）
        with torch.no_grad():
            next_q_values = self.target_network(next_obs).max(1)[0]
            target_q_values = reward + self.gamma * next_q_values * (1 - done)

        # 損失計算
        loss = self.loss_func(current_q_values, target_q_values).mean()

        # バックプロパゲーション
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ターゲットネットワーク更新
        if self.step_count % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.step_count += 1

        return loss.item()


def load_symbols_from_file(filename="symbols.txt"):
    """テキストファイルから銘柄リストを読み込む"""
    try:
        with open(filename, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(symbols)} symbols from {filename}: {symbols}")
        return symbols
    except FileNotFoundError:
        print(f"Symbol file {filename} not found. Using default symbol.")
        return ["7203.T"]
    except Exception as e:
        print(f"Error reading symbol file: {e}. Using default symbol.")
        return ["7203.T"]


def train_dqn_multiple_symbols(n_episodes=200, symbols_file="symbols.txt"):
    """複数銘柄に対応したDQN学習メイン関数（CSV専用）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 銘柄リストを読み込み
    symbols = load_symbols_from_file(symbols_file)

    # 各銘柄の結果を格納
    all_results = {}
    failed_symbols = []

    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Training DQN for symbol: {symbol}")
        print(f"{'='*50}")

        try:
            # 環境とエージェントの初期化（テクニカル指標を使用）
            env = StockTradingEnv(symbol=symbol, use_technical_indicators=True)

            # テクニカル指標の読み込み状況を表示
            if env.technical_indicators is not None:
                tech_features_count = env._get_technical_features_count()
                print(f"Technical indicators loaded: {tech_features_count} features")
                print(f"Total observation space: {env.observation_space.shape[0]} features")
            else:
                print("Warning: Technical indicators not available, using price data only")
                print(f"Observation space: {env.observation_space.shape[0]} features (price data only)")

            agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, device)
            replay_buffer = ReplayBuffer(50000)

            # 学習記録
            episode_rewards = []
            portfolio_values = []

            for episode in range(n_episodes):
                obs = env.reset(seed=RANDOM_SEED + episode)
                obs = torch.FloatTensor(obs).to(device)

                total_reward = 0
                episode_portfolio_values = []

                while True:
                    action = agent.act(obs)
                    next_obs, reward, done, info = env.step(action)
                    next_obs = torch.FloatTensor(next_obs).to(device)

                    # 経験をバッファに保存
                    replay_buffer.push([obs.cpu(), action, reward, next_obs.cpu(), done])

                    total_reward += reward
                    episode_portfolio_values.append(info['portfolio_value'])

                    # ネットワーク更新
                    _ = agent.update(replay_buffer)

                    obs = next_obs

                    if done:
                        break

                episode_rewards.append(total_reward)
                portfolio_values.append(episode_portfolio_values)

                if (episode + 1) % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    final_portfolio = episode_portfolio_values[-1] if episode_portfolio_values else 0
                    print(f"Episode {episode + 1}/{n_episodes}, "
                          f"Avg Reward: {avg_reward:.2f}, "
                          f"Final Portfolio: ¥{final_portfolio:,.0f}, "
                          f"Epsilon: {agent.get_epsilon():.3f}")

            # 結果を保存
            all_results[symbol] = {
                'agent': agent,
                'env': env,
                'episode_rewards': episode_rewards,
                'portfolio_values': portfolio_values
            }

        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"\n{'!'*50}")
            print(f"ERROR: Failed to train {symbol}")
            print(f"{'!'*50}")
            print(f"Error message: {error_msg}")

            # データ不足エラーの場合、より詳細な情報を表示
            if "too short" in error_msg.lower() or "empty" in error_msg.lower() or "data length" in error_msg.lower():
                print(f"\n原因: {symbol}には指定期間のデータが不足しています")
                try:
                    csv_file = f"yfinance_data/{symbol.replace('.', '_')}.csv"
                    if os.path.exists(csv_file):
                        data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                        if not data.empty:
                            print(f"  データ期間: {data.index[0].date()} ～ {data.index[-1].date()}")
                            print(f"  データ件数: {len(data)}件")
                except:
                    pass
            # NaN値エラーの場合
            elif "nan" in error_msg.lower() or "invalid price data" in error_msg.lower():
                print(f"\n原因: {symbol}のデータに無効な値(NaN)が含まれています")
                try:
                    csv_file = f"yfinance_data/{symbol.replace('.', '_')}.csv"
                    if os.path.exists(csv_file):
                        data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                        nan_count = data['Close'].isna().sum()
                        print(f"  NaN値の数: {nan_count}件")
                        if nan_count > 0:
                            print(f"  対策: データを再取得するか、NaN値を補完してください")
                except:
                    pass

            print(f"\nスタックトレース:")
            traceback.print_exc()
            print(f"{'!'*50}")
            print(f"{symbol}をスキップして次の銘柄に進みます...\n")

            failed_symbols.append({'symbol': symbol, 'error': error_msg})

    # 学習結果のサマリーを表示
    print(f"\n{'='*60}")
    print(f"学習完了サマリー")
    print(f"{'='*60}")
    print(f"成功: {len(all_results)}/{len(symbols)}銘柄")
    if all_results:
        print(f"成功した銘柄: {', '.join(all_results.keys())}")
    if failed_symbols:
        print(f"\n失敗: {len(failed_symbols)}/{len(symbols)}銘柄")
        for failed in failed_symbols:
            print(f"  - {failed['symbol']}: {failed['error'][:80]}")
    print(f"{'='*60}\n")

    return all_results


def test_agent(agent, env, device):
    """学習済みエージェントのテスト"""
    obs = env.reset(seed=RANDOM_SEED)
    obs = torch.FloatTensor(obs).to(device)

    portfolio_history = []
    actions_taken = []
    prices = []

    while True:
        # テスト時はε=0（常にgreedyに選択）
        with torch.no_grad():
            action = torch.argmax(agent.q_network(obs.unsqueeze(0))).item()

        next_obs, _, done, info = env.step(action)

        portfolio_history.append(info['portfolio_value'])
        actions_taken.append(action)
        prices.append(info['current_price'])

        obs = torch.FloatTensor(next_obs).to(device)

        if done:
            break

    return portfolio_history, actions_taken, prices, env.trading_records


def test_multiple_agents(all_results):
    """複数銘柄の学習済みエージェントをテスト"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_results = {}

    for symbol, result in all_results.items():
        print(f"\nTesting agent for {symbol}...")
        agent = result['agent']
        env = result['env']

        portfolio_history, actions_taken, prices, trading_records = test_agent(agent, env, device)
        test_results[symbol] = {
            'portfolio_history': portfolio_history,
            'actions_taken': actions_taken,
            'prices': prices,
            'trading_records': trading_records,
            'agent': agent,
            'env': env
        }

    return test_results


def predict_next_actions(test_results):
    """各銘柄について翌取引日の推奨アクションを予測"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    action_names = ['ホールド', '買い', '売り']  # 内部的な名前
    predictions = {}

    for symbol, data in test_results.items():
        agent = data['agent']
        env = data['env']

        # 環境を最新状態にリセット
        obs = env.reset(seed=RANDOM_SEED)
        obs = torch.FloatTensor(obs).to(device)

        # 最後のステップまで進める（データの最後まで）
        while True:
            with torch.no_grad():
                action = torch.argmax(agent.q_network(obs.unsqueeze(0))).item()

            next_obs, _, done, _ = env.step(action)

            if done:
                # doneになった時点で、current_stepはデータ末尾を指している
                # この状態の観測データを使って翌取引日の推奨を予測
                obs = torch.FloatTensor(next_obs).to(device)
                break

            obs = torch.FloatTensor(next_obs).to(device)

        # データ最終日の観測データから翌取引日のアクションを予測
        with torch.no_grad():
            q_values = agent.q_network(obs.unsqueeze(0))
            next_action = torch.argmax(q_values).item()

        # 現在のポジション情報を取得
        current_cash = env.cash
        current_shares = env.shares
        current_price = int(round(env.stock_data[-1])) if len(env.stock_data) > 0 else 0

        # ポジションに基づいて推奨アクションを調整
        q_values_copy = q_values.clone()

        if current_shares == 0:
            # 株を持っていない場合は「売り」を無効化
            q_values_copy[0][2] = float('-inf')

            # 100株購入に必要な資金がない場合は「買い」も無効化
            if current_cash < current_price * 100:
                q_values_copy[0][1] = float('-inf')

            # 有効なアクションから最良のものを選択
            next_action = torch.argmax(q_values_copy).item()

        elif current_shares > 0:
            # 株を持っている場合は「買い」を無効化（連続買いを防ぐため）
            q_values_copy[0][1] = float('-inf')

            # 有効なアクションから最良のものを選択
            next_action = torch.argmax(q_values_copy).item()

        # 推奨アクションを保存（ポジションに応じて名前を調整）
        if next_action == 0:  # ホールドの場合
            if current_shares == 0:
                predictions[symbol] = '様子見'  # ポジションを持たない中立スタンス
            else:
                predictions[symbol] = '保有継続'  # ポジションを持っているが継続して保有
        else:
            predictions[symbol] = action_names[next_action]  # 買い、売り

    return predictions


def calculate_trade_pnl(trading_records):
    """取引記録からP&Lを計算"""
    buy_prices = []
    buy_shares = []
    detailed_trades = []
    total_pnl = 0

    for record in trading_records:
        if record['action'] == 'Buy':
            buy_prices.append(record['price'])
            buy_shares.append(record['shares'])
        elif record['action'] == 'Sell' and buy_prices:
            # 平均取得価格を計算
            total_buy_amount = sum(price * shares for price, shares in zip(buy_prices, buy_shares))
            total_shares = sum(buy_shares)
            avg_buy_price = total_buy_amount / total_shares if total_shares > 0 else 0

            # P&L計算
            sell_price = record['price']
            shares_sold = record['shares']
            pnl = (sell_price - avg_buy_price) * shares_sold
            pnl_percentage = ((sell_price - avg_buy_price) / avg_buy_price * 100) if avg_buy_price > 0 else 0

            detailed_trades.append({
                'sell_date': record['date'],
                'avg_buy_price': avg_buy_price,
                'sell_price': sell_price,
                'shares': shares_sold,
                'pnl': pnl,
                'pnl_percentage': pnl_percentage
            })

            total_pnl += pnl
            # リセット（単純化のため）
            buy_prices = []
            buy_shares = []

    return detailed_trades, total_pnl


def generate_detailed_reports(test_results, next_action_predictions=None):
    """銘柄ごとの詳細レポートを生成してresultフォルダに保存"""
    os.makedirs("result", exist_ok=True)

    overall_summary = []

    for symbol, data in test_results.items():
        portfolio_history = data['portfolio_history']
        trading_records = data['trading_records']

        # 基本統計
        initial_value = 1000000
        final_value = portfolio_history[-1] if portfolio_history else initial_value
        total_return = ((final_value - initial_value) / initial_value) * 100

        # 取引P&L計算
        detailed_trades, total_trade_pnl = calculate_trade_pnl(trading_records)

        # レポート作成
        report_content = f"""
=== {symbol} 取引レポート ===
生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== ポートフォリオサマリー ===
初期資金: ¥{initial_value:,.0f}
最終評価額: ¥{final_value:,.0f}
総利益率: {total_return:+.2f}%
絶対利益: ¥{final_value - initial_value:+,.0f}

=== 取引サマリー ===
総取引回数: {len(trading_records)}
買い注文数: {len([r for r in trading_records if r['action'] == 'Buy'])}
売り注文数: {len([r for r in trading_records if r['action'] == 'Sell'])}
実現P&L: ¥{total_trade_pnl:+,.2f}

=== 詳細取引履歴 ===
"""

        # 全取引履歴
        if trading_records:
            report_content += "日付            アクション  価格        株数      金額\n"
            report_content += "-" * 60 + "\n"
            for record in trading_records:
                date_str = record['date'].strftime('%Y-%m-%d') if hasattr(record['date'], 'strftime') else str(record['date'])[:10]
                report_content += f"{date_str:15} {record['action']:8} ¥{record['price']:8.2f} {record['shares']:8} ¥{record['amount']:10,.0f}\n"
        else:
            report_content += "取引履歴なし\n"

        # 個別取引P&L
        if detailed_trades:
            report_content += "\n=== 個別取引損益 ===\n"
            report_content += "売却日          平均取得価格  売却価格   株数      損益       損益率\n"
            report_content += "-" * 80 + "\n"
            for trade in detailed_trades:
                date_str = trade['sell_date'].strftime('%Y-%m-%d') if hasattr(trade['sell_date'], 'strftime') else str(trade['sell_date'])[:10]
                report_content += f"{date_str:15} ¥{trade['avg_buy_price']:8.2f}   ¥{trade['sell_price']:8.2f} {trade['shares']:8} ¥{trade['pnl']:+9.2f} {trade['pnl_percentage']:+7.2f}%\n"

        # ファイル保存
        filename = f"result/{symbol.replace('.', '_')}_trading_report.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"レポート保存: {filename}")

        # 全体サマリー用データ
        overall_summary.append({
            'symbol': symbol,
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'trade_count': len(trading_records),
            'realized_pnl': total_trade_pnl
        })

    # 対象期間の情報を取得
    target_period_info = ""
    if USE_DAYS_BACK is not None:
        target_period_info = f"対象期間: 過去{USE_DAYS_BACK}日分"
    elif USE_DATE_FROM is not None or USE_DATE_TO is not None:
        start_date = USE_DATE_FROM if USE_DATE_FROM else "開始日指定なし"
        end_date = USE_DATE_TO if USE_DATE_TO else "終了日指定なし"
        target_period_info = f"対象期間: {start_date} ～ {end_date}"
    else:
        target_period_info = "対象期間: 全期間"

    # 全体サマリーレポート
    summary_content = f"""
=== 全銘柄取引サマリーレポート ===
生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{target_period_info}

=== 銘柄別成績 ===
銘柄       初期資金        最終評価額      利益率      実現損益     取引回数    翌取引日推奨
"""
    summary_content += "-" * 95 + "\n"

    total_initial = 0
    total_final = 0
    total_realized_pnl = 0

    for summary in overall_summary:
        next_action = next_action_predictions.get(summary['symbol'], '不明') if next_action_predictions else '不明'
        summary_content += f"{summary['symbol']:10} ¥{summary['initial_value']:>10,.0f} ¥{summary['final_value']:>13,.0f} {summary['total_return']:>8.2f}% ¥{summary['realized_pnl']:>10,.0f} {summary['trade_count']:>8}    {next_action:>8}\n"
        total_initial += summary['initial_value']
        total_final += summary['final_value']
        total_realized_pnl += summary['realized_pnl']


    # 全体サマリー保存
    with open("result/overall_summary.txt", 'w', encoding='utf-8') as f:
        f.write(summary_content)

    print("全体サマリー保存: result/overall_summary.txt")

    return overall_summary


def generate_summary_stats(test_results):
    """複数銘柄の結果統計を生成"""
    summary_stats = {}

    for symbol, data in test_results.items():
        portfolio_history = data['portfolio_history']
        actions_taken = data['actions_taken']
        trading_records = data['trading_records']

        # 統計計算
        initial_value = 1000000
        final_value = portfolio_history[-1] if portfolio_history else initial_value
        total_return = ((final_value - initial_value) / initial_value) * 100
        action_counts = [actions_taken.count(j) for j in range(3)]

        # 取引記録の数を確認して表示
        buy_records = [r for r in trading_records if r['action'] == 'Buy']
        sell_records = [r for r in trading_records if r['action'] == 'Sell']
        print(f"  {symbol}: Buy records: {len(buy_records)}, Sell records: {len(sell_records)}")

        summary_stats[symbol] = {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'action_counts': action_counts
        }

    # 複数銘柄の統計サマリー
    print("\n" + "="*80)
    print("MULTI-SYMBOL TRADING RESULTS SUMMARY")
    print("="*80)

    for symbol, stats in summary_stats.items():
        print(f"\n=== {symbol} ===")
        print(f"Initial Portfolio Value: ¥{stats['initial_value']:,.0f}")
        print(f"Final Portfolio Value: ¥{stats['final_value']:,.0f}")
        print(f"Total Return: {stats['total_return']:+.2f}%")
        print(f"Actions - Hold: {stats['action_counts'][0]}, Buy: {stats['action_counts'][1]}, Sell: {stats['action_counts'][2]}")

    # 全体サマリー
    total_initial = sum(stats['initial_value'] for stats in summary_stats.values())
    total_final = sum(stats['final_value'] for stats in summary_stats.values())
    overall_return = ((total_final - total_initial) / total_initial) * 100

    print(f"\n=== OVERALL PORTFOLIO ===")
    print(f"Total Initial Value: ¥{total_initial:,.0f}")
    print(f"Total Final Value: ¥{total_final:,.0f}")
    print(f"Overall Return: {overall_return:+.2f}%")
    print(f"Number of Symbols: {len(summary_stats)}")

    return summary_stats


def check_technical_indicators_availability(symbols):
    """テクニカル指標ファイルの存在確認"""
    print("Checking technical indicators availability...")
    missing_indicators = []

    for symbol in symbols:
        indicators_file = f"technical_indicators/{symbol.replace('.', '_')}_indicators.csv"
        if not os.path.exists(indicators_file):
            missing_indicators.append(symbol)

    if missing_indicators:
        print("Warning: Technical indicators not found for the following symbols:")
        for symbol in missing_indicators:
            print(f"  - {symbol}")
        print("\nTo generate technical indicators, run:")
        print("  python calculate_technical_indicators.py")
        print("\nProceeding with price data only for these symbols...")
        return False
    else:
        print("✓ Technical indicators available for all symbols")
        return True


if __name__ == "__main__":
    # 結果格納先フォルダがなければ作成
    os.makedirs("result", exist_ok=True)

    print("Starting Multi-Symbol DQN Trading for Stock Trading (CSV Mode)...")
    print("="*60)

    # 日付範囲設定を表示
    print("Date Range Configuration:")
    if USE_DAYS_BACK is not None:
        print(f"  Using last {USE_DAYS_BACK} days of data")
    elif USE_DATE_FROM is not None or USE_DATE_TO is not None:
        print(f"  Using date range: {USE_DATE_FROM or 'beginning'} to {USE_DATE_TO or 'end'}")
    else:
        print(f"  Using all available data")
    print("="*60)

    try:
        # 銘柄リストを読み込み
        symbols = load_symbols_from_file("symbols.txt")

        # テクニカル指標の存在確認
        check_technical_indicators_availability(symbols)

        # 通常の銘柄リストで実行
        all_results = train_dqn_multiple_symbols(n_episodes=10, symbols_file="symbols.txt")

        if not all_results:
            print("No symbols were successfully trained. Please check your CSV files.")
            print("Run fetch_stock_data.py first to download stock data.")
            exit(1)

        # 複数銘柄のテスト実行
        print("\n" + "="*60)
        print("Testing trained agents for all symbols...")
        print("="*60)
        test_results = test_multiple_agents(all_results)

        # 複数銘柄の結果統計生成
        print("\nGenerating multi-symbol statistics...")
        summary_stats = generate_summary_stats(test_results)

        # 翌取引日の推奨アクション予測
        next_action_predictions = predict_next_actions(test_results)

        # 詳細レポート生成
        print("\nGenerating detailed trading reports...")
        generate_detailed_reports(test_results, next_action_predictions)

    except Exception as e:
        print(f"\nProgram stopped due to error: {e}")
        print("Please make sure CSV files exist in the data/ folder.")
        print("Run fetch_stock_data.py first to download stock data.")