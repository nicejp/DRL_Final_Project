import os
import pandas as pd
import numpy as np
from datetime import datetime

def calculate_sma(data, window):
    """単純移動平均 (Simple Moving Average)"""
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    """指数移動平均 (Exponential Moving Average)"""
    return data.ewm(span=window).mean()

def calculate_rsi(data, window=14):
    """RSI (Relative Strength Index)"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """MACD (Moving Average Convergence Divergence)"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': histogram
    })

def calculate_bollinger_bands(data, window=20, std_dev=2):
    """ボリンジャーバンド (Bollinger Bands)"""
    sma = calculate_sma(data, window)
    std = data.rolling(window=window).std()

    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)

    return pd.DataFrame({
        'Upper': upper_band,
        'Middle': sma,
        'Lower': lower_band,
        'Width': upper_band - lower_band,
        'Position': (data - lower_band) / (upper_band - lower_band)
    })

def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    """ストキャスティクス (Stochastic Oscillator)"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()

    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()

    return pd.DataFrame({
        'K': k_percent,
        'D': d_percent
    })

def calculate_williams_r(high, low, close, window=14):
    """ウィリアムズ%R (Williams %R)"""
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()

    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r

def calculate_atr(high, low, close, window=14):
    """ATR (Average True Range)"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()

    return atr

def calculate_cci(high, low, close, window=20):
    """CCI (Commodity Channel Index)"""
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=window).mean()
    mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))

    cci = (typical_price - sma_tp) / (0.015 * mad)
    return cci

def calculate_adx(high, low, close, window=14):
    """ADX (Average Directional Index)"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)

    atr = calculate_atr(high, low, close, window)

    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=window).mean()

    return pd.DataFrame({
        'ADX': adx,
        'Plus_DI': plus_di,
        'Minus_DI': minus_di
    })

def calculate_obv(close, volume):
    """OBV (On-Balance Volume)"""
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])

    return pd.Series(obv, index=close.index)

def calculate_momentum(data, window=10):
    """モメンタム (Momentum)"""
    return data.diff(window)

def calculate_roc(data, window=10):
    """ROC (Rate of Change)"""
    return ((data / data.shift(window)) - 1) * 100

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

def calculate_all_indicators(symbol):
    """すべてのテクニカル指標を計算"""
    csv_file = f"yfinance_data/{symbol.replace('.', '_')}.csv"

    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return None

    try:
        # データ読み込み
        data = pd.read_csv(csv_file, index_col=0, parse_dates=True)

        if data.empty:
            print(f"Empty data for {symbol}")
            return None

        print(f"Calculating technical indicators for {symbol}...")

        # 各指標を計算
        indicators = pd.DataFrame(index=data.index)

        # 移動平均系
        indicators['SMA_5'] = calculate_sma(data['Close'], 5)
        indicators['SMA_10'] = calculate_sma(data['Close'], 10)
        indicators['SMA_20'] = calculate_sma(data['Close'], 20)
        indicators['SMA_50'] = calculate_sma(data['Close'], 50)
        indicators['SMA_200'] = calculate_sma(data['Close'], 200)

        indicators['EMA_12'] = calculate_ema(data['Close'], 12)
        indicators['EMA_26'] = calculate_ema(data['Close'], 26)
        indicators['EMA_50'] = calculate_ema(data['Close'], 50)

        # オシレーター系
        indicators['RSI_14'] = calculate_rsi(data['Close'])
        indicators['RSI_21'] = calculate_rsi(data['Close'], 21)

        # MACD
        macd_data = calculate_macd(data['Close'])
        indicators['MACD'] = macd_data['MACD']
        indicators['MACD_Signal'] = macd_data['Signal']
        indicators['MACD_Histogram'] = macd_data['Histogram']

        # ボリンジャーバンド
        bb_data = calculate_bollinger_bands(data['Close'])
        indicators['BB_Upper'] = bb_data['Upper']
        indicators['BB_Middle'] = bb_data['Middle']
        indicators['BB_Lower'] = bb_data['Lower']
        indicators['BB_Width'] = bb_data['Width']
        indicators['BB_Position'] = bb_data['Position']

        # ストキャスティクス
        stoch_data = calculate_stochastic(data['High'], data['Low'], data['Close'])
        indicators['Stoch_K'] = stoch_data['K']
        indicators['Stoch_D'] = stoch_data['D']

        # その他の指標
        indicators['Williams_R'] = calculate_williams_r(data['High'], data['Low'], data['Close'])
        indicators['ATR'] = calculate_atr(data['High'], data['Low'], data['Close'])
        indicators['CCI'] = calculate_cci(data['High'], data['Low'], data['Close'])

        # ADX
        adx_data = calculate_adx(data['High'], data['Low'], data['Close'])
        indicators['ADX'] = adx_data['ADX']
        indicators['Plus_DI'] = adx_data['Plus_DI']
        indicators['Minus_DI'] = adx_data['Minus_DI']

        # 出来高系（Volumeが利用可能な場合）
        if 'Volume' in data.columns and data['Volume'].notna().any():
            indicators['OBV'] = calculate_obv(data['Close'], data['Volume'])

        # モメンタム系
        indicators['Momentum_10'] = calculate_momentum(data['Close'])
        indicators['ROC_10'] = calculate_roc(data['Close'])

        # 価格情報も追加（参照用）
        indicators['Price'] = data['Close']
        indicators['Volume'] = data['Volume'] if 'Volume' in data.columns else np.nan

        print(f"Calculated {len(indicators.columns)} technical indicators")
        return indicators

    except Exception as e:
        print(f"Error calculating indicators for {symbol}: {e}")
        return None

def save_indicators(symbol, indicators):
    """テクニカル指標をCSVファイルに保存"""
    if indicators is None or indicators.empty:
        print(f"No indicators to save for {symbol}")
        return

    # technical_indicators フォルダを作成
    os.makedirs("technical_indicators", exist_ok=True)

    # ファイル名を作成
    filename = f"technical_indicators/{symbol.replace('.', '_')}_indicators.csv"

    try:
        # CSVファイルに保存
        indicators.to_csv(filename)
        print(f"Technical indicators saved: {filename}")

        # 統計情報を表示
        print(f"  - Data range: {indicators.index[0].date()} to {indicators.index[-1].date()}")
        print(f"  - Total records: {len(indicators)}")
        print(f"  - Indicators calculated: {len(indicators.columns)}")

    except Exception as e:
        print(f"Error saving indicators for {symbol}: {e}")

def generate_indicators_summary(symbols):
    """指標計算の全体サマリーを生成"""
    print("\n" + "="*80)
    print("TECHNICAL INDICATORS CALCULATION SUMMARY")
    print("="*80)
    print(f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    summary_content = []
    summary_content.append("=== Technical Indicators Calculation Summary ===")
    summary_content.append(f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_content.append("")

    total_indicators = 0
    successful_symbols = 0

    for symbol in symbols:
        indicators_file = f"technical_indicators/{symbol.replace('.', '_')}_indicators.csv"

        if os.path.exists(indicators_file):
            try:
                indicators = pd.read_csv(indicators_file, index_col=0, parse_dates=True)
                successful_symbols += 1
                total_indicators += len(indicators.columns)

                summary_content.append(f"Symbol: {symbol}")
                summary_content.append(f"  - File: {indicators_file}")
                summary_content.append(f"  - Data range: {indicators.index[0].date()} to {indicators.index[-1].date()}")
                summary_content.append(f"  - Records: {len(indicators)}")
                summary_content.append(f"  - Indicators: {len(indicators.columns)}")
                summary_content.append("")

                print(f"{symbol}: ✓ {len(indicators)} records, {len(indicators.columns)} indicators")

            except Exception as e:
                summary_content.append(f"Symbol: {symbol}")
                summary_content.append(f"  - Error reading file: {e}")
                summary_content.append("")
                print(f"{symbol}: ✗ Error reading file")
        else:
            summary_content.append(f"Symbol: {symbol}")
            summary_content.append(f"  - File not found: {indicators_file}")
            summary_content.append("")
            print(f"{symbol}: ✗ File not found")

    # サマリー統計
    summary_content.append("=== Overall Statistics ===")
    summary_content.append(f"Total symbols processed: {len(symbols)}")
    summary_content.append(f"Successful calculations: {successful_symbols}")
    summary_content.append(f"Average indicators per symbol: {total_indicators / max(successful_symbols, 1):.1f}")

    print(f"\nOverall: {successful_symbols}/{len(symbols)} symbols processed successfully")

    # サマリーファイルに保存
    os.makedirs("technical_indicators", exist_ok=True)
    with open("technical_indicators/calculation_summary.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_content))

    print("Summary saved: technical_indicators/calculation_summary.txt")

def main():
    """メイン実行関数"""
    # 指標格納先フォルダがなければ作成
    os.makedirs("technical_indicators", exist_ok=True)

    print("Starting Technical Indicators Calculation...")
    print("="*60)

    # 銘柄リストを読み込み
    symbols = load_symbols_from_file("symbols.txt")

    if not symbols:
        print("No symbols found. Please check symbols.txt file.")
        return

    print(f"Processing {len(symbols)} symbols...")

    # 各銘柄についてテクニカル指標を計算
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")

        # 指標計算
        indicators = calculate_all_indicators(symbol)

        # 結果保存
        if indicators is not None:
            save_indicators(symbol, indicators)
        else:
            print(f"Failed to calculate indicators for {symbol}")

    # 全体サマリー生成
    generate_indicators_summary(symbols)

    print("\n" + "="*60)
    print("Technical Indicators Calculation Completed!")
    print("="*60)
    print("\nGenerated files:")
    print("- technical_indicators/{symbol}_indicators.csv: Individual indicator files")
    print("- technical_indicators/calculation_summary.txt: Overall summary")
    print("\nNext step: Run stock_trading_dqn_csv.py for DQN training")

if __name__ == "__main__":
    main()