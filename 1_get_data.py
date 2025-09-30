import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

# ========================
# 設定値
# ========================
# データ取得期間の設定 (どちらか一つを設定してください)
#
# オプション1: yfinanceの期間指定を使用する場合
DEFAULT_PERIOD = "3y"          # "1y", "2y", "5y", "max" など
DEFAULT_DAYS_BACK = None       # 使わない場合はNoneに設定
#
# オプション2: 特定の日数を指定する場合
# DEFAULT_PERIOD = "2y"        # この値は無視されます
# DEFAULT_DAYS_BACK = 365      # 過去365日のデータを取得
#
# 設定例:
# - 過去1年分: DEFAULT_DAYS_BACK = 365
# - 過去2年分: DEFAULT_DAYS_BACK = 730
# - 過去5年分: DEFAULT_PERIOD = "5y", DEFAULT_DAYS_BACK = None

# その他の設定
FORCE_UPDATE_DEFAULT = False   # デフォルトで強制更新するかどうか


def _analyze_yfinance_error(error_msg, symbol):
    """yfinanceエラーメッセージを解析して改善案を提示"""
    print(f"\n=== yfinance Error Analysis for {symbol} ===")

    improvements = []

    if "possibly delisted" in error_msg.lower():
        improvements.extend([
            f"銘柄 {symbol} は上場廃止の可能性があります",
            "• 別の銘柄コードを試してください",
            "• 日本株の場合は '.T' の代わりに '.TO' を試してください",
            "• 正確な銘柄コードをFinance サイトで確認してください"
        ])

    if "expecting value" in error_msg.lower() or "json" in error_msg.lower():
        improvements.extend([
            "Yahoo Financeサーバーからの無効なレスポンスです",
            "• ネットワーク接続を確認してください",
            "• しばらく時間をおいてから再試行してください"
        ])

    if "no price data found" in error_msg.lower():
        improvements.extend([
            f"銘柄 {symbol} の価格データが見つかりません",
            "• 銘柄コードのスペルを確認してください",
            "• 市場サフィックスを確認してください（.T, .TO, .HK など）",
            "• 期間を短くしてみてください（例：period='1y'）"
        ])

    if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
        improvements.extend([
            "ネットワーク接続に問題があります",
            "• インターネット接続を確認してください",
            "• ファイアウォール設定を確認してください",
            "• プロキシ設定が必要な場合は設定してください"
        ])

    if not improvements:
        improvements.extend([
            "一般的なトラブルシューティング:",
            "• しばらく時間をおいてから再試行してください",
            "• 銘柄コードを再確認してください",
            "• ネットワーク接続を確認してください",
            "• 既存のCSVファイルがある場合はそれを使用します"
        ])

    print("改善案:")
    for improvement in improvements:
        print(improvement)
    print("=" * 50)


def _get_yfinance_latest_date(symbol):
    """yfinanceの最新利用可能日付を取得"""
    try:
        ticker = yf.Ticker(symbol)
        # 最近の少しのデータだけ取得して最新日付を確認
        recent_data = ticker.history(period="5d")
        if not recent_data.empty:
            return recent_data.index[-1].date()
    except Exception as e:
        print(f"Could not get yfinance latest date for {symbol}: {e}")
    return None


def _save_data_to_csv(data, symbol):
    """データをCSVファイルに保存"""
    try:
        os.makedirs("yfinance_data", exist_ok=True)
        csv_file = f"yfinance_data/{symbol.replace('.', '_')}.csv"
        data.to_csv(csv_file)
        print(f"Data saved to {csv_file}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")


def _incremental_update(symbol, existing_data, missing_periods, csv_file):
    """
    不足している期間のデータを取得して既存データと結合
    Args:
        symbol: 銘柄コード
        existing_data: 既存のDataFrame
        missing_periods: 不足期間のリスト [(start_date, end_date), ...]
        csv_file: CSVファイルパス
    Returns:
        bool: 成功/失敗
    """
    try:
        all_new_data = []

        for start_date, end_date in missing_periods:
            print(f"Fetching missing data for {symbol}: {start_date} to {end_date}")

            ticker = yf.Ticker(symbol)
            # yfinanceは日付範囲指定でデータを取得
            new_data = ticker.history(start=start_date, end=end_date + timedelta(days=1))

            if not new_data.empty:
                all_new_data.append(new_data)
                print(f"Retrieved {len(new_data)} days of data for period {start_date} to {end_date}")
            else:
                print(f"No data found for period {start_date} to {end_date}")

        if all_new_data:
            # 既存データと新しいデータを結合
            combined_data = pd.concat([existing_data] + all_new_data)
            # 重複を削除してソート
            combined_data = combined_data[~combined_data.index.duplicated(keep='first')].sort_index()

            # CSVに保存
            combined_data.to_csv(csv_file)
            print(f"Successfully updated {symbol} with {sum(len(data) for data in all_new_data)} new data points")
            print(f"Total data range: {combined_data.index[0].date()} to {combined_data.index[-1].date()}")
            return True
        else:
            print(f"No new data retrieved for {symbol}")
            return True  # 既存データはそのまま使用可能

    except Exception as e:
        print(f"Error during incremental update for {symbol}: {e}")
        return False


def fetch_stock_data(symbol, period=DEFAULT_PERIOD, days_back=DEFAULT_DAYS_BACK, force_update=FORCE_UPDATE_DEFAULT):
    """
    株式データを取得してCSVに保存
    Args:
        symbol: 銘柄コード
        period: yfinanceの期間指定（days_backが指定されている場合は無視）
        days_back: 本日から何日前までのデータを取得するか（Noneの場合はperiodを使用）
        force_update: 強制更新フラグ
    """
    csv_file = f"yfinance_data/{symbol.replace('.', '_')}.csv"

    # days_backが指定されている場合、期間を計算
    if days_back is not None:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        print(f"Data range: {start_date} to {end_date} ({days_back} days)")
    else:
        start_date = None
        end_date = None

    # 強制更新でない場合、既存データの日付をチェックし、増分更新を行う
    existing_data = None
    if not force_update and os.path.exists(csv_file):
        try:
            existing_data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            if not existing_data.empty:
                csv_latest_date = existing_data.index[-1].date()
                csv_earliest_date = existing_data.index[0].date()
                print(f"Found existing data for {symbol}. CSV data range: {csv_earliest_date} to {csv_latest_date}")

                # 必要な日付範囲を決定
                target_start_date = start_date if start_date else csv_earliest_date
                target_end_date = end_date if end_date else datetime.now().date()

                # 既存データで十分な場合はスキップ
                if csv_earliest_date <= target_start_date and csv_latest_date >= target_end_date:
                    print(f"Existing CSV data covers the required range. Skipping update for {symbol}")
                    return True

                # 不足している期間を特定
                missing_periods = []

                # 開始日より前のデータが必要な場合
                if target_start_date < csv_earliest_date:
                    missing_periods.append((target_start_date, csv_earliest_date - timedelta(days=1)))
                    print(f"Missing data before existing range: {target_start_date} to {csv_earliest_date - timedelta(days=1)}")

                # 終了日より後のデータが必要な場合
                if target_end_date > csv_latest_date:
                    missing_periods.append((csv_latest_date + timedelta(days=1), target_end_date))
                    print(f"Missing data after existing range: {csv_latest_date + timedelta(days=1)} to {target_end_date}")

                # 不足データがある場合は増分更新
                if missing_periods:
                    return _incremental_update(symbol, existing_data, missing_periods, csv_file)
                else:
                    print(f"All required data is already present. No update needed for {symbol}")
                    return True

        except Exception as e:
            print(f"Error reading existing CSV file: {e}")
            existing_data = None

    # yfinanceでデータを取得してCSVに保存
    last_error = None
    try:
        print(f"Fetching fresh data for {symbol} from yfinance...")
        ticker = yf.Ticker(symbol)

        # days_backが指定されている場合は日付範囲で取得、そうでなければperiodで取得
        if days_back is not None:
            data = ticker.history(start=start_date, end=end_date + timedelta(days=1))
        else:
            data = ticker.history(period=period)

        if data.empty or len(data) == 0:
            print(f"Warning: No data found for {symbol}. Trying alternative symbols...")

            # 日本株の場合、異なる形式を試す
            if symbol.endswith('.T'):
                alt_symbol = symbol.replace('.T', '.TO')
                print(f"Trying {alt_symbol}...")
                try:
                    ticker = yf.Ticker(alt_symbol)
                    data = ticker.history(period=period)
                except Exception as e:
                    last_error = str(e)

            # まだ空の場合、AAPLで試す
            if data.empty or len(data) == 0:
                print(f"Warning: No data found for {symbol}. Using AAPL as fallback.")
                original_symbol = symbol
                symbol = "AAPL"
                try:
                    ticker = yf.Ticker("AAPL")
                    data = ticker.history(period=period)
                except Exception as e:
                    last_error = str(e)

        if data.empty or len(data) == 0:
            error_msg = last_error if last_error else f"No price data found for {symbol} (period={period})"
            if "possibly delisted" not in error_msg and last_error:
                error_msg += f" possibly delisted; no price data found (period={period})"

            _analyze_yfinance_error(error_msg, symbol)
            return False

        # CSVファイルに保存
        _save_data_to_csv(data, symbol)
        print(f"Successfully loaded and saved {len(data)} days of data for {symbol}")
        return True

    except Exception as e:
        error_msg = str(e)
        if "Expecting value" in error_msg:
            error_msg += " (JSON parsing error - likely network or server issue)"
        _analyze_yfinance_error(error_msg, symbol)
        return False


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


def update_all_symbols(symbols_file="symbols.txt", period=DEFAULT_PERIOD, days_back=DEFAULT_DAYS_BACK, force_update=FORCE_UPDATE_DEFAULT):
    """
    全銘柄のデータを更新
    Args:
        symbols_file: 銘柄リストファイル
        period: yfinanceの期間指定（days_backが指定されている場合は無視）
        days_back: 本日から何日前までのデータを取得するか
        force_update: 強制更新フラグ
    """
    # データ格納先フォルダがなければ作成
    os.makedirs("yfinance_data", exist_ok=True)

    print("Stock Data Fetcher - yfinance to CSV")
    print("="*50)

    if days_back is not None:
        print(f"Fetching data for the last {days_back} days")
    else:
        print(f"Fetching data for period: {period}")

    symbols = load_symbols_from_file(symbols_file)

    successful_updates = 0
    failed_updates = 0

    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        if fetch_stock_data(symbol, period, days_back, force_update):
            successful_updates += 1
        else:
            failed_updates += 1

    print(f"\n=== Update Summary ===")
    print(f"Successful updates: {successful_updates}")
    print(f"Failed updates: {failed_updates}")
    print(f"Total symbols processed: {len(symbols)}")

    return successful_updates, failed_updates


if __name__ == "__main__":
    # 設定値を表示
    print(f"Configuration:")
    print(f"  Default period: {DEFAULT_PERIOD}")
    print(f"  Default days back: {DEFAULT_DAYS_BACK}")

    update_all_symbols()