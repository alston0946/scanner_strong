# -*- coding: utf-8 -*-

import os
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import pandas as pd
import numpy as np


# =========================
# 清理代理
# =========================
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)


# =========================
# 基础路径
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CODE_FILE = DATA_DIR / "a_share_codes_for_akshare.csv"
ST_FILE = DATA_DIR / "st_stocks.csv"


# =========================
# 批次参数（由 GitHub Actions 传入）
# BATCH_TOTAL=2
# BATCH_INDEX=0 或 1
# =========================
BATCH_TOTAL = int(os.getenv("BATCH_TOTAL", "1"))
BATCH_INDEX = int(os.getenv("BATCH_INDEX", "0"))

if BATCH_INDEX < 0 or BATCH_INDEX >= BATCH_TOTAL:
    raise ValueError(f"BATCH_INDEX 非法: {BATCH_INDEX}, BATCH_TOTAL={BATCH_TOTAL}")

BATCH_TAG = f"batch_{BATCH_INDEX}_of_{BATCH_TOTAL}"

OUTPUT_FILE = OUTPUT_DIR / f"strong_surge_pullback_final_{BATCH_TAG}.csv"
FAILED_FILE = OUTPUT_DIR / f"strong_surge_pullback_final_failed_{BATCH_TAG}.csv"
LOG_FILE = OUTPUT_DIR / f"run_log_{BATCH_TAG}.txt"
SUMMARY_FILE = OUTPUT_DIR / f"summary_{BATCH_TAG}.txt"


# =========================
# 参数区
# =========================
START_DATE = "20241001"

TARGET_DATE_ENV = os.getenv("TARGET_DATE", "").strip()
if TARGET_DATE_ENV:
    TARGET_DATE = pd.Timestamp(TARGET_DATE_ENV)
else:
    TARGET_DATE = pd.Timestamp.today().normalize()

END_DATE = TARGET_DATE.strftime("%Y%m%d")

# 每个 batch 内部的线程数
MAX_WORKERS = 1

# 适当降低 sleep，减少总耗时
SLEEP_MIN = 0.05
SLEEP_MAX = 0.15

# 调试时可设 100；全量时改成 None
TEST_LIMIT = None


# =========================
# 选股参数
# =========================
MA_SHORT = 20
MA_MID = 30
MA_LONG = 60

MIN_PULLBACK_DAYS = 2
MAX_PULLBACK_DAYS = 7

MIN_PULLBACK_PCT = 0.05
MAX_PULLBACK_PCT = 0.12

MAX_BREAK_MA20_DAYS = 1

IMPULSE_LOOKBACK = 25
MIN_IMPULSE_RISE_PCT = 0.08
MIN_IMPULSE_DAYS = 4
MAX_IMPULSE_DRAWDOWN_PCT = 0.05

PEAK_NEW_HIGH_LOOKBACK = 50
MAX_PULLBACK_REBOUND_PCT = 0.04


# =========================
# 日志函数
# =========================
def reset_log():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")


def log(*args):
    msg = " ".join(str(x) for x in args)
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# =========================
# 工具函数
# =========================
def safe_to_datetime(series):
    return pd.to_datetime(series, errors="coerce")


def pct_change(a, b):
    if a is None or b is None or a == 0:
        return np.nan
    return b / a - 1.0


def standardize_hist_tx(df):
    rename_map = {}

    for col in df.columns:
        c = str(col).strip().lower()
        if c in ["date", "日期"]:
            rename_map[col] = "date"
        elif c in ["open", "开盘"]:
            rename_map[col] = "open"
        elif c in ["close", "收盘"]:
            rename_map[col] = "close"
        elif c in ["high", "最高"]:
            rename_map[col] = "high"
        elif c in ["low", "最低"]:
            rename_map[col] = "low"
        elif c in ["volume", "vol", "成交量"]:
            rename_map[col] = "volume"
        elif c in ["amount", "成交额"]:
            rename_map[col] = "amount"

    out = df.rename(columns=rename_map).copy()

    if "date" not in out.columns or "close" not in out.columns:
        raise ValueError(f"hist_tx 缺少 date/close 列, 实际列: {list(df.columns)}")

    out["date"] = safe_to_datetime(out["date"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    for c in ["open", "high", "low", "volume", "amount"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["date", "close"]).copy()
    out = out.sort_values("date").reset_index(drop=True)
    return out


def calc_ma(df, windows=(20, 30, 60)):
    out = df.copy()
    for w in windows:
        out[f"ma{w}"] = out["close"].rolling(w).mean()
    return out


def load_st_symbols():
    if not ST_FILE.exists():
        log(f"[警告] ST 文件不存在：{ST_FILE}")
        return set()

    try:
        df = pd.read_csv(ST_FILE)
    except pd.errors.EmptyDataError:
        log(f"[警告] ST 文件为空：{ST_FILE}")
        return set()

    if df.empty:
        log(f"[警告] ST 文件为空表：{ST_FILE}")
        return set()

    if "ticker" not in df.columns:
        raise ValueError(f"ST 文件里没有 ticker 列，实际列名: {list(df.columns)}")

    s = df["ticker"].astype(str).str.strip().str.zfill(6)
    st_codes = set(s.dropna().tolist())
    return st_codes


def load_universe_from_csv():
    df = pd.read_csv(CODE_FILE)

    if "symbol" not in df.columns:
        raise ValueError("代码文件里没有 symbol 列")

    out = df[["symbol"]].copy()
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out = out[out["symbol"].str.startswith(("sh", "sz"))].copy()
    out = out.drop_duplicates("symbol").reset_index(drop=True)
    out["name"] = out["symbol"]

    st_codes = load_st_symbols()
    out["code6"] = out["symbol"].str[-6:]
    out = out[~out["code6"].isin(st_codes)].copy()

    if TEST_LIMIT is not None:
        out = out.head(TEST_LIMIT).copy()

    out = out[["symbol", "name"]].reset_index(drop=True)
    return out


def split_universe_for_batch(universe: pd.DataFrame, batch_total: int, batch_index: int) -> pd.DataFrame:
    """
    把 universe 均匀切成 batch_total 份，当前 job 只跑其中一份
    """
    n = len(universe)
    if n == 0:
        return universe.copy()

    start = n * batch_index // batch_total
    end = n * (batch_index + 1) // batch_total
    return universe.iloc[start:end].reset_index(drop=True)


def fetch_hist_tx_with_retry(symbol, start_date, end_date, max_retry=3):
    last_err = None
    for attempt in range(max_retry):
        try:
            df = ak.stock_zh_a_hist_tx(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=""
            )
            if df is not None and not df.empty:
                return df
            last_err = "empty dataframe"
        except Exception as e:
            last_err = str(e)

        time.sleep(1.2 + attempt * 1.2 + random.uniform(0.2, 0.6))

    raise RuntimeError(str(last_err))


def find_recent_peak_and_pullback(df):
    if len(df) < MAX_PULLBACK_DAYS + 2:
        return None

    last_idx = len(df) - 1
    start_idx = max(0, last_idx - MAX_PULLBACK_DAYS)

    recent = df.iloc[start_idx:last_idx].copy()
    if recent.empty:
        return None

    peak_idx = recent["close"].idxmax()
    pullback_days = last_idx - peak_idx

    if pullback_days < MIN_PULLBACK_DAYS or pullback_days > MAX_PULLBACK_DAYS:
        return None

    peak_close = df.loc[peak_idx, "close"]
    last_close = df.loc[last_idx, "close"]

    if peak_close <= 0 or last_close <= 0:
        return None

    pullback_pct = peak_close / last_close - 1.0
    if pullback_pct < 0:
        return None

    return peak_idx, pullback_days, pullback_pct


def find_impulse_start(df, peak_idx):
    left = max(0, peak_idx - IMPULSE_LOOKBACK)
    right = peak_idx - 1

    if right <= left:
        return None

    seg = df.iloc[left:right + 1].copy()
    if seg.empty:
        return None

    start_idx = seg["close"].idxmin()
    if peak_idx - start_idx < MIN_IMPULSE_DAYS:
        return None

    return start_idx


def is_peak_new_high(df, peak_idx, lookback=PEAK_NEW_HIGH_LOOKBACK):
    if peak_idx <= 0:
        return False

    left = max(0, peak_idx - lookback)
    prev_seg = df.iloc[left:peak_idx].copy()
    if prev_seg.empty:
        return False

    peak_close = df.loc[peak_idx, "close"]
    prev_max_close = prev_seg["close"].max()
    return peak_close > prev_max_close


def calc_max_drawdown_in_segment(df, start_idx, end_idx):
    seg = df.iloc[start_idx:end_idx + 1].copy()
    if seg.empty or len(seg) < 2:
        return np.nan

    closes = seg["close"].values
    running_peak = closes[0]
    max_dd = 0.0

    for c in closes:
        if c > running_peak:
            running_peak = c
        dd = (running_peak - c) / running_peak
        if dd > max_dd:
            max_dd = dd

    return max_dd


def calc_max_rebound_in_segment(df, start_idx, end_idx):
    seg = df.iloc[start_idx:end_idx + 1].copy()
    if seg.empty or len(seg) < 2:
        return np.nan

    closes = seg["close"].values
    running_low = closes[0]
    max_rebound = 0.0

    for c in closes:
        if c < running_low:
            running_low = c
        rebound = (c - running_low) / running_low
        if rebound > max_rebound:
            max_rebound = rebound

    return max_rebound


def evaluate_one_stock(symbol, name):
    try:
        df = fetch_hist_tx_with_retry(symbol, START_DATE, END_DATE, max_retry=3)
        df = standardize_hist_tx(df)
        df = df[df["date"] <= TARGET_DATE].copy()
        df = df.sort_values("date").reset_index(drop=True)

        if df.empty:
            return {"symbol": symbol, "name": name, "error": "no data before target"}

        if len(df) < 100:
            return {"symbol": symbol, "name": name, "error": "not enough bars"}

        df = calc_ma(df, windows=(MA_SHORT, MA_MID, MA_LONG))

        last = df.iloc[-1]
        prev = df.iloc[-2]

        if pd.isna(last[f"ma{MA_LONG}"]) or pd.isna(prev[f"ma{MA_LONG}"]):
            return {"symbol": symbol, "name": name, "error": "ma60 nan"}

        cond_ma = (
            last["close"] > last[f"ma{MA_LONG}"]
            and last[f"ma{MA_LONG}"] > prev[f"ma{MA_LONG}"]
            and last[f"ma{MA_SHORT}"] > prev[f"ma{MA_SHORT}"]
            and last[f"ma{MA_MID}"] > prev[f"ma{MA_MID}"]
        )
        if not cond_ma:
            return {"symbol": symbol, "name": name, "error": "ma condition fail"}

        peak_info = find_recent_peak_and_pullback(df)
        if peak_info is None:
            return {"symbol": symbol, "name": name, "error": "no recent pullback"}

        peak_idx, pullback_days, pullback_pct = peak_info

        if pullback_pct < MIN_PULLBACK_PCT:
            return {"symbol": symbol, "name": name, "error": "pullback too small"}

        if pullback_pct > MAX_PULLBACK_PCT:
            return {"symbol": symbol, "name": name, "error": "pullback too deep"}

        if not is_peak_new_high(df, peak_idx, PEAK_NEW_HIGH_LOOKBACK):
            return {"symbol": symbol, "name": name, "error": "peak not 50d new high"}

        pullback_df = df.iloc[peak_idx + 1:].copy()
        if pullback_df.empty:
            return {"symbol": symbol, "name": name, "error": "pullback empty"}

        break_days = (pullback_df["close"] < pullback_df[f"ma{MA_SHORT}"]).sum()
        if break_days > MAX_BREAK_MA20_DAYS:
            return {"symbol": symbol, "name": name, "error": "break ma20 in pullback"}

        pullback_max_rebound = calc_max_rebound_in_segment(df, peak_idx, len(df) - 1)
        if pd.isna(pullback_max_rebound):
            return {"symbol": symbol, "name": name, "error": "pullback rebound nan"}

        if pullback_max_rebound >= MAX_PULLBACK_REBOUND_PCT:
            return {"symbol": symbol, "name": name, "error": "pullback too choppy"}

        impulse_start_idx = find_impulse_start(df, peak_idx)
        if impulse_start_idx is None:
            return {"symbol": symbol, "name": name, "error": "no impulse start"}

        impulse_start_close = df.loc[impulse_start_idx, "close"]
        peak_close = df.loc[peak_idx, "close"]
        impulse_rise_pct = pct_change(impulse_start_close, peak_close)

        if pd.isna(impulse_rise_pct) or impulse_rise_pct < MIN_IMPULSE_RISE_PCT:
            return {"symbol": symbol, "name": name, "error": "impulse rise too small"}

        impulse_max_dd = calc_max_drawdown_in_segment(df, impulse_start_idx, peak_idx)
        if pd.isna(impulse_max_dd):
            return {"symbol": symbol, "name": name, "error": "impulse drawdown nan"}

        if impulse_max_dd >= MAX_IMPULSE_DRAWDOWN_PCT:
            return {"symbol": symbol, "name": name, "error": "impulse drawdown too large"}

        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

        return {
            "symbol": symbol,
            "name": name,
            "date": df.iloc[-1]["date"].strftime("%Y-%m-%d"),

            "close": round(last["close"], 4),
            "ma20": round(last[f"ma{MA_SHORT}"], 4),
            "ma30": round(last[f"ma{MA_MID}"], 4),
            "ma60": round(last[f"ma{MA_LONG}"], 4),

            "peak_date": df.loc[peak_idx, "date"].strftime("%Y-%m-%d"),
            "peak_close": round(peak_close, 4),
            "pullback_days": int(pullback_days),
            "pullback_pct": round(pullback_pct, 6),
            "break_ma20_days": int(break_days),
            "pullback_max_rebound_pct": round(pullback_max_rebound, 6),

            "impulse_start_date": df.loc[impulse_start_idx, "date"].strftime("%Y-%m-%d"),
            "impulse_start_close": round(impulse_start_close, 4),
            "impulse_rise_pct": round(impulse_rise_pct, 6),
            "impulse_max_drawdown_pct": round(impulse_max_dd, 6),
        }

    except Exception as e:
        return {"symbol": symbol, "name": name, "error": str(e)}


def main():
    reset_log()

    log("========== 开始运行 ==========")
    log("BATCH_TOTAL:", BATCH_TOTAL)
    log("BATCH_INDEX:", BATCH_INDEX)
    log("BATCH_TAG:", BATCH_TAG)
    log("股票池文件:", CODE_FILE)
    log("ST 文件:", ST_FILE)
    log("输出目录:", OUTPUT_DIR)
    log("TARGET_DATE:", TARGET_DATE.strftime("%Y-%m-%d"))
    log("END_DATE:", END_DATE)
    log("MAX_WORKERS:", MAX_WORKERS)

    log("\n1) 读取本地全市场股票列表（已过滤ST）...")
    universe_all = load_universe_from_csv()
    log("全量股票总数:", len(universe_all))

    universe = split_universe_for_batch(universe_all, BATCH_TOTAL, BATCH_INDEX)
    log("当前批次股票数:", len(universe))
    log(universe.head())

    matched = []
    failed = []
    error_counter = {}

    log("\n2) 开始当前批次扫描...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(
                evaluate_one_stock,
                row["symbol"],
                row["name"],
            ): row["symbol"]
            for _, row in universe.iterrows()
        }

        total = len(future_map)
        for i, future in enumerate(as_completed(future_map), 1):
            result = future.result()

            if isinstance(result, dict) and "error" not in result:
                matched.append(result)
            else:
                failed.append(result)
                err = result.get("error", "unknown")
                error_counter[err] = error_counter.get(err, 0) + 1

            if i % 100 == 0 or i == total:
                log(f"进度: {i}/{total} | 命中: {len(matched)} | 未命中/失败: {len(failed)}")

    pd.DataFrame(failed).to_csv(FAILED_FILE, index=False, encoding="utf-8-sig")
    log("\n失败/未命中记录已保存:", FAILED_FILE)

    log("\n失败原因统计：")
    for k, v in sorted(error_counter.items(), key=lambda x: -x[1]):
        log(k, v)

    if matched:
        out_df = pd.DataFrame(matched).sort_values(
            by=["impulse_rise_pct", "pullback_pct"],
            ascending=[False, True]
        ).reset_index(drop=True)

        out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        log("\n命中结果已保存:", OUTPUT_FILE)

        log("\n结果预览:")
        log(out_df.head(30))
    else:
        log("\n没有筛到符合条件的股票。")
        pd.DataFrame(columns=[
            "symbol", "name", "date", "close", "ma20", "ma30", "ma60",
            "peak_date", "peak_close", "pullback_days", "pullback_pct",
            "break_ma20_days", "pullback_max_rebound_pct",
            "impulse_start_date", "impulse_start_close",
            "impulse_rise_pct", "impulse_max_drawdown_pct"
        ]).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(f"BATCH_TOTAL: {BATCH_TOTAL}\n")
        f.write(f"BATCH_INDEX: {BATCH_INDEX}\n")
        f.write(f"BATCH_TAG: {BATCH_TAG}\n")
        f.write(f"TARGET_DATE: {TARGET_DATE.strftime('%Y-%m-%d')}\n")
        f.write(f"全量股票总数: {len(universe_all)}\n")
        f.write(f"当前批次股票数: {len(universe)}\n")
        f.write(f"命中数量: {len(matched)}\n")
        f.write(f"失败/未命中数量: {len(failed)}\n")
        f.write(f"结果文件: {OUTPUT_FILE.name}\n")
        f.write(f"失败文件: {FAILED_FILE.name}\n")

    log("\n摘要文件已保存:", SUMMARY_FILE)
    log("========== 运行结束 ==========")


if __name__ == "__main__":
    main()
