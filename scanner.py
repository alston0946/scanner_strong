# scanner.py
from pathlib import Path
import traceback
import time
import pandas as pd

# =========================
# 基础路径
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

CODE_FILE = DATA_DIR / "a_share_codes_for_akshare.csv"
ST_FILE = DATA_DIR / "st_stocks.csv"

OUTPUT_DIR.mkdir(exist_ok=True)

RESULT_FILE = OUTPUT_DIR / "result.csv"
FAILED_FILE = OUTPUT_DIR / "failed_symbols.csv"
LOG_FILE = OUTPUT_DIR / "run_log.txt"
SUMMARY_FILE = OUTPUT_DIR / "summary.txt"

# =========================
# 测试数量
# 第一次在 GitHub 上先只跑 100 只
# =========================
TEST_LIMIT = 100

# =========================
# 日志函数
# =========================
def log(msg):
    text = str(msg)
    print(text, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# =========================
# 读取 ST 股票列表
# 兼容常见列名：ticker / symbol / secID
# 最终统一转成 6 位代码集合
# =========================
def load_st_symbols():
    if not ST_FILE.exists():
        log(f"[警告] ST 文件不存在：{ST_FILE}")
        return set()

    df = pd.read_csv(ST_FILE)
    if df.empty:
        log("[警告] ST 文件为空")
        return set()

    possible_cols = ["ticker", "symbol", "secID", "code"]
    use_col = None
    for col in possible_cols:
        if col in df.columns:
            use_col = col
            break

    if use_col is None:
        raise ValueError(
            f"ST 文件中未找到可识别列，当前列为：{list(df.columns)}"
        )

    s = df[use_col].astype(str).str.strip()

    # secID 例如 000004.XSHE
    # symbol 例如 sz000004 / sh600000
    # ticker 例如 000004
    code6 = (
        s.str.extract(r"(\d{6})", expand=False)
         .dropna()
         .astype(str)
         .str.zfill(6)
    )

    st_codes = set(code6.tolist())
    log(f"ST 股票数量：{len(st_codes)}")
    return st_codes

# =========================
# 从股票池 CSV 里读取 universe
# 要求至少有 symbol 列，例如：
# sh600000 / sz000001
# =========================
def load_universe_from_csv():
    if not CODE_FILE.exists():
        raise FileNotFoundError(f"股票池文件不存在：{CODE_FILE}")

    df = pd.read_csv(CODE_FILE)

    if "symbol" not in df.columns:
        raise ValueError("代码文件里没有 symbol 列")

    out = df[["symbol"]].copy()
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out = out[out["symbol"].str.startswith(("sh", "sz"))].copy()
    out = out.drop_duplicates("symbol").reset_index(drop=True)
    out["name"] = out["symbol"]

    # 过滤 ST 股票
    st_codes = load_st_symbols()
    out["code6"] = out["symbol"].str[-6:]
    before_cnt = len(out)
    out = out[~out["code6"].isin(st_codes)].copy()
    after_cnt = len(out)

    log(f"原始股票池数量：{before_cnt}")
    log(f"过滤 ST 后数量：{after_cnt}")

    if TEST_LIMIT is not None:
        out = out.head(TEST_LIMIT).copy()
        log(f"TEST_LIMIT 已启用，仅取前 {TEST_LIMIT} 只股票")

    return out[["symbol", "name"]]

# =========================
# 你的单只股票扫描逻辑
# 这里先给一个占位版本，确保 GitHub 流程先跑通
#
# 后面你把自己的选股逻辑替换进来即可
# 返回：
#   dict  -> 选中，写入结果表
#   None  -> 未选中
# =========================
def scan_one_symbol(symbol, name):
    # =========
    # 这里先放一个“演示版占位逻辑”
    # 每 10 只选 1 只，确保 result.csv 有内容
    # 你后面把这里替换成真正策略即可
    # =========
    code6 = symbol[-6:]
    if code6.endswith("0"):
        return {
            "symbol": symbol,
            "name": name,
            "status": "selected",
            "remark": "demo_selected_for_github_test"
        }
    return None

# =========================
# 主流程
# =========================
def main():
    t0 = time.time()

    # 清空旧日志
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("========== 开始运行 ==========")
    log(f"当前脚本路径：{BASE_DIR}")
    log(f"股票池文件：{CODE_FILE}")
    log(f"ST 文件：{ST_FILE}")
    log(f"输出目录：{OUTPUT_DIR}")
    log(f"TEST_LIMIT：{TEST_LIMIT}")

    universe = load_universe_from_csv()
    total = len(universe)
    log(f"本次待扫描股票数：{total}")

    results = []
    failed = []

    for i, row in universe.iterrows():
        symbol = row["symbol"]
        name = row["name"]

        if (i + 1) % 10 == 0 or i == 0:
            log(f"[进度] {i + 1}/{total} 处理中：{symbol}")

        try:
            ret = scan_one_symbol(symbol, name)
            if ret is not None:
                results.append(ret)
        except Exception as e:
            failed.append({
                "symbol": symbol,
                "name": name,
                "error": repr(e)
            })
            log(f"[失败] {symbol} -> {repr(e)}")

    result_df = pd.DataFrame(results)
    failed_df = pd.DataFrame(failed)

    if result_df.empty:
        result_df = pd.DataFrame(columns=["symbol", "name", "status", "remark"])
    if failed_df.empty:
        failed_df = pd.DataFrame(columns=["symbol", "name", "error"])

    result_df.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
    failed_df.to_csv(FAILED_FILE, index=False, encoding="utf-8-sig")

    elapsed = time.time() - t0
    summary = [
        "========== 运行摘要 ==========",
        f"股票池总数：{total}",
        f"选中数量：{len(result_df)}",
        f"失败数量：{len(failed_df)}",
        f"耗时秒数：{elapsed:.2f}",
        f"结果文件：{RESULT_FILE.name}",
        f"失败文件：{FAILED_FILE.name}",
        f"日志文件：{LOG_FILE.name}",
    ]
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    for line in summary:
        log(line)

    log("========== 运行结束 ==========")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n========== 程序异常退出 ==========\n")
            f.write(traceback.format_exc())
            f.write("\n")
        raise
