"""A 股多因子排名（动量 + 质量 + 估值，等权）。

一次性研究报告，非可回测策略：复用声明式 factor_combo（FactorStrategy）+
ScreeningEngine。默认限定消费大类细分行业（CONSUMER_SECTORS），可通过
命令行参数换成任意行业或全市场。

用法:
    uv run python3 scripts/consumer_rank.py                         # 默认消费大类
    uv run python3 scripts/consumer_rank.py --sectors 半导体,元器件,IT设备
    uv run python3 scripts/consumer_rank.py --all                   # 不分行业，全市场
    uv run python3 scripts/consumer_rank.py --top-n 10 --date 2026-05-15
    uv run python3 scripts/consumer_rank.py --list-sectors          # 查看可选行业名
"""

import argparse
from datetime import date

from sqlalchemy import text

from trendspec.config.settings import get_settings
from trendspec.data.markets import Market
from trendspec.data.parquet_loader import bars, scan_parquet
from trendspec.engine.base_engine import EngineConfig
from trendspec.engine.screening_engine import ScreeningEngine
from trendspec.ingest.mariadb_client import create_engine_from_settings
from trendspec.combo import FactorSpec
from trendspec.strategy.base import get_strategy
import trendspec.strategy.examples  # noqa: F401 — triggers @register_strategy
import trendspec.strategy.factor_strategy  # noqa: F401 — registers "factor_combo"


def _list_sectors() -> list[str]:
    df = scan_parquet(None, Market.CN, "sectors").collect()
    return sorted(df["sector"].unique().to_list())


def _company_names(tickers: list[str]) -> dict[str, str]:
    """群辉 stocks 表查中文名，best-effort（查不到就留空，不影响排名输出）。"""
    if not tickers:
        return {}
    try:
        engine = create_engine_from_settings(get_settings().db)
        placeholders = ", ".join(f":t{i}" for i in range(len(tickers)))
        sql = text(f"SELECT ticker, name FROM stocks WHERE ticker IN ({placeholders})")
        params = {f"t{i}": t for i, t in enumerate(tickers)}
        with engine.connect() as conn:
            return {row[0]: row[1] for row in conn.execute(sql, params)}
    except Exception:
        return {}


def _latest_available_date(market: Market) -> date:
    """CN 日线摄入可能滞后于日历日 — 用数据里实际存在的最新交易日，而不是
    today()，否则 screening 当天没有任何行情行，排名会静默返回 0 支。
    """
    recent = bars(market=market, start_date=date(2020, 1, 1), end_date=date.today())
    if recent.is_empty():
        raise RuntimeError(f"{market} 日线数据为空，请先跑 ingest daily")
    return recent["date"].max()

# CN sectors 数据集实际用的是细分行业中文名（同花顺板块类，非申万一级代码）。
# 消费相关子行业：食品饮料/家电家居/纺织服饰/商贸零售/社服休闲/日化。
CONSUMER_SECTORS = [
    "白酒", "啤酒", "红黄酒", "软饮料", "食品", "乳制品",
    "家用电器", "家居用品",
    "服饰",
    "百货", "商品城",
    "酒店餐饮", "旅游景点", "旅游服务", "文教休闲",
    "日用化工",
]

TOP_N = 50


def main(
    target_date: date | None = None,
    sectors: list[str] | None = CONSUMER_SECTORS,
    top_n: int = TOP_N,
) -> None:
    target_date = target_date or _latest_available_date(Market.CN)

    spec = FactorSpec(
        market="cn",
        factors=[
            {"name": "price_momentum", "params": {"period": 60}, "direction": "high", "weight": 1.0},
            {"name": "fund_roe", "direction": "high", "weight": 1.0},
            {"name": "fund_pe_ttm", "direction": "low", "weight": 1.0},
        ],
        top_k=top_n,
        rebalance=1,
        sector_filter=sectors,
        rationale="A股：动量+质量(ROE)+估值(PE_ttm)，等权组合排名",
    )

    config = EngineConfig(
        market=Market.CN,
        start_date=target_date,
        end_date=target_date,
    )
    engine = ScreeningEngine(config)
    strategy_class = get_strategy("factor_combo")
    result = engine.run(strategy_class, params={"spec": spec.model_dump()})

    buys = sorted(
        (s for s in result.signals if s.is_buy()),
        key=lambda s: s.trigger_value if s.trigger_value is not None else float("-inf"),
        reverse=True,
    )

    print(f"=== A股多因子排名 {target_date.isoformat()} ===")
    print(f"因子: 动量(60日,权重1) + ROE(权重1) + PE_ttm低优先(权重1)，等权 z-score 组合")
    print(f"候选池行业: {sectors if sectors else '全市场（不限行业）'}")
    print(f"结果: {len(buys)} 支\n")

    if not buys:
        print("无候选 — 检查 fundamentals/valuation 数据是否已摄入到该日期。")
        return

    names = _company_names([s.ticker for s in buys])

    print(f"{'排名':<4}{'代码':<14}{'名称':<10}{'combo_score':>12}")
    for rank, sig in enumerate(buys, start=1):
        name = names.get(sig.ticker, "")
        print(f"{rank:<4}{sig.ticker:<14}{name:<10}{sig.trigger_value:>12.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sectors", type=str, default=None,
                        help="逗号分隔的行业名列表，覆盖默认消费大类，如: 半导体,元器件,IT设备")
    parser.add_argument("--all", action="store_true",
                        help="不限行业，全市场排名（忽略 --sectors）")
    parser.add_argument("--top-n", type=int, default=TOP_N, help=f"排名前 N 支（默认 {TOP_N}）")
    parser.add_argument("--date", type=str, default=None,
                        help="筛选日期 YYYY-MM-DD（默认数据里最新可用交易日）")
    parser.add_argument("--list-sectors", action="store_true",
                        help="打印所有可选行业名后退出，不跑排名")
    args = parser.parse_args()

    if args.list_sectors:
        for s in _list_sectors():
            print(s)
        raise SystemExit(0)

    if args.all:
        chosen_sectors = None
    elif args.sectors:
        chosen_sectors = [s.strip() for s in args.sectors.split(",") if s.strip()]
    else:
        chosen_sectors = CONSUMER_SECTORS

    chosen_date = date.fromisoformat(args.date) if args.date else None

    main(target_date=chosen_date, sectors=chosen_sectors, top_n=args.top_n)
