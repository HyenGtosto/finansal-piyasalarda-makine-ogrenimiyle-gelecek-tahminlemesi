"""Streamlit dashboard for the financial forecasting thesis.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Finansal Tahminleme",
    page_icon="📈",
    layout="wide",
)

st.title("Finansal Piyasalarda Makine Öğrenimiyle Gelecek Tahminlemesi")
st.caption("LSTM + VADER Sentiment Analysis — Ablation Study Dashboard")

# ---------------------------------------------------------------------------
# Asset catalogue
# ---------------------------------------------------------------------------
ASSETS: dict[str, dict] = {
    "Bitcoin — BTC-USD (4H bars)": {
        "symbol": "BTC-USD",
        "interval": "4h",
        "default_task": "regression",
        "info": "📅 Jun 2025 – Jun 2026  ·  Local hourly data aggregated to 4-hour bars",
    },
    "Ethereum — ETH-USD (Daily)": {
        "symbol": "ETH-USD",
        "interval": "1d",
        "default_task": "regression",
        "info": "📅 Jan 2021 – Sep 2022  ·  yfinance daily bars",
    },
    "NVIDIA — NVDA (Weekly)": {
        "symbol": "NVDA",
        "interval": "1wk",
        "default_task": "regression",
        "info": "📅 Jan 2018 – Feb 2024  ·  yfinance weekly bars  ·  Target: weekly % return",
    },
    "Apple — AAPL (Weekly)": {
        "symbol": "AAPL",
        "interval": "1wk",
        "default_task": "regression",
        "info": "📅 Jan 2018 – Nov 2024  ·  yfinance weekly bars  ·  Target: weekly % return",
    },
}

SCENARIO_LABELS = {
    "price_only":                "Price Only",
    "price_technical":           "Price + Technical Indicators",
    "price_technical_sentiment": "Price + Technical + Sentiment",
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    asset_label = st.selectbox("Asset", list(ASSETS.keys()))
    asset_cfg   = ASSETS[asset_label]
    symbol      = asset_cfg["symbol"]
    interval    = asset_cfg["interval"]

    st.info(asset_cfg["info"])

    task = st.radio(
        "Prediction task",
        ["regression", "classification"],
        index=0,
        help=(
            "Regression → predicts % price change; direction accuracy shows up/down correctness.\n"
            "Classification → directly predicts up (1) or down (0)."
        ),
    )

    st.divider()
    run_btn = st.button("▶  Run Ablation", type="primary", use_container_width=True)
    st.caption("⏱ Training takes ~2–3 minutes per asset.")

    st.divider()
    st.subheader("ℹ️ About the scenarios")
    st.markdown(
        "- **Price Only** — close + volume\n"
        "- **Price + Technical** — + 18 technical indicators (RSI, MACD, Bollinger…)\n"
        "- **+ Sentiment** — + VADER/finvader sentiment score"
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
reports_dir = PROJECT_ROOT / "reports" / "figures"
safe_sym    = symbol.replace("-", "_")


def _ablation_img() -> Path:
    return reports_dir / f"{safe_sym}_ablation.png"


def _history_img() -> Path:
    return reports_dir / f"{safe_sym}_history.png"


def _show_charts() -> None:
    col1, col2 = st.columns(2)
    with col1:
        p = _ablation_img()
        if p.exists():
            st.image(str(p), caption="Ablation Scenario Comparison", use_container_width=True)
        else:
            st.info("No ablation chart yet — click **Run Ablation** to generate.")
    with col2:
        p = _history_img()
        if p.exists():
            st.image(str(p), caption="Training History (all 3 scenarios)", use_container_width=True)
        else:
            st.info("No history chart yet.")


def _metrics_table(results: dict, task: str) -> None:
    rows = []
    for scenario, m in results.items():
        row: dict = {"Scenario": SCENARIO_LABELS.get(scenario, scenario)}
        da = m.get("direction_accuracy", m.get("accuracy", float("nan")))
        row["Direction Accuracy"] = f"{da * 100:.1f}%"
        if "r2" in m:
            row["R²"] = f"{m['r2']:.4f}"
        if "mape" in m:
            row["MAPE"] = f"{m['mape']:.2f}%"
        if "mae" in m:
            row["MAE"] = f"{m['mae']:.4f}"
        if "f1" in m:
            row["F1"] = f"{m['f1'] * 100:.1f}%"
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
if run_btn:
    from scripts.run_training import load_config, run_symbol  # lazy import

    cfg     = load_config()
    log_buf = io.StringIO()

    with st.spinner(f"Training {symbol} ({interval}) — please wait…"):
        with contextlib.redirect_stdout(log_buf):
            results = run_symbol(
                symbol, cfg,
                task=task,
                ablation=True,
                interval_override=interval,
            )

    log_text = log_buf.getvalue()
    st.session_state["log"]        = log_text
    st.session_state["results"]    = results
    st.session_state["ran_symbol"] = symbol
    st.session_state["ran_task"]   = task
    st.success(f"✅ Training complete for **{symbol}**!")

    _show_charts()

    if results:
        st.subheader("📊 Metrics Summary")
        _metrics_table(results, task)

    with st.expander("📋 Training log"):
        st.code(log_text, language=None)

else:
    _show_charts()

    # If this symbol was run earlier in the same session, show those results
    if (
        "results" in st.session_state
        and st.session_state.get("ran_symbol") == symbol
    ):
        st.subheader("📊 Metrics Summary (last run)")
        _metrics_table(st.session_state["results"], st.session_state.get("ran_task", task))

        with st.expander("📋 Training log"):
            st.code(st.session_state.get("log", ""), language=None)
