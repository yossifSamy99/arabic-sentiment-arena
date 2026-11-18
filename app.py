"""
Arabic Sentiment Arena — Gradio App
واجهة Gradio احترافية بـ Dark Theme كامل
"""

import os
import json
import tempfile
from functools import lru_cache

import pandas as pd
import plotly.graph_objects as go
import gradio as gr
import torch
import spaces
from transformers import AutoTokenizer, AutoModel

from core.config import (
    REPO_ID,
    BASE_CAMELBERT_MODEL,
    MODEL_OPTIONS,
    MAX_BATCH_PREDICT_ROWS,
    TRAINER_MIN_ROWS,
    TRAINER_MAX_ROWS,
)
from core.wrappers import TfidfWrapper, FastTextWrapper, BiLstmWrapper, CamelBertWrapper
from core.live_trainer import GuardrailError, train_dual_custom_models, predict_custom_tweet

# ──────────────────────────────────────────────────────────────
# مسار benchmark_results.json — مطلق
# ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_FILE_PATH = os.path.join(_HERE, "artifacts", "benchmark_results.json")

# ──────────────────────────────────────────────────────────────
# CSS مخصص كامل
# ──────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg:         #0D1117;
    --surface:    #161B22;
    --surface2:   #1C2333;
    --border:     #30363D;
    --accent:     #58A6FF;
    --accent2:    #3FB950;
    --accent3:    #F78166;
    --accent4:    #D2A8FF;
    --text-p:     #E6EDF3;
    --text-m:     #8B949E;
    --pos:        #3FB950;
    --neg:        #F85149;
    --neu:        #58A6FF;
    --gold:       #E3B341;
}

/* ═══ قاعدة عامة ═══ */
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    font-family: 'IBM Plex Sans Arabic', 'Segoe UI', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text-p) !important;
    direction: rtl;
}
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; padding: 0 1.5rem 3rem !important; }
footer { display: none !important; }

/* ═══ Hero ═══ */
.hero-wrap {
    background: linear-gradient(135deg, #161B22 0%, #1C2333 60%, #0D1117 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content:'';
    position:absolute; top:-70px; right:-70px;
    width:240px; height:240px;
    background: radial-gradient(circle, rgba(88,166,255,.1) 0%, transparent 70%);
    pointer-events:none;
}
.hero-title { font-size:1.9rem; font-weight:700; color:var(--text-p); margin:0 0 .3rem; line-height:1.2; }
.hero-title span { color:var(--accent); }
.hero-sub { font-size:.9rem; color:var(--text-m); margin:0 0 .9rem; }
.hero-badges { display:flex; gap:.45rem; flex-wrap:wrap; }
.hbadge {
    background:var(--surface2); border:1px solid var(--border);
    border-radius:20px; padding:.2rem .7rem;
    font-size:.75rem; color:var(--text-m);
    font-family:'IBM Plex Mono',monospace;
}

/* ═══ KPI Cards ═══ */
.kpi-row { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1.5rem; }
.kpi-card {
    background:var(--surface); border:1px solid var(--border);
    border-radius:12px; padding:1.1rem 1.4rem;
    position:relative; overflow:hidden;
}
.kpi-card::after {
    content:''; position:absolute; bottom:0; left:0; right:0;
    height:3px; border-radius:0 0 12px 12px;
}
.kpi-card.gold::after  { background:var(--gold); }
.kpi-card.green::after { background:var(--pos); }
.kpi-card.blue::after  { background:var(--accent); }
.kpi-label { font-size:.76rem; color:var(--text-m); margin-bottom:.3rem; }
.kpi-value { font-size:1.75rem; font-weight:700; color:var(--text-p); font-family:'IBM Plex Mono',monospace; line-height:1; }
.kpi-sub   { font-size:.74rem; color:var(--text-m); margin-top:.25rem; }

/* ═══ Tabs ═══ */
.tab-nav { display:flex; gap:0; background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:4px; margin-bottom:1.25rem; }
button.tab-btn {
    flex:1; background:transparent; border:none; cursor:pointer;
    padding:.45rem .9rem; border-radius:8px; font-size:.875rem; font-weight:500;
    color:var(--text-m); font-family:'IBM Plex Sans Arabic',sans-serif;
    transition:all .15s;
}
button.tab-btn.active { background:var(--surface2); color:var(--text-p); border:1px solid var(--border); }
.tab-panel { display:none; }
.tab-panel.active { display:block; }

/* ═══ Gradio overrides ═══ */
.gr-box, .gr-form, .gr-panel, .gr-padded,
div.svelte-1gfkfd6, div.svelte-10ogue4 {
    background:var(--surface) !important;
    border-color:var(--border) !important;
    border-radius:12px !important;
}
.gr-button-primary {
    background:var(--accent) !important;
    border:none !important; color:#0D1117 !important;
    font-family:'IBM Plex Sans Arabic',sans-serif !important;
    font-weight:600 !important; border-radius:8px !important;
}
.gr-button-primary:hover { background:#79B8FF !important; }
.gr-button-secondary {
    background:var(--surface) !important;
    border:1px solid var(--border) !important;
    color:var(--text-p) !important;
    font-family:'IBM Plex Sans Arabic',sans-serif !important;
    border-radius:8px !important;
}
.gr-input, .gr-textarea, input, textarea, select {
    background:var(--surface) !important;
    border:1px solid var(--border) !important;
    border-radius:8px !important;
    color:var(--text-p) !important;
    font-family:'IBM Plex Sans Arabic',sans-serif !important;
}
.gr-input:focus, .gr-textarea:focus, input:focus, textarea:focus {
    border-color:var(--accent) !important;
    box-shadow:0 0 0 3px rgba(88,166,255,.12) !important;
    outline:none !important;
}
label, .gr-label { color:var(--text-m) !important; font-size:.82rem !important; }
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 { color:var(--text-p) !important; }
.gr-markdown p, .gr-markdown li { color:var(--text-m) !important; }

/* ═══ Result Cards ═══ */
.result-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:.9rem; margin-top:1rem; }
.result-card {
    background:var(--surface); border:1px solid var(--border);
    border-radius:12px; padding:1.1rem 1.3rem;
    transition:border-color .2s;
}
.result-card:hover { border-color:var(--accent); }
.result-card.winner { border-color:var(--gold) !important; }
.rc-name { font-size:.74rem; color:var(--text-m); margin-bottom:.5rem; font-family:'IBM Plex Mono',monospace; }
.rc-label-pos { font-size:1.25rem; font-weight:700; color:var(--pos); }
.rc-label-neg { font-size:1.25rem; font-weight:700; color:var(--neg); }
.rc-label-neu { font-size:1.25rem; font-weight:700; color:var(--neu); }
.rc-bar-wrap { background:var(--surface2); border-radius:6px; height:5px; margin:.55rem 0; overflow:hidden; }
.rc-bar { height:100%; border-radius:6px; }
.rc-meta { display:flex; justify-content:space-between; font-size:.73rem; color:var(--text-m); font-family:'IBM Plex Mono',monospace; }
.winner-badge { font-size:.68rem; color:var(--gold); }

/* ═══ Section headers ═══ */
.sec-title { font-size:.95rem; font-weight:600; color:var(--text-p); margin:.5rem 0 .9rem; display:flex; align-items:center; gap:.4rem; }
.divider { border:none; border-top:1px solid var(--border); margin:1.1rem 0; }

/* ═══ Bench Table ═══ */
.bench-table { width:100%; border-collapse:collapse; font-size:.855rem; }
.bench-table th {
    background:var(--surface2); color:var(--text-m); font-weight:500;
    padding:.55rem 1rem; text-align:right;
    border-bottom:1px solid var(--border);
    font-size:.76rem; font-family:'IBM Plex Mono',monospace;
}
.bench-table td { padding:.6rem 1rem; border-bottom:1px solid var(--border); color:var(--text-p); vertical-align:middle; }
.bench-table tr:last-child td { border-bottom:none; }
.bench-table tr:hover td { background:rgba(255,255,255,.02); }
.rank { display:inline-block; width:22px; height:22px; line-height:22px; text-align:center; border-radius:50%; font-size:.7rem; font-weight:700; background:var(--surface2); color:var(--text-m); }
.rank.r1 { background:var(--gold); color:#000; }
.rank.r2 { background:#9E9E9E; color:#000; }
.rank.r3 { background:#CD7F32; color:#fff; }
.bar-cell { display:flex; align-items:center; gap:.45rem; }
.mini-bar-wrap { flex:1; background:var(--surface2); border-radius:4px; height:4px; overflow:hidden; min-width:60px; }
.mini-bar { height:100%; border-radius:4px; }

/* ═══ Status boxes ═══ */
.stat-box { padding:.65rem 1rem; border-radius:8px; font-size:.855rem; margin:.6rem 0; display:flex; align-items:center; gap:.5rem; }
.stat-success { background:rgba(63,185,80,.1); border:1px solid rgba(63,185,80,.3); color:var(--pos); }
.stat-error   { background:rgba(248,81,73,.1);  border:1px solid rgba(248,81,73,.3);  color:var(--neg); }
.stat-info    { background:rgba(88,166,255,.1); border:1px solid rgba(88,166,255,.3); color:var(--accent); }

/* ═══ Comparison table (custom trainer) ═══ */
.cmp-table { width:100%; border-collapse:collapse; font-size:.855rem; margin-top:.5rem; }
.cmp-table th { background:var(--surface2); color:var(--text-m); font-weight:500; padding:.55rem 1rem; text-align:right; border-bottom:1px solid var(--border); font-size:.76rem; }
.cmp-table td { padding:.6rem 1rem; border-bottom:1px solid var(--border); color:var(--text-p); }
.cmp-table tr:last-child td { border-bottom:none; }
"""

# ──────────────────────────────────────────────────────────────
# تحميل النماذج
# ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=2)
def get_model(key: str):
    if key == "tfidf":     return TfidfWrapper(repo_id=REPO_ID)
    if key == "fasttext":  return FastTextWrapper(repo_id=REPO_ID)
    if key == "bilstm":    return BiLstmWrapper(repo_id=REPO_ID)
    if key == "camelbert": return CamelBertWrapper(repo_id=REPO_ID)
    raise ValueError(key)


@lru_cache(maxsize=1)
def get_encoder():
    tok = AutoTokenizer.from_pretrained(BASE_CAMELBERT_MODEL)
    enc = AutoModel.from_pretrained(BASE_CAMELBERT_MODEL)
    enc.to(torch.device("cpu")); enc.eval()
    return tok, enc


# ──────────────────────────────────────────────────────────────
# مساعدات العرض
# ──────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "tfidf":     "#58A6FF",
    "fasttext":  "#D2A8FF",
    "bilstm":    "#79C0FF",
    "camelbert": "#3FB950",
}
MODEL_SHORT = {
    "tfidf":     "TF-IDF",
    "fasttext":  "FastText",
    "bilstm":    "Bi-LSTM",
    "camelbert": "CAMeLBERT",
}

PLOT_CFG = dict(
    plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
    font=dict(family="IBM Plex Mono, monospace", color="#8B949E"),
    margin=dict(l=0, r=70, t=30, b=0),
    height=270,
)
GRID = "#21262D"


def label_cls(label: str) -> str:
    lbl = str(label).lower()
    if lbl in {"pos", "positive", "إيجابي", "1"}: return "pos"
    if lbl in {"neg", "negative", "سلبي", "0"}:   return "neg"
    return "neu"


def label_ar(label: str) -> str:
    return {"pos": "إيجابي ✦", "neg": "سلبي ✦"}.get(label_cls(label), f"{label} ✦")


def result_cards_html(results: list, keys: list) -> str:
    if not results:
        return ""
    best = max(r["confidence"] for r in results)
    cards = '<div class="result-grid">'
    for res, key in zip(results, keys):
        cls     = label_cls(res["label"])
        winner  = res["confidence"] == best and len(results) > 1
        conf    = res["confidence"] * 100
        bar_clr = {"pos": "#3FB950", "neg": "#F85149", "neu": "#58A6FF"}[cls]
        color   = MODEL_COLORS.get(key, "#58A6FF")
        wbadge  = '<div class="winner-badge">🏆 الأعلى ثقةً</div>' if winner else ""
        wclass  = "winner" if winner else ""
        cards += f"""
        <div class="result-card {wclass}">
            <div class="rc-name" style="color:{color};">{MODEL_SHORT.get(key, key)}</div>
            {wbadge}
            <div class="rc-label-{cls}">{label_ar(res['label'])}</div>
            <div class="rc-bar-wrap">
                <div class="rc-bar" style="width:{conf:.1f}%;background:{bar_clr};"></div>
            </div>
            <div class="rc-meta">
                <span>ثقة {conf:.1f}%</span>
                <span>{res['latency_ms']} ms</span>
            </div>
        </div>"""
    cards += "</div>"
    return cards


def bench_table_html(bench_df: pd.DataFrame) -> str:
    df = bench_df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    max_acc = df["accuracy"].max()
    max_lat = df["avg_latency_ms"].max()
    ranks = ["r1", "r2", "r3", ""]
    rows = ""
    for i, row in df.iterrows():
        rc    = ranks[i] if i < 4 else ""
        bw    = (row["accuracy"] / max_acc) * 100
        lc    = "#3FB950" if row["avg_latency_ms"] < 5 else ("#E3B341" if row["avg_latency_ms"] < 20 else "#F85149")
        lbw   = (row["avg_latency_ms"] / max_lat) * 100
        rows += f"""
        <tr>
            <td><span class="rank {rc}">{i+1}</span></td>
            <td style="font-weight:600;">{row['model_name']}</td>
            <td>
                <div class="bar-cell">
                    <div class="mini-bar-wrap"><div class="mini-bar" style="width:{bw:.0f}%;background:#58A6FF;"></div></div>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:.8rem;">{row['accuracy']*100:.2f}%</span>
                </div>
            </td>
            <td style="font-family:'IBM Plex Mono',monospace;font-size:.8rem;">{row['macro_f1']*100:.2f}%</td>
            <td>
                <div class="bar-cell">
                    <div class="mini-bar-wrap"><div class="mini-bar" style="width:{lbw:.0f}%;background:{lc};"></div></div>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:.8rem;color:{lc};">{row['avg_latency_ms']} ms</span>
                </div>
            </td>
        </tr>"""
    return f"""
    <table class="bench-table">
        <thead><tr>
            <th style="width:50px;">#</th>
            <th>النموذج</th><th>Accuracy</th><th>Macro F1</th><th>CPU Latency</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


# ──────────────────────────────────────────────────────────────
# تحميل بيانات Benchmark
# ──────────────────────────────────────────────────────────────
def load_bench():
    if not os.path.exists(BENCHMARK_FILE_PATH):
        return None
    with open(BENCHMARK_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame.from_dict(data, orient="index").reset_index(drop=True)


bench_df_global = load_bench()


def kpi_html() -> str:
    if bench_df_global is None:
        return ""
    b_acc = bench_df_global.loc[bench_df_global["accuracy"].idxmax()]
    b_f1  = bench_df_global.loc[bench_df_global["macro_f1"].idxmax()]
    fast  = bench_df_global.loc[bench_df_global["avg_latency_ms"].idxmin()]
    return f"""
    <div class="kpi-row">
        <div class="kpi-card gold">
            <div class="kpi-label">🏆 أعلى دقة (Accuracy)</div>
            <div class="kpi-value">{b_acc['accuracy']*100:.2f}<span style="font-size:1rem;">%</span></div>
            <div class="kpi-sub">{b_acc['model_name']}</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-label">🎯 أعلى Macro F1</div>
            <div class="kpi-value">{b_f1['macro_f1']*100:.2f}<span style="font-size:1rem;">%</span></div>
            <div class="kpi-sub">{b_f1['model_name']}</div>
        </div>
        <div class="kpi-card blue">
            <div class="kpi-label">⚡ أسرع استجابة (CPU)</div>
            <div class="kpi-value">{fast['avg_latency_ms']}<span style="font-size:1rem;"> ms</span></div>
            <div class="kpi-sub">{fast['model_name']}</div>
        </div>
    </div>"""


# ──────────────────────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────────────────────
def make_acc_chart(df: pd.DataFrame):
    df_s = df.sort_values("accuracy")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Accuracy", y=df_s["model_name"], x=df_s["accuracy"],
        orientation="h", marker_color="#58A6FF", marker_line_width=0,
        text=[f"{v*100:.2f}%" for v in df_s["accuracy"]],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono,monospace", size=11, color="#8B949E"),
    ))
    fig.add_trace(go.Bar(
        name="Macro F1", y=df_s["model_name"], x=df_s["macro_f1"],
        orientation="h", marker_color="#3FB950", marker_line_width=0,
        text=[f"{v*100:.2f}%" for v in df_s["macro_f1"]],
        textposition="outside", opacity=0.75,
        textfont=dict(family="IBM Plex Mono,monospace", size=11, color="#8B949E"),
    ))
    fig.update_layout(
        **PLOT_CFG, barmode="group", bargap=0.3, bargroupgap=0.08,
        xaxis=dict(range=[0, 1.15], showgrid=True, gridcolor=GRID,
                   zeroline=False, tickformat=".0%", tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#C9D1D9")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def make_lat_chart(df: pd.DataFrame):
    df_s = df.sort_values("avg_latency_ms", ascending=False)
    colors = ["#3FB950" if v < 5 else ("#E3B341" if v < 20 else "#F85149")
              for v in df_s["avg_latency_ms"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_s["model_name"], x=df_s["avg_latency_ms"],
        orientation="h", marker_color=colors, marker_line_width=0,
        text=[f"{v:.3f} ms" for v in df_s["avg_latency_ms"]],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono,monospace", size=11, color="#8B949E"),
    ))
    fig.update_layout(
        **PLOT_CFG, showlegend=False,
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False,
                   ticksuffix=" ms", tickfont=dict(size=10), type="log"),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#C9D1D9")),
    )
    fig.add_vline(x=10, line_dash="dot", line_color="rgba(255,255,255,.12)",
                  annotation_text="10ms", annotation_font_size=10,
                  annotation_font_color="#8B949E")
    return fig


def make_scatter_chart(df: pd.DataFrame):
    keys   = ["tfidf", "fasttext", "bilstm", "camelbert"]
    colors = [MODEL_COLORS.get(k, "#58A6FF") for k in keys]
    fig = go.Figure()
    for i, (_, row) in enumerate(df.iterrows()):
        c = colors[i] if i < len(colors) else "#58A6FF"
        fig.add_trace(go.Scatter(
            x=[row["avg_latency_ms"]], y=[row["accuracy"]],
            mode="markers+text",
            marker=dict(size=15, color=c, line=dict(width=2, color="rgba(255,255,255,.18)")),
            text=[row["model_name"].split(" (")[0].split(" +")[0]],
            textposition="top center",
            textfont=dict(size=11, color="#C9D1D9", family="IBM Plex Mono,monospace"),
            showlegend=False,
        ))
    fig.update_layout(
        **PLOT_CFG, height=300,
        xaxis=dict(title="CPU Latency (ms) — log scale", type="log",
                   showgrid=True, gridcolor=GRID, zeroline=False,
                   tickfont=dict(size=10), title_font=dict(size=11)),
        yaxis=dict(title="Accuracy", range=[0.65, 1.0],
                   showgrid=True, gridcolor=GRID, zeroline=False,
                   tickformat=".0%", tickfont=dict(size=10), title_font=dict(size=11)),
        margin=dict(l=50, r=20, t=20, b=50),
    )
    return fig


def make_cmp_chart(tfidf_r, camel_r):
    fig = go.Figure()
    metrics = ["Accuracy", "Macro F1"]
    tv = [tfidf_r["accuracy"], tfidf_r["macro_f1"]]
    cv = [camel_r["accuracy"], camel_r["macro_f1"]]
    fig.add_trace(go.Bar(
        name=tfidf_r["model_name"], x=metrics, y=tv,
        marker_color="#58A6FF", marker_line_width=0,
        text=[f"{v*100:.2f}%" for v in tv], textposition="outside",
        textfont=dict(family="IBM Plex Mono,monospace", size=11, color="#8B949E"),
    ))
    fig.add_trace(go.Bar(
        name=camel_r["model_name"], x=metrics, y=cv,
        marker_color="#3FB950", marker_line_width=0,
        text=[f"{v*100:.2f}%" for v in cv], textposition="outside",
        textfont=dict(family="IBM Plex Mono,monospace", size=11, color="#8B949E"),
    ))
    fig.update_layout(
        **PLOT_CFG, barmode="group", bargap=0.4, bargroupgap=0.1,
        xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#C9D1D9")),
        yaxis=dict(range=[0, 1.15], showgrid=True, gridcolor=GRID,
                   tickformat=".0%", zeroline=False, tickfont=dict(size=10)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# دوال المنطق
# ──────────────────────────────────────────────────────────────
SAMPLE_TWEETS = {
    "— اختر مثالاً جاهزاً —": "",
    "إيجابي — عامية مصرية": "الخدمة النهاردة كانت فوق الممتاز بجد والتعامل راقي جداً 😍👏",
    "سلبي — أداة نفي": "للأسف التجربة مش حلوة خالص والتوصيل اتأخر ومش هكررها تاني 😡",
    "ساركازم — لهجة مختلطة": "يعني بعد انتظار ساعتين الأكل يوصل بارد! شابوه بجد على الاحترافية 🙄",
}


def on_example_select(key: str) -> str:
    return SAMPLE_TWEETS.get(key, "")


@spaces.GPU
def run_analysis(tweet: str, compare_all: bool, single_key: str):
    if not tweet or not tweet.strip():
        return '<div class="stat-box stat-error">⚠️ من فضلك اكتب نص التويتة أولاً.</div>', ""

    keys = list(MODEL_OPTIONS.keys()) if compare_all else [single_key]
    results = []
    for key in keys:
        wrapper = get_model(key)
        results.append(wrapper.predict(tweet))

    cards = result_cards_html(results, keys)

    # جدول تفصيلي
    rows = "".join(
        f"<tr><td>{r['model_name']}</td><td>{r['label']}</td>"
        f"<td style='font-family:\"IBM Plex Mono\",monospace;'>{r['confidence']*100:.1f}%</td>"
        f"<td style='font-family:\"IBM Plex Mono\",monospace;'>{r['latency_ms']} ms</td></tr>"
        for r in results
    )
    table = f"""
    <hr class="divider" style="margin-top:1.2rem;">
    <div class="sec-title">جدول تفصيلي</div>
    <table class="bench-table">
        <thead><tr><th>النموذج</th><th>التصنيف</th><th>الثقة</th><th>الزمن</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>"""

    return cards, table


def inspect_batch(file):
    if file is None:
        return gr.Dropdown(choices=[], value=None), ""
    df = pd.read_csv(file.name, nrows=5)
    cols = list(df.columns)
    info = f'<div class="stat-box stat-info">📄 تم التحميل — اختر عمود النصوص</div>'
    return gr.Dropdown(choices=cols, value=cols[0] if cols else None), info


@spaces.GPU
def run_batch(file, text_col: str, model_key: str):
    if file is None or not text_col:
        return '<div class="stat-box stat-error">⚠️ ارفع ملف واختر عمود النصوص أولاً.</div>', None

    df = pd.read_csv(file.name, nrows=MAX_BATCH_PREDICT_ROWS)
    texts = df[text_col].fillna("").astype(str).tolist()
    wrapper = get_model(model_key)
    preds = wrapper.predict_batch(texts)
    df["predicted_label"] = [p["label"] for p in preds]
    df["confidence"]      = [p["confidence"] for p in preds]

    out = os.path.join(tempfile.gettempdir(), f"preds_{model_key}.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    info = f'<div class="stat-box stat-success">✅ تم التنبؤ على {len(df)} صف</div>'
    return info, out


def inspect_trainer(file):
    if file is None:
        return gr.Dropdown(choices=[], value=None), gr.Dropdown(choices=[], value=None)
    df = pd.read_csv(file.name, nrows=5)
    cols = list(df.columns)
    t_def = cols[0] if cols else None
    l_def = cols[1] if len(cols) > 1 else t_def
    return (gr.Dropdown(choices=cols, value=t_def),
            gr.Dropdown(choices=cols, value=l_def))


@spaces.GPU
def run_training(file, text_col: str, label_col: str):
    if file is None:
        return ('<div class="stat-box stat-error">⚠️ ارفع ملف CSV أولاً.</div>',
                None, None, None, None)
    try:
        df = pd.read_csv(file.name)
        tok, enc = get_encoder()
        bundle = train_dual_custom_models(df, text_col, label_col, tok, enc)

        status = (f'<div class="stat-box stat-success">'
                  f'✅ تم التدريب على {bundle["dataset_info"]["train_rows"]} صف '
                  f'| تقييم على {bundle["dataset_info"]["val_rows"]} صف</div>')

        t = bundle["tfidf"]
        c = bundle["camelbert_emb"]
        t_win = t["accuracy"] >= c["accuracy"]

        def _h(v, win):
            clr = "#3FB950" if win else "var(--text-p)"
            return f'<b style="color:{clr};">{v*100:.2f}%</b>'

        cmp = f"""
        <table class="cmp-table">
            <thead><tr><th>النموذج</th><th>Accuracy</th><th>Macro F1</th><th>وقت التدريب</th><th>Latency</th></tr></thead>
            <tbody>
            <tr>
                <td style="color:#58A6FF;font-weight:600;">{t['model_name']}</td>
                <td>{_h(t['accuracy'], t_win)}</td>
                <td style="font-family:'IBM Plex Mono',monospace;">{t['macro_f1']*100:.2f}%</td>
                <td style="font-family:'IBM Plex Mono',monospace;">{t['train_time_sec']} ث</td>
                <td style="font-family:'IBM Plex Mono',monospace;">{t['avg_latency_ms']} ms</td>
            </tr>
            <tr>
                <td style="color:#3FB950;font-weight:600;">{c['model_name']}</td>
                <td>{_h(c['accuracy'], not t_win)}</td>
                <td style="font-family:'IBM Plex Mono',monospace;">{c['macro_f1']*100:.2f}%</td>
                <td style="font-family:'IBM Plex Mono',monospace;">{c['train_time_sec']} ث</td>
                <td style="font-family:'IBM Plex Mono',monospace;">{c['avg_latency_ms']} ms</td>
            </tr>
            </tbody>
        </table>"""

        fig = make_cmp_chart(t, c)
        return status, cmp, fig, bundle, '<div class="stat-box stat-info">💡 اكتب جملة أدناه لاختبار النموذجين</div>'

    except GuardrailError as ge:
        return (f'<div class="stat-box stat-error">⚠️ {ge}</div>',
                None, None, None, None)
    except Exception as ex:
        return (f'<div class="stat-box stat-error">❌ خطأ: {ex}</div>',
                None, None, None, None)


@spaces.GPU
def run_custom_predict(text: str, bundle):
    if bundle is None:
        return '<div class="stat-box stat-error">⚠️ درّب النماذج أولاً.</div>'
    if not text or not text.strip():
        return '<div class="stat-box stat-error">⚠️ اكتب جملة للاختبار.</div>'
    tok, enc = get_encoder()
    preds = predict_custom_tweet(text, bundle, tok, enc)
    keys  = ["tfidf", "camelbert"]
    return result_cards_html(preds, keys)


# ──────────────────────────────────────────────────────────────
# بناء واجهة Gradio
# ──────────────────────────────────────────────────────────────
MODEL_CHOICES = [(v, k) for k, v in MODEL_OPTIONS.items()]

with gr.Blocks(css=CUSTOM_CSS, title="Arabic Sentiment Arena") as demo:

    # Hero
    gr.HTML("""
    <div class="hero-wrap">
        <div class="hero-title">Arabic Sentiment <span>Arena</span> 🎯</div>
        <p class="hero-sub">منصة مقارنة وتحليل المشاعر العربية — 4 مدارس من تعلم الآلة والتعلم العميق والـ Transformers</p>
        <div class="hero-badges">
            <span class="hbadge">TF-IDF + LR</span>
            <span class="hbadge">FastText + LR</span>
            <span class="hbadge">Bi-LSTM PyTorch</span>
            <span class="hbadge">CAMeLBERT-Mix ✦</span>
            <span class="hbadge">58K tweets</span>
        </div>
    </div>
    """)

    # KPI Cards
    gr.HTML(kpi_html())

    # ══ Tabs ══════════════════════════════════════════════════
    with gr.Tabs():

        # ── TAB 1: Live Arena ─────────────────────────────────
        with gr.Tab("⚔️  Live Arena"):

            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML('<div class="sec-title">📝 أدخل التويتة</div>')
                    example_dd = gr.Dropdown(
                        choices=list(SAMPLE_TWEETS.keys()),
                        value="— اختر مثالاً جاهزاً —",
                        label="مثال جاهز",
                        show_label=False,
                    )
                    tweet_box = gr.Textbox(
                        lines=4, placeholder="اكتب أي جملة أو تويتة باللهجة المصرية أو الخليجية أو الفصحى...",
                        label="نص التويتة", show_label=False,
                        rtl=True,
                    )

                with gr.Column(scale=2):
                    gr.HTML('<div class="sec-title">⚙️ إعدادات التحليل</div>')
                    compare_chk = gr.Checkbox(label="مقارنة النماذج الأربعة معاً", value=True)
                    single_dd   = gr.Dropdown(
                        choices=MODEL_CHOICES, value="camelbert",
                        label="اختر النموذج (عند إيقاف المقارنة)",
                        interactive=True,
                    )
                    analyze_btn = gr.Button("تحليل المشاعر ◀", variant="primary")

            result_cards  = gr.HTML()
            result_detail = gr.HTML()

            gr.HTML('<hr class="divider">')

            with gr.Accordion("📂 تحليل ملف CSV كامل (Batch)", open=False):
                batch_file = gr.File(label="ارفع ملف CSV", file_types=[".csv"])
                batch_info = gr.HTML()
                with gr.Row():
                    batch_col_dd  = gr.Dropdown(label="عمود النصوص", choices=[])
                    batch_model_dd = gr.Dropdown(choices=MODEL_CHOICES, value="camelbert", label="النموذج")
                batch_btn     = gr.Button("▶ تشغيل التنبؤ على الملف")
                batch_status  = gr.HTML()
                batch_dl      = gr.File(label="⬇️ تحميل النتائج")

            # Events
            example_dd.change(on_example_select, example_dd, tweet_box)
            analyze_btn.click(run_analysis,
                              [tweet_box, compare_chk, single_dd],
                              [result_cards, result_detail])
            batch_file.change(inspect_batch, batch_file, [batch_col_dd, batch_info])
            batch_btn.click(run_batch,
                            [batch_file, batch_col_dd, batch_model_dd],
                            [batch_status, batch_dl])

        # ── TAB 2: Benchmark ──────────────────────────────────
        with gr.Tab("📊  Benchmark"):

            if bench_df_global is not None:
                gr.HTML('<div class="sec-title">📊 نتائج التقييم الموحد على 11,520 تويتة</div>')
                gr.HTML(bench_table_html(bench_df_global))
                gr.HTML('<hr class="divider">')

                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div class="sec-title">دقة النماذج (Accuracy & Macro F1)</div>')
                        gr.Plot(make_acc_chart(bench_df_global))
                    with gr.Column():
                        gr.HTML('<div class="sec-title">سرعة الاستجابة (ms/تويتة) — log scale</div>')
                        gr.Plot(make_lat_chart(bench_df_global))

                gr.HTML('<hr class="divider"><div class="sec-title">التوازن بين الدقة والسرعة (Trade-off)</div>')
                gr.Plot(make_scatter_chart(bench_df_global))
            else:
                gr.HTML(f'<div class="stat-box stat-error">⚠️ ملف النتائج غير موجود في <code>{BENCHMARK_FILE_PATH}</code></div>')

        # ── TAB 3: Custom Trainer ─────────────────────────────
        with gr.Tab("⚡  Custom Trainer"):

            gr.HTML(f"""
            <div class="stat-box stat-info">
                📌 ارفع ملف CSV عربي بين <b>{TRAINER_MIN_ROWS}</b> و <b>{TRAINER_MAX_ROWS}</b> صف،
                يحتوي على عمود نص وعمود تصنيف بفئتين على الأقل.
            </div>""")

            trainer_file = gr.File(label="ارفع ملف التدريب (CSV)", file_types=[".csv"])
            with gr.Row():
                tr_text_dd  = gr.Dropdown(label="عمود النص",       choices=[])
                tr_label_dd = gr.Dropdown(label="عمود التصنيف",    choices=[])
            train_btn    = gr.Button("🚀 ابدأ التدريب اللحظي والمقارنة", variant="primary")

            train_status = gr.HTML()
            train_cmp    = gr.HTML()
            train_fig    = gr.Plot()
            bundle_state = gr.State(value=None)
            test_hint    = gr.HTML()

            gr.HTML('<hr class="divider"><div class="sec-title">🧪 اختبر نموذجك الجديد</div>')
            test_input   = gr.Textbox(label="اكتب جملة", placeholder="اكتب هنا ...", rtl=True)
            test_result  = gr.HTML()

            # Events
            trainer_file.change(inspect_trainer, trainer_file, [tr_text_dd, tr_label_dd])
            train_btn.click(run_training,
                            [trainer_file, tr_text_dd, tr_label_dd],
                            [train_status, train_cmp, train_fig, bundle_state, test_hint])
            test_input.change(run_custom_predict,
                              [test_input, bundle_state],
                              test_result)

    # ZeroGPU startup hook
    @spaces.GPU
    def _zerogpu_hook(): return ""
    _hidden = gr.Button(visible=False)
    _hidden.click(_zerogpu_hook, outputs=gr.Textbox(visible=False))


if __name__ == "__main__":
    demo.launch()
