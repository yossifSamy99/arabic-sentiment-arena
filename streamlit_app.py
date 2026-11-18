"""
Arabic Sentiment Arena — واجهة Streamlit المُعاد تصميمها
"""

import os
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel

try:
    import spaces

    @spaces.GPU
    def _streamlit_zerogpu_hook():
        pass
except ImportError:
    pass

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
# إعداد الصفحة
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Arabic Sentiment Arena",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# CSS المُخصص — تصميم احترافي وموحد
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ═══ خط + متغيرات الألوان ═══ */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg:          #0D1117;
    --surface:     #161B22;
    --surface2:    #1C2333;
    --border:      #30363D;
    --accent:      #58A6FF;
    --accent2:     #3FB950;
    --accent3:     #F78166;
    --accent4:     #D2A8FF;
    --text-primary:#E6EDF3;
    --text-muted:  #8B949E;
    --pos:         #3FB950;
    --neg:         #F85149;
    --neu:         #58A6FF;
    --gold:        #E3B341;
}

/* ═══ قاعدة عامة ═══ */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans Arabic', 'Segoe UI', sans-serif !important;
    direction: rtl;
}
.main { background: var(--bg) !important; }
.block-container { padding: 1.5rem 2.5rem 3rem !important; max-width: 1400px !important; }

/* ═══ إخفاء عناصر افتراضية ═══ */
#MainMenu, footer, header { visibility: hidden; }

/* ═══ Hero Banner ═══ */
.hero-banner {
    background: linear-gradient(135deg, #161B22 0%, #1C2333 50%, #0D1117 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.75rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(88,166,255,.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 .4rem;
    line-height: 1.25;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    font-size: .925rem;
    color: var(--text-muted);
    margin: 0;
    font-weight: 400;
}
.hero-badges {
    display: flex;
    gap: .5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.badge {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: .25rem .75rem;
    font-size: .78rem;
    color: var(--text-muted);
    font-family: 'IBM Plex Mono', monospace;
}

/* ═══ بطاقات KPI ═══ */
.kpi-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 0 0 12px 12px;
}
.kpi-card.gold::after   { background: var(--gold); }
.kpi-card.green::after  { background: var(--pos); }
.kpi-card.blue::after   { background: var(--accent); }
.kpi-label { font-size: .78rem; color: var(--text-muted); margin-bottom: .35rem; }
.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text-primary);
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1;
}
.kpi-sub { font-size: .78rem; color: var(--text-muted); margin-top: .3rem; }

/* ═══ بطاقات نتائج التحليل ═══ */
.result-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-top: 1rem; }
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    transition: border-color .2s;
}
.result-card:hover { border-color: var(--accent); }
.result-card.winner { border-color: var(--gold) !important; }
.result-card-name { font-size: .78rem; color: var(--text-muted); margin-bottom: .6rem; font-family: 'IBM Plex Mono', monospace; }
.result-label-pos { font-size: 1.3rem; font-weight: 700; color: var(--pos); }
.result-label-neg { font-size: 1.3rem; font-weight: 700; color: var(--neg); }
.result-label-neu { font-size: 1.3rem; font-weight: 700; color: var(--neu); }
.result-conf-bar-wrap { background: var(--surface2); border-radius: 6px; height: 6px; margin: .6rem 0; overflow: hidden; }
.result-conf-bar { height: 100%; border-radius: 6px; background: linear-gradient(90deg, var(--accent), var(--accent2)); }
.result-meta { display: flex; justify-content: space-between; font-size: .75rem; color: var(--text-muted); font-family: 'IBM Plex Mono', monospace; }

/* ═══ بطاقة نتيجة مفردة (للـ Custom Trainer) ═══ */
.mini-result-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: .75rem; }

/* ═══ تنسيق عام للحاويات ═══ */
.section-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.25rem;
}
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: .9rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}

/* ═══ جدول المقارنة ═══ */
.cmp-table { width: 100%; border-collapse: collapse; font-size: .875rem; }
.cmp-table th {
    background: var(--surface2);
    color: var(--text-muted);
    font-weight: 500;
    padding: .6rem 1rem;
    text-align: right;
    border-bottom: 1px solid var(--border);
    font-size: .78rem;
    font-family: 'IBM Plex Mono', monospace;
}
.cmp-table td {
    padding: .65rem 1rem;
    border-bottom: 1px solid var(--border);
    color: var(--text-primary);
    vertical-align: middle;
}
.cmp-table tr:last-child td { border-bottom: none; }
.cmp-table tr:hover td { background: rgba(255,255,255,.025); }
.rank-badge {
    display: inline-block;
    width: 22px; height: 22px;
    line-height: 22px;
    text-align: center;
    border-radius: 50%;
    font-size: .72rem;
    font-weight: 700;
    background: var(--surface2);
    color: var(--text-muted);
}
.rank-badge.r1 { background: var(--gold); color: #000; }
.rank-badge.r2 { background: #9E9E9E; color: #000; }
.rank-badge.r3 { background: #CD7F32; color: #fff; }
.bar-cell { display: flex; align-items: center; gap: .5rem; }
.mini-bar-wrap { flex: 1; background: var(--surface2); border-radius: 4px; height: 5px; overflow: hidden; }
.mini-bar { height: 100%; border-radius: 4px; }

/* ═══ تبويبات مخصصة ═══ */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: var(--surface) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: .5rem 1.1rem !important;
    font-size: .88rem !important;
    color: var(--text-muted) !important;
    background: transparent !important;
    border: none !important;
    font-family: 'IBM Plex Sans Arabic', sans-serif !important;
    font-weight: 500 !important;
    transition: all .15s !important;
}
.stTabs [aria-selected="true"] {
    background: var(--surface2) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.25rem !important; }
[data-baseweb="tab-highlight"] { display: none !important; }

/* ═══ حقول الإدخال ═══ */
.stTextArea textarea, .stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Sans Arabic', sans-serif !important;
    direction: rtl !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,.1) !important;
}
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* ═══ أزرار ═══ */
.stButton > button {
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans Arabic', sans-serif !important;
    font-weight: 600 !important;
    transition: all .15s !important;
}
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border: none !important;
    color: #0D1117 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #79B8FF !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(88,166,255,.3) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
}

/* ═══ رسائل الحالة ═══ */
.status-box {
    padding: .75rem 1rem;
    border-radius: 8px;
    font-size: .875rem;
    margin: .75rem 0;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.status-box.success { background: rgba(63,185,80,.1); border: 1px solid rgba(63,185,80,.3); color: var(--pos); }
.status-box.error   { background: rgba(248,81,73,.1);  border: 1px solid rgba(248,81,73,.3);  color: var(--neg); }
.status-box.info    { background: rgba(88,166,255,.1); border: 1px solid rgba(88,166,255,.3); color: var(--accent); }

/* ═══ فاصل مخصص ═══ */
.custom-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.25rem 0;
}

/* ═══ تسميات المدخلات ═══ */
.stTextArea label, .stTextInput label,
.stSelectbox label, .stFileUploader label,
.stCheckbox label span {
    color: var(--text-muted) !important;
    font-size: .82rem !important;
    font-weight: 500 !important;
}

/* ═══ نتائج الجدول (Dataframe) ═══ */
.stDataFrame { border-radius: 8px; overflow: hidden; }
iframe[title="st_aggrid.agGrid"] { border-radius: 8px; }

/* ═══ Spinner ═══ */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ═══ Upload ═══ */
.stFileUploader > div {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
}
.stFileUploader > div:hover { border-color: var(--accent) !important; }

/* ═══ Checkbox toggle ═══ */
.stCheckbox { margin: .25rem 0; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# مسار benchmark_results.json — مطلق لتجنب مشكلة الـ CWD
# ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_FILE_PATH = os.path.join(_HERE, "artifacts", "benchmark_results.json")


# ──────────────────────────────────────────────────────────────
# تحميل النماذج — مع حد أقصى نموذجين في RAM
# ──────────────────────────────────────────────────────────────
@st.cache_resource(max_entries=2)
def get_sentiment_model(model_key: str):
    if model_key == "tfidf":
        return TfidfWrapper(repo_id=REPO_ID)
    if model_key == "fasttext":
        return FastTextWrapper(repo_id=REPO_ID)
    if model_key == "bilstm":
        return BiLstmWrapper(repo_id=REPO_ID)
    if model_key == "camelbert":
        return CamelBertWrapper(repo_id=REPO_ID)
    raise ValueError(f"Unknown model key: {model_key}")


@st.cache_resource(max_entries=1)
def get_frozen_camelbert_encoder():
    tokenizer = AutoTokenizer.from_pretrained(BASE_CAMELBERT_MODEL)
    model = AutoModel.from_pretrained(BASE_CAMELBERT_MODEL)
    model.to(torch.device("cpu"))
    model.eval()
    return tokenizer, model


# ──────────────────────────────────────────────────────────────
# مساعدات العرض
# ──────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "tfidf":      "#58A6FF",
    "fasttext":   "#D2A8FF",
    "bilstm":     "#79C0FF",
    "camelbert":  "#3FB950",
}

MODEL_SHORT = {
    "tfidf":      "TF-IDF",
    "fasttext":   "FastText",
    "bilstm":     "Bi-LSTM",
    "camelbert":  "CAMeLBERT",
}


def label_class(label: str) -> str:
    lbl = str(label).lower()
    if lbl in {"pos", "positive", "إيجابي", "1"}:
        return "pos"
    if lbl in {"neg", "negative", "سلبي", "0"}:
        return "neg"
    return "neu"


def label_ar(label: str) -> str:
    cls = label_class(label)
    return {"pos": "إيجابي ✦", "neg": "سلبي ✦", "neu": f"{label} ✦"}[cls]


def render_result_cards(results: list, keys: list):
    """يرسم بطاقات النتائج بـ HTML مباشرة."""
    best_conf = max(r["confidence"] for r in results)
    cards_html = '<div class="result-grid">'
    for res, key in zip(results, keys):
        cls = label_class(res["label"])
        is_winner = res["confidence"] == best_conf and len(results) > 1
        label_css = f"result-label-{cls}"
        conf_pct = res["confidence"] * 100
        bar_color = {"pos": "#3FB950", "neg": "#F85149", "neu": "#58A6FF"}[cls]
        winner_cls = "winner" if is_winner else ""
        winner_badge = '<span style="font-size:.7rem;color:#E3B341;margin-right:.35rem;">🏆 الأعلى ثقةً</span>' if is_winner else ""
        color = MODEL_COLORS.get(key, "#58A6FF")
        cards_html += f"""
        <div class="result-card {winner_cls}">
            <div class="result-card-name" style="color:{color};">{MODEL_SHORT.get(key, key)}</div>
            {winner_badge}
            <div class="{label_css}">{label_ar(res["label"])}</div>
            <div class="result-conf-bar-wrap">
                <div class="result-conf-bar" style="width:{conf_pct:.1f}%;background:{bar_color};"></div>
            </div>
            <div class="result-meta">
                <span>ثقة {conf_pct:.1f}%</span>
                <span>{res['latency_ms']} ms</span>
            </div>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


def render_benchmark_table(bench_df: pd.DataFrame):
    """جدول مقارنة مرئي بـ HTML."""
    sorted_df = bench_df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    max_acc = sorted_df["accuracy"].max()
    max_lat = sorted_df["avg_latency_ms"].max()

    rank_classes = ["r1", "r2", "r3", ""]
    rows_html = ""
    for i, row in sorted_df.iterrows():
        rank_cls = rank_classes[i] if i < 4 else ""
        acc_pct = row["accuracy"] * 100
        f1_pct  = row["macro_f1"]  * 100
        bar_w_acc = (row["accuracy"] / max_acc) * 100
        bar_w_lat = (row["avg_latency_ms"] / max_lat) * 100
        lat_color = "#3FB950" if row["avg_latency_ms"] < 5 else ("#E3B341" if row["avg_latency_ms"] < 20 else "#F85149")
        rows_html += f"""
        <tr>
            <td><span class="rank-badge {rank_cls}">{i+1}</span></td>
            <td style="font-weight:600;color:var(--text-primary);">{row['model_name']}</td>
            <td>
                <div class="bar-cell">
                    <div class="mini-bar-wrap">
                        <div class="mini-bar" style="width:{bar_w_acc:.0f}%;background:#58A6FF;"></div>
                    </div>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:.82rem;">{acc_pct:.2f}%</span>
                </div>
            </td>
            <td style="font-family:'IBM Plex Mono',monospace;font-size:.82rem;">{f1_pct:.2f}%</td>
            <td>
                <div class="bar-cell">
                    <div class="mini-bar-wrap">
                        <div class="mini-bar" style="width:{bar_w_lat:.0f}%;background:{lat_color};"></div>
                    </div>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:.82rem;color:{lat_color};">{row['avg_latency_ms']} ms</span>
                </div>
            </td>
        </tr>"""

    table_html = f"""
    <table class="cmp-table">
        <thead>
            <tr>
                <th style="width:50px;">#</th>
                <th>النموذج</th>
                <th>Accuracy</th>
                <th>Macro F1</th>
                <th>CPU Latency</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>"""
    st.markdown(table_html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Hero Banner
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">Arabic Sentiment <span>Arena</span> 🎯</div>
    <p class="hero-sub">منصة مقارنة وتحليل المشاعر العربية — 4 مدارس من تعلم الآلة والتعلم العميق والـ Transformers</p>
    <div class="hero-badges">
        <span class="badge">TF-IDF + LR</span>
        <span class="badge">FastText + LR</span>
        <span class="badge">Bi-LSTM PyTorch</span>
        <span class="badge">CAMeLBERT-Mix ✦</span>
        <span class="badge">58K tweets</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# تحميل بيانات Benchmark مرة واحدة
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_benchmark():
    if not os.path.exists(BENCHMARK_FILE_PATH):
        return None
    with open(BENCHMARK_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame.from_dict(data, orient="index").reset_index(drop=True)

bench_df_global = load_benchmark()


# ──────────────────────────────────────────────────────────────
# KPI Cards (ثابتة في الأعلى)
# ──────────────────────────────────────────────────────────────
if bench_df_global is not None:
    best_acc  = bench_df_global.loc[bench_df_global["accuracy"].idxmax()]
    best_f1   = bench_df_global.loc[bench_df_global["macro_f1"].idxmax()]
    fastest   = bench_df_global.loc[bench_df_global["avg_latency_ms"].idxmin()]

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card gold">
            <div class="kpi-label">🏆 أعلى دقة (Accuracy)</div>
            <div class="kpi-value">{best_acc['accuracy']*100:.2f}<span style="font-size:1rem;">%</span></div>
            <div class="kpi-sub">{best_acc['model_name']}</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-label">🎯 أعلى Macro F1</div>
            <div class="kpi-value">{best_f1['macro_f1']*100:.2f}<span style="font-size:1rem;">%</span></div>
            <div class="kpi-sub">{best_f1['model_name']}</div>
        </div>
        <div class="kpi-card blue">
            <div class="kpi-label">⚡ أسرع استجابة (CPU)</div>
            <div class="kpi-value">{fastest['avg_latency_ms']}<span style="font-size:1rem;"> ms</span></div>
            <div class="kpi-sub">{fastest['model_name']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# التبويبات الثلاثة
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "⚔️  Live Arena",
    "📊  Benchmark",
    "⚡  Custom Trainer",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — Live Arena
# ══════════════════════════════════════════════════════════════
with tab1:

    SAMPLE_TWEETS = {
        "— اختر مثالاً جاهزاً —": "",
        "إيجابي — عامية مصرية": "الخدمة النهاردة كانت فوق الممتاز بجد والتعامل راقي جداً 😍👏",
        "سلبي — أداة نفي (اختبار)": "للأسف التجربة مش حلوة خالص والتوصيل اتأخر ومش هكررها تاني 😡",
        "ساركازم — لهجة مختلطة": "يعني بعد انتظار ساعتين الأكل يوصل بارد! شابوه بجد على الاحترافية 🙄",
    }

    col_input, col_options = st.columns([3, 2], gap="large")

    with col_input:
        st.markdown('<div class="section-title">📝 أدخل التويتة</div>', unsafe_allow_html=True)

        example_key = st.selectbox(
            "اختر مثالاً جاهزاً",
            list(SAMPLE_TWEETS.keys()),
            label_visibility="collapsed",
        )
        default_text = SAMPLE_TWEETS[example_key]

        user_tweet = st.text_area(
            "نص التويتة",
            value=default_text,
            height=120,
            placeholder="اكتب أي جملة أو تويتة باللهجة المصرية أو الخليجية أو الفصحى...",
            label_visibility="collapsed",
        )

    with col_options:
        st.markdown('<div class="section-title">⚙️ إعدادات التحليل</div>', unsafe_allow_html=True)

        compare_all = st.checkbox("مقارنة النماذج الأربعة معاً", value=True)
        if not compare_all:
            single_key = st.selectbox(
                "اختر النموذج",
                options=list(MODEL_OPTIONS.keys()),
                format_func=lambda k: MODEL_OPTIONS[k],
            )
        else:
            single_key = None

        st.markdown("<div style='height:.4rem;'></div>", unsafe_allow_html=True)
        analyze_btn = st.button("تحليل المشاعر ◀", type="primary", use_container_width=True)

    # ── النتائج ──────────────────────────────────────────────
    if analyze_btn:
        if not user_tweet.strip():
            st.markdown('<div class="status-box error">⚠️ من فضلك اكتب نص التويتة أولاً.</div>', unsafe_allow_html=True)
        else:
            keys_to_run = list(MODEL_OPTIONS.keys()) if compare_all else [single_key]
            results = []
            with st.spinner("جاري التحليل ..."):
                for key in keys_to_run:
                    wrapper = get_sentiment_model(key)
                    results.append(wrapper.predict(user_tweet))

            st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">📋 نتائج التحليل</div>', unsafe_allow_html=True)
            render_result_cards(results, keys_to_run)

            # جدول تفصيلي قابل للطي
            with st.expander("عرض الجدول التفصيلي"):
                detail_df = pd.DataFrame([{
                    "النموذج": r["model_name"],
                    "التصنيف": r["label"],
                    "الثقة (%)": f"{r['confidence']*100:.1f}%",
                    "زمن الاستجابة (ms)": r["latency_ms"],
                } for r in results])
                st.dataframe(detail_df, use_container_width=True, hide_index=True)

    # ── Batch CSV ─────────────────────────────────────────────
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    with st.expander("📂 تحليل ملف CSV كامل (Batch Prediction)"):
        batch_file = st.file_uploader(
            "ارفع ملف CSV",
            type=["csv"],
            key="batch_csv",
            help=f"الحد الأقصى {MAX_BATCH_PREDICT_ROWS} صف",
        )
        if batch_file is not None:
            # ◀ قراءة بـ nrows مباشرةً لتجنب تحميل ملف ضخم كاملاً
            batch_df = pd.read_csv(batch_file, nrows=MAX_BATCH_PREDICT_ROWS)
            st.markdown(
                f'<div class="status-box info">📄 تم التحميل — <b>{len(batch_df)}</b> صف (الحد الأقصى: {MAX_BATCH_PREDICT_ROWS})</div>',
                unsafe_allow_html=True,
            )
            bc1, bc2 = st.columns(2)
            with bc1:
                batch_text_col = st.selectbox("عمود النصوص", batch_df.columns, key="b_col")
            with bc2:
                batch_model_key = st.selectbox(
                    "النموذج",
                    list(MODEL_OPTIONS.keys()),
                    format_func=lambda k: MODEL_OPTIONS[k],
                    key="b_model",
                )
            if st.button("▶ تشغيل التنبؤ على الملف", key="batch_run"):
                texts_list = batch_df[batch_text_col].fillna("").astype(str).tolist()
                with st.spinner("جاري معالجة الملف ..."):
                    wrapper = get_sentiment_model(batch_model_key)
                    preds = wrapper.predict_batch(texts_list)
                batch_df["predicted_label"] = [p["label"] for p in preds]
                batch_df["confidence"]       = [p["confidence"] for p in preds]
                st.dataframe(batch_df.head(20), use_container_width=True, hide_index=True)
                csv_bytes = batch_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "⬇️ تحميل النتائج CSV",
                    data=csv_bytes,
                    file_name=f"predictions_{batch_model_key}.csv",
                    mime="text/csv",
                )


# ══════════════════════════════════════════════════════════════
# TAB 2 — Benchmark Dashboard
# ══════════════════════════════════════════════════════════════
with tab2:

    if bench_df_global is None:
        st.markdown(
            f'<div class="status-box error">⚠️ ملف النتائج غير موجود في <code>{BENCHMARK_FILE_PATH}</code></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="section-title">📊 نتائج التقييم الموحد على 11,520 تويتة</div>',
            unsafe_allow_html=True,
        )
        render_benchmark_table(bench_df_global)

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2, gap="large")

        PLOT_BG   = "#0D1117"
        PAPER_BG  = "#0D1117"
        GRID_CLR  = "#21262D"
        TEXT_CLR  = "#8B949E"
        FONT_FAM  = "IBM Plex Mono, monospace"

        # ── Chart 1: Accuracy & F1 ──
        with chart_col1:
            st.markdown('<div class="section-title">دقة النماذج (Accuracy & Macro F1)</div>', unsafe_allow_html=True)
            df_sorted = bench_df_global.sort_values("accuracy")
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                name="Accuracy",
                y=df_sorted["model_name"],
                x=df_sorted["accuracy"],
                orientation="h",
                marker_color="#58A6FF",
                marker_line_width=0,
                text=[f"{v*100:.2f}%" for v in df_sorted["accuracy"]],
                textposition="outside",
                textfont=dict(family=FONT_FAM, size=11, color="#8B949E"),
            ))
            fig1.add_trace(go.Bar(
                name="Macro F1",
                y=df_sorted["model_name"],
                x=df_sorted["macro_f1"],
                orientation="h",
                marker_color="#3FB950",
                marker_line_width=0,
                text=[f"{v*100:.2f}%" for v in df_sorted["macro_f1"]],
                textposition="outside",
                textfont=dict(family=FONT_FAM, size=11, color="#8B949E"),
                opacity=0.75,
            ))
            fig1.update_layout(
                barmode="group",
                bargap=0.3,
                bargroupgap=0.08,
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=PAPER_BG,
                font=dict(family=FONT_FAM, color=TEXT_CLR),
                xaxis=dict(
                    range=[0, 1.12],
                    showgrid=True, gridcolor=GRID_CLR,
                    zeroline=False,
                    tickformat=".0%",
                    tickfont=dict(size=10),
                ),
                yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#C9D1D9")),
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=1.02,
                    xanchor="left", x=0,
                    font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)",
                ),
                margin=dict(l=0, r=60, t=36, b=0),
                height=280,
            )
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

        # ── Chart 2: Latency ──
        with chart_col2:
            st.markdown('<div class="section-title">سرعة الاستجابة على CPU (ms / تويتة)</div>', unsafe_allow_html=True)
            df_lat = bench_df_global.sort_values("avg_latency_ms", ascending=False)

            # لون مشروط حسب السرعة
            lat_colors = []
            for v in df_lat["avg_latency_ms"]:
                if v < 5:
                    lat_colors.append("#3FB950")
                elif v < 20:
                    lat_colors.append("#E3B341")
                else:
                    lat_colors.append("#F85149")

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name="CPU Latency",
                y=df_lat["model_name"],
                x=df_lat["avg_latency_ms"],
                orientation="h",
                marker_color=lat_colors,
                marker_line_width=0,
                text=[f"{v:.3f} ms" for v in df_lat["avg_latency_ms"]],
                textposition="outside",
                textfont=dict(family=FONT_FAM, size=11, color="#8B949E"),
            ))
            fig2.update_layout(
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=PAPER_BG,
                font=dict(family=FONT_FAM, color=TEXT_CLR),
                xaxis=dict(
                    showgrid=True, gridcolor=GRID_CLR,
                    zeroline=False,
                    ticksuffix=" ms",
                    tickfont=dict(size=10),
                    type="log",   # log scale لأن القيم بتفرق جداً (0.2 vs 94)
                ),
                yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#C9D1D9")),
                showlegend=False,
                margin=dict(l=0, r=80, t=36, b=0),
                height=280,
            )
            # خط مرجعي عند 10ms
            fig2.add_vline(
                x=10,
                line_dash="dot",
                line_color="rgba(255,255,255,.15)",
                annotation_text="10ms",
                annotation_font_size=10,
                annotation_font_color="#8B949E",
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        # ── Chart 3: Scatter — Accuracy vs Latency (trade-off) ──
        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">التوازن بين الدقة والسرعة (Accuracy ↔ Latency Trade-off)</div>', unsafe_allow_html=True)

        scatter_colors = [MODEL_COLORS.get(k, "#58A6FF") for k in bench_df_global.index
                          if k in MODEL_COLORS] or ["#58A6FF"] * len(bench_df_global)
        # نعيد البناء بمفاتيح صحيحة
        bench_keys = ["tfidf", "fasttext", "bilstm", "camelbert"]
        scatter_colors = [MODEL_COLORS.get(k, "#58A6FF") for k in bench_keys]

        fig3 = go.Figure()
        for i, (_, row) in enumerate(bench_df_global.iterrows()):
            color = scatter_colors[i] if i < len(scatter_colors) else "#58A6FF"
            fig3.add_trace(go.Scatter(
                x=[row["avg_latency_ms"]],
                y=[row["accuracy"]],
                mode="markers+text",
                marker=dict(size=16, color=color, line=dict(width=2, color="rgba(255,255,255,.2)")),
                text=[row["model_name"].split(" (")[0].split(" +")[0]],
                textposition="top center",
                textfont=dict(size=11, color="#C9D1D9", family=FONT_FAM),
                name=row["model_name"],
                showlegend=False,
            ))

        fig3.update_layout(
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=PAPER_BG,
            font=dict(family=FONT_FAM, color=TEXT_CLR),
            xaxis=dict(
                title="CPU Latency (ms) — مقياس لوغاريتمي",
                type="log",
                showgrid=True, gridcolor=GRID_CLR,
                zeroline=False,
                tickfont=dict(size=10),
                title_font=dict(size=11),
            ),
            yaxis=dict(
                title="Accuracy",
                range=[0.65, 1.0],
                showgrid=True, gridcolor=GRID_CLR,
                zeroline=False,
                tickformat=".0%",
                tickfont=dict(size=10),
                title_font=dict(size=11),
            ),
            margin=dict(l=50, r=20, t=20, b=50),
            height=320,
        )
        # منطقة "الأفضل" (سريع + دقيق)
        fig3.add_annotation(
            x=0.5, y=0.97, xref="paper", yref="paper",
            text="الأفضل: دقيق وسريع →",
            showarrow=False,
            font=dict(size=10, color="rgba(255,255,255,.2)", family=FONT_FAM),
            xanchor="right",
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════
# TAB 3 — Custom Live Trainer
# ══════════════════════════════════════════════════════════════
with tab3:

    st.markdown(f"""
    <div class="status-box info">
        📌 &nbsp; ارفع ملف CSV عربي بين <b>{TRAINER_MIN_ROWS}</b> و <b>{TRAINER_MAX_ROWS}</b> صف،
        يحتوي على عمود نص وعمود تصنيف بفئتين على الأقل.
    </div>
    """, unsafe_allow_html=True)

    custom_file = st.file_uploader("ارفع ملف التدريب (CSV)", type=["csv"], key="trainer_csv")

    if custom_file is not None:
        user_df = pd.read_csv(custom_file)
        st.markdown(
            f'<div class="status-box success">✅ تم التحميل — <b>{len(user_df)}</b> صف، <b>{len(user_df.columns)}</b> عمود</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(user_df.head(5), use_container_width=True, hide_index=True)

        tc1, tc2 = st.columns(2)
        with tc1:
            t_col = st.selectbox("عمود النص", user_df.columns, key="tr_text")
        with tc2:
            default_lbl = user_df.columns[1] if len(user_df.columns) > 1 else user_df.columns[0]
            l_col = st.selectbox("عمود التصنيف", user_df.columns,
                                 index=list(user_df.columns).index(default_lbl), key="tr_label")

        if st.button("🚀 ابدأ التدريب اللحظي والمقارنة", type="primary", use_container_width=True):
            try:
                with st.spinner("جاري التحقق من البيانات، استخراج الـ Embeddings، تدريب النموذجين ..."):
                    tok, enc = get_frozen_camelbert_encoder()
                    trained_bundle = train_dual_custom_models(user_df, t_col, l_col, tok, enc)
                    st.session_state["custom_trained_bundle"] = trained_bundle

                st.markdown(
                    f'<div class="status-box success">'
                    f'✅ تم التدريب على <b>{trained_bundle["dataset_info"]["train_rows"]}</b> صف '
                    f'| التقييم على <b>{trained_bundle["dataset_info"]["val_rows"]}</b> صف'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            except GuardrailError as ge:
                st.markdown(f'<div class="status-box error">⚠️ {ge}</div>', unsafe_allow_html=True)
            except Exception as ex:
                st.markdown(f'<div class="status-box error">❌ حدث خطأ غير متوقع: {ex}</div>', unsafe_allow_html=True)

    # ── نتائج التدريب المخصص ──────────────────────────────────
    if "custom_trained_bundle" in st.session_state:
        bundle = st.session_state["custom_trained_bundle"]

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 مقارنة النموذجين على بياناتك</div>', unsafe_allow_html=True)

        tfidf_r  = bundle["tfidf"]
        camel_r  = bundle["camelbert_emb"]

        # جدول مقارنة مرئي
        t_win = tfidf_r["accuracy"] >= camel_r["accuracy"]
        c_win = not t_win

        def _highlight(val, is_winner):
            return f'<b style="color:{"#3FB950" if is_winner else "var(--text-primary)"};">{val}</b>'

        cmp_html = f"""
        <table class="cmp-table">
            <thead>
                <tr>
                    <th>النموذج</th>
                    <th>Accuracy</th>
                    <th>Macro F1</th>
                    <th>زمن التدريب</th>
                    <th>Latency / تويتة</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="color:#58A6FF;font-weight:600;">{tfidf_r['model_name']}</td>
                    <td>{_highlight(f"{tfidf_r['accuracy']*100:.2f}%", t_win)}</td>
                    <td>{tfidf_r['macro_f1']*100:.2f}%</td>
                    <td style="font-family:'IBM Plex Mono',monospace;">{tfidf_r['train_time_sec']} ث</td>
                    <td style="font-family:'IBM Plex Mono',monospace;">{tfidf_r['avg_latency_ms']} ms</td>
                </tr>
                <tr>
                    <td style="color:#3FB950;font-weight:600;">{camel_r['model_name']}</td>
                    <td>{_highlight(f"{camel_r['accuracy']*100:.2f}%", c_win)}</td>
                    <td>{camel_r['macro_f1']*100:.2f}%</td>
                    <td style="font-family:'IBM Plex Mono',monospace;">{camel_r['train_time_sec']} ث</td>
                    <td style="font-family:'IBM Plex Mono',monospace;">{camel_r['avg_latency_ms']} ms</td>
                </tr>
            </tbody>
        </table>"""
        st.markdown(cmp_html, unsafe_allow_html=True)

        # مقارنة بصرية بشكل شريطين
        st.markdown("<div style='height:.75rem;'></div>", unsafe_allow_html=True)
        fig_cmp = go.Figure()
        metrics    = ["Accuracy", "Macro F1"]
        tfidf_vals = [tfidf_r["accuracy"], tfidf_r["macro_f1"]]
        camel_vals = [camel_r["accuracy"], camel_r["macro_f1"]]

        fig_cmp.add_trace(go.Bar(
            name=tfidf_r["model_name"],
            x=metrics, y=tfidf_vals,
            marker_color="#58A6FF", marker_line_width=0,
            text=[f"{v*100:.2f}%" for v in tfidf_vals],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono, monospace", size=11, color="#8B949E"),
        ))
        fig_cmp.add_trace(go.Bar(
            name=camel_r["model_name"],
            x=metrics, y=camel_vals,
            marker_color="#3FB950", marker_line_width=0,
            text=[f"{v*100:.2f}%" for v in camel_vals],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono, monospace", size=11, color="#8B949E"),
        ))
        fig_cmp.update_layout(
            barmode="group", bargap=0.4, bargroupgap=0.1,
            plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
            font=dict(family="IBM Plex Mono, monospace", color="#8B949E"),
            xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#C9D1D9")),
            yaxis=dict(range=[0, 1.15], showgrid=True, gridcolor="#21262D",
                       tickformat=".0%", zeroline=False, tickfont=dict(size=10)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                        font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=20, t=36, b=0),
            height=260,
        )
        st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})

        # ── اختبار فوري ──────────────────────────────────────
        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧪 اختبر نموذجك الجديد</div>', unsafe_allow_html=True)

        test_input = st.text_input(
            "اكتب جملة لاختبار النموذجين المتدربين على بياناتك",
            placeholder="اكتب هنا ...",
            label_visibility="collapsed",
        )
        if test_input.strip():
            tok2, enc2 = get_frozen_camelbert_encoder()
            custom_preds = predict_custom_tweet(test_input, bundle, tok2, enc2)

            cards2 = '<div class="mini-result-grid">'
            custom_keys = ["tfidf", "camelbert"]
            for p, k in zip(custom_preds, custom_keys):
                cls = label_class(p["label"])
                bar_color = {"pos": "#3FB950", "neg": "#F85149", "neu": "#58A6FF"}[cls]
                conf_pct = p["confidence"] * 100
                color = MODEL_COLORS.get(k, "#58A6FF")
                cards2 += f"""
                <div class="result-card">
                    <div class="result-card-name" style="color:{color};">{MODEL_SHORT.get(k, k)}</div>
                    <div class="result-label-{cls}">{label_ar(p["label"])}</div>
                    <div class="result-conf-bar-wrap">
                        <div class="result-conf-bar" style="width:{conf_pct:.1f}%;background:{bar_color};"></div>
                    </div>
                    <div class="result-meta">
                        <span>ثقة {conf_pct:.1f}%</span>
                        <span>{p['latency_ms']} ms</span>
                    </div>
                </div>"""
            cards2 += "</div>"
            st.markdown(cards2, unsafe_allow_html=True)