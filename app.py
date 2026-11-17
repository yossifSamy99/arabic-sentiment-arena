import os
import json
import tempfile
from functools import lru_cache
import pandas as pd
import plotly.express as px
import gradio as gr
import torch
import spaces
from transformers import AutoTokenizer, AutoModel

from core.config import (
    REPO_ID,
    BASE_CAMELBERT_MODEL,
    BENCHMARK_FILE_PATH,
    MODEL_OPTIONS,
    MAX_BATCH_PREDICT_ROWS,
    TRAINER_MIN_ROWS,
    TRAINER_MAX_ROWS,
)
from core.wrappers import (
    TfidfWrapper,
    FastTextWrapper,
    BiLstmWrapper,
    CamelBertWrapper,
)
from core.live_trainer import (
    GuardrailError,
    train_dual_custom_models,
    predict_custom_tweet,
)


# ==========================================
# إدارة الذاكرة وتحميل النماذج عند الطلب
# ==========================================
@lru_cache(maxsize=2)
def get_sentiment_model(model_key: str):
    if model_key == "tfidf":
        return TfidfWrapper(repo_id=REPO_ID)
    elif model_key == "fasttext":
        return FastTextWrapper(repo_id=REPO_ID)
    elif model_key == "bilstm":
        return BiLstmWrapper(repo_id=REPO_ID)
    elif model_key == "camelbert":
        return CamelBertWrapper(repo_id=REPO_ID)
    raise ValueError(f"Unknown model key: {model_key}")


@lru_cache(maxsize=1)
def get_frozen_camelbert_encoder():
    tokenizer = AutoTokenizer.from_pretrained(BASE_CAMELBERT_MODEL)
    model = AutoModel.from_pretrained(BASE_CAMELBERT_MODEL)
    model.to(torch.device("cpu"))
    model.eval()
    return tokenizer, model


def format_sentiment_badge(label: str) -> str:
    lbl = str(label).lower()
    if lbl in {"pos", "positive", "إيجابي", "1"}:
        return f"🟢 إيجابي ({label})"
    elif lbl in {"neg", "negative", "سلبي", "0"}:
        return f"🔴 سلبي ({label})"
    return f"🔵 {label}"


# ==========================================
# دوال التبويب الأول: Live Arena & Batch CSV
# ==========================================
SAMPLE_TWEETS = {
    "اكتب نصك الخاص...": "",
    "مثال إيجابي بالعامية + إيموجي": "الخدمة النهاردة كانت فوق الممتاز بجد والتعامل راقي جداً 😍👏",
    "مثال سلبي بأداة نفي (اختبار المعالجة)": "للأسف التجربة مش حلوة خالص والتوصيل اتأخر ومش هكررها تاني 😡",
    "مثال ساركازم / لهجة مختلطة": "يعني بعد انتظار ساعتين الأكل يوصل بارد! شابوه بجد على الاحترافية 🙄",
}


def on_example_select(example_key: str) -> str:
    return SAMPLE_TWEETS.get(example_key, "")


def toggle_single_model_dropdown(compare_all: bool):
    return gr.Dropdown(interactive=not compare_all)


@spaces.GPU
def run_live_analysis(tweet_text: str, compare_all: bool, single_model_key: str):
    if not tweet_text or not tweet_text.strip():
        gr.Warning("من فضلك اكتب نص التويتة أولاً.")
        return "", pd.DataFrame()

    keys_to_run = list(MODEL_OPTIONS.keys()) if compare_all else [single_model_key]
    rows = []
    cards_md = []

    for key in keys_to_run:
        wrapper = get_sentiment_model(key)
        res = wrapper.predict(tweet_text)
        badge = format_sentiment_badge(res["label"])
        conf_pct = round(res["confidence"] * 100, 1)

        rows.append({
            "النموذج": res["model_name"],
            "التصنيف (Sentiment)": badge,
            "نسبة الثقة (%)": f"{conf_pct}%",
            "زمن الاستجابة (ms)": res["latency_ms"],
        })
        cards_md.append(
            f"**{res['model_name']}**: {badge} — ثقة **{conf_pct}%** (`{res['latency_ms']} ms`)"
        )

    summary_md = "\n\n".join(cards_md)
    return summary_md, pd.DataFrame(rows)


def inspect_batch_csv(file_obj):
    if file_obj is None:
        return gr.Dropdown(choices=[], value=None), ""
    file_path = file_obj if isinstance(file_obj, str) else file_obj.name
    df = pd.read_csv(file_path)
    cols = list(df.columns)
    info = f"تم تحميل الملف — إجمالي الصفوف: **{len(df)}** (الحد الأقصى للمعالجة: {MAX_BATCH_PREDICT_ROWS} صف)"
    return gr.Dropdown(choices=cols, value=cols[0] if cols else None), info


@spaces.GPU
def run_batch_prediction(file_obj, text_col: str, model_key: str):
    if file_obj is None or not text_col:
        gr.Warning("ارفع ملف CSV واختر عمود النصوص أولاً.")
        return pd.DataFrame(), None

    file_path = file_obj if isinstance(file_obj, str) else file_obj.name
    batch_df = pd.read_csv(file_path)
    subset_df = batch_df.head(MAX_BATCH_PREDICT_ROWS).copy()
    texts_list = subset_df[text_col].fillna("").astype(str).tolist()

    model_wrapper = get_sentiment_model(model_key)
    preds = model_wrapper.predict_batch(texts_list)

    subset_df["predicted_label"] = [p["label"] for p in preds]
    subset_df["confidence"] = [p["confidence"] for p in preds]

    out_path = os.path.join(tempfile.gettempdir(), f"predictions_{model_key}.csv")
    subset_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return subset_df.head(20), out_path


# ==========================================
# دوال التبويب الثاني: Benchmark Dashboard
# ==========================================
def build_benchmark_views():
    if not os.path.exists(BENCHMARK_FILE_PATH):
        return f"⚠️ ملف النتائج غير موجود في `{BENCHMARK_FILE_PATH}`.", None, None, pd.DataFrame()

    with open(BENCHMARK_FILE_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    bench_df = pd.DataFrame.from_dict(bench_data, orient="index").reset_index(drop=True)

    best_acc = bench_df.loc[bench_df["accuracy"].idxmax()]
    best_f1 = bench_df.loc[bench_df["macro_f1"].idxmax()]
    fastest = bench_df.loc[bench_df["avg_latency_ms"].idxmin()]

    kpi_md = (
        f"### ملخص الأداء على مجموعة الاختبار الموحدة (11,520 تويتة)\n"
        f"- 🏆 **أعلى دقة (Accuracy):** `{best_acc['accuracy'] * 100:.2f}%` ({best_acc['model_name']})\n"
        f"- 🎯 **أعلى Macro F1:** `{best_f1['macro_f1'] * 100:.2f}%` ({best_f1['model_name']})\n"
        f"- ⚡ **أسرع استجابة (CPU Latency):** `{fastest['avg_latency_ms']} ms` ({fastest['model_name']})"
    )

    melted = bench_df.melt(
        id_vars=["model_name"],
        value_vars=["accuracy", "macro_f1"],
        var_name="المقياس",
        value_name="النتيجة",
    )
    melted["المقياس"] = melted["المقياس"].map({
        "accuracy": "Accuracy",
        "macro_f1": "Macro F1-Score",
    })
    fig_acc = px.bar(
        melted,
        x="model_name",
        y="النتيجة",
        color="المقياس",
        barmode="group",
        text_auto=".3f",
        title="مقارنة الدقة و Macro F1-Score بين المدارس الأربعة",
    )
    fig_acc.update_layout(yaxis_range=[0, 1.05], xaxis_title="", legend_title="")

    fig_lat = px.bar(
        bench_df.sort_values("avg_latency_ms"),
        x="avg_latency_ms",
        y="model_name",
        orientation="h",
        text_auto=".2f",
        title="متوسط زمن الاستجابة للتويتة الواحدة (ms - الأقل أسرع)",
    )
    fig_lat.update_layout(xaxis_title="Latency (ms)", yaxis_title="")

    return kpi_md, fig_acc, fig_lat, bench_df


# ==========================================
# دوال التبويب الثالث: Custom Live Trainer
# ==========================================
def inspect_trainer_csv(file_obj):
    if file_obj is None:
        return pd.DataFrame(), gr.Dropdown(choices=[], value=None), gr.Dropdown(choices=[], value=None)
    file_path = file_obj if isinstance(file_obj, str) else file_obj.name
    df = pd.read_csv(file_path)
    cols = list(df.columns)
    default_text = cols[0] if cols else None
    default_label = cols[1] if len(cols) > 1 else default_text
    return (
        df.head(5),
        gr.Dropdown(choices=cols, value=default_text),
        gr.Dropdown(choices=cols, value=default_label),
    )


@spaces.GPU
def run_custom_training(file_obj, text_col: str, label_col: str):
    if file_obj is None:
        gr.Warning("من فضلك ارفع ملف CSV للتدريب أولاً.")
        return "⚠️ لم يتم رفع ملف.", pd.DataFrame(), None

    try:
        file_path = file_obj if isinstance(file_obj, str) else file_obj.name
        user_df = pd.read_csv(file_path)
        tok, enc = get_frozen_camelbert_encoder()
        bundle = train_dual_custom_models(user_df, text_col, label_col, tok, enc)

        comp_df = pd.DataFrame([
            {
                "النموذج": bundle["tfidf"]["model_name"],
                "Accuracy": bundle["tfidf"]["accuracy"],
                "Macro F1": bundle["tfidf"]["macro_f1"],
                "زمن التدريب (ثانية)": bundle["tfidf"]["train_time_sec"],
                "سرعة التنبؤ (ms)": bundle["tfidf"]["avg_latency_ms"],
            },
            {
                "النموذج": bundle["camelbert_emb"]["model_name"],
                "Accuracy": bundle["camelbert_emb"]["accuracy"],
                "Macro F1": bundle["camelbert_emb"]["macro_f1"],
                "زمن التدريب (ثانية)": bundle["camelbert_emb"]["train_time_sec"],
                "سرعة التنبؤ (ms)": bundle["camelbert_emb"]["avg_latency_ms"],
            },
        ])
        status_msg = (
            f"✅ تم التدريب بنجاح على **{bundle['dataset_info']['train_rows']}** صف "
            f"والاختبار على **{bundle['dataset_info']['val_rows']}** صف!"
        )
        return status_msg, comp_df, bundle
    except GuardrailError as ge:
        return f"⚠️ **خطأ في بيانات الملف:** {ge}", pd.DataFrame(), None
    except Exception as ex:
        return f"❌ **حدث خطأ غير متوقع:** {ex}", pd.DataFrame(), None


@spaces.GPU
def run_custom_prediction(text: str, bundle):
    if bundle is None:
        gr.Warning("درّب النماذج على ملفك أولاً قبل الاختبار.")
        return pd.DataFrame()
    if not text or not text.strip():
        gr.Warning("اكتب جملة للاختبار أولاً.")
        return pd.DataFrame()

    tok, enc = get_frozen_camelbert_encoder()
    preds = predict_custom_tweet(text, bundle, tok, enc)
    rows = [
        {
            "النموذج": p["model_name"],
            "التصنيف (Sentiment)": format_sentiment_badge(p["label"]),
            "نسبة الثقة (%)": f"{p['confidence'] * 100:.1f}%",
            "زمن الاستجابة (ms)": p["latency_ms"],
        }
        for p in preds
    ]
    return pd.DataFrame(rows)


# ==========================================
# بناء واجهة Gradio Blocks الكاملة
# ==========================================
MODEL_CHOICES = [(label, key) for key, label in MODEL_OPTIONS.items()]

with gr.Blocks(title="Arabic Sentiment Arena") as demo:
    gr.Markdown(
        "# 🎯 Arabic Sentiment Arena\n"
        "منصة مقارنة وتحليل المشاعر العربية عبر 4 مدارس من تعلم الآلة والتعلم العميق والـ Transformers"
    )

    with gr.Tabs():
        # ------------------------------------------
        # TAB 1: Live Arena
        # ------------------------------------------
        with gr.Tab("⚔️ ساحة التجربة الحية (Live Arena)"):
            example_dd = gr.Dropdown(
                choices=list(SAMPLE_TWEETS.keys()),
                value="اكتب نصك الخاص...",
                label="اختر مثالاً جاهزاً أو اكتب تويتة جديدة",
            )
            tweet_box = gr.Textbox(
                label="نص التويتة",
                lines=3,
                placeholder="اكتب أي جملة أو تويتة باللهجة المصرية أو الخليجية أو الفصحى...",
                rtl=True,
            )

            with gr.Row():
                compare_chk = gr.Checkbox(label="مقارنة النماذج الأربعة معاً", value=True)
                single_model_dd = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value="camelbert",
                    label="اختر النموذج (عند إيقاف المقارنة الشاملة)",
                    interactive=False,
                )

            analyze_btn = gr.Button("تحليل المشاعر الآن", variant="primary")
            live_summary_md = gr.Markdown()
            live_results_df = gr.Dataframe(label="نتائج التحليل التفصيلية", interactive=False)

            example_dd.change(fn=on_example_select, inputs=example_dd, outputs=tweet_box)
            compare_chk.change(fn=toggle_single_model_dropdown, inputs=compare_chk, outputs=single_model_dd)
            analyze_btn.click(
                fn=run_live_analysis,
                inputs=[tweet_box, compare_chk, single_model_dd],
                outputs=[live_summary_md, live_results_df],
            )

            with gr.Accordion("📂 تحليل ملف كامل (Batch CSV Prediction)", open=False):
                batch_file_in = gr.File(label="ارفع ملف CSV يحتوي على عمود النصوص", file_types=[".csv"])
                batch_info_md = gr.Markdown()
                with gr.Row():
                    batch_col_dd = gr.Dropdown(label="اختر عمود النصوص", choices=[])
                    batch_model_dd = gr.Dropdown(
                        label="اختر النموذج للتنبؤ الجماعي",
                        choices=MODEL_CHOICES,
                        value="camelbert",
                    )
                batch_btn = gr.Button("تشغيل التنبؤ على الملف")
                batch_out_df = gr.Dataframe(label="معاينة النتائج (أول 20 صف)", interactive=False)
                batch_dl_file = gr.File(label="⬇️ تحميل النتائج كـ CSV")

                batch_file_in.change(
                    fn=inspect_batch_csv,
                    inputs=batch_file_in,
                    outputs=[batch_col_dd, batch_info_md],
                )
                batch_btn.click(
                    fn=run_batch_prediction,
                    inputs=[batch_file_in, batch_col_dd, batch_model_dd],
                    outputs=[batch_out_df, batch_dl_file],
                )

        # ------------------------------------------
        # TAB 2: Benchmark Dashboard
        # ------------------------------------------
        with gr.Tab("📊 لوحة المقارنة المرجعية (Benchmark Dashboard)"):
            kpi_text, fig_accuracy, fig_latency, benchmark_table = build_benchmark_views()
            gr.Markdown(kpi_text)
            with gr.Row():
                gr.Plot(value=fig_accuracy)
                gr.Plot(value=fig_latency)
            gr.Dataframe(value=benchmark_table, label="جدول الأرقام التفصيلي", interactive=False)

        # ------------------------------------------
        # TAB 3: Custom Live Auto-Trainer
        # ------------------------------------------
        with gr.Tab("⚡ درّب نموذجك لحظياً (Custom Live Trainer)"):
            gr.Markdown(
                f"درّب وقارن **TF-IDF** مع **CAMeLBERT Embeddings** على بياناتك الخاصة في ثوانٍ.\n\n"
                f"**شروط الملف:** ملف CSV عربي يحتوي على عمود نص وعمود تصنيف، "
                f"بعدد صفوف بين **{TRAINER_MIN_ROWS}** و **{TRAINER_MAX_ROWS}** صف، وفئتين على الأقل."
            )

            trainer_state = gr.State(value=None)
            trainer_file_in = gr.File(label="ارفع ملف التدريب (CSV)", file_types=[".csv"])
            trainer_preview_df = gr.Dataframe(label="معاينة البيانات (أول 5 صفوف)", interactive=False)

            with gr.Row():
                tr_text_dd = gr.Dropdown(label="اختر عمود النص (Text)", choices=[])
                tr_label_dd = gr.Dropdown(label="اختر عمود التصنيف (Label)", choices=[])

            train_btn = gr.Button("🚀 ابدأ التدريب اللحظي والمقارنة", variant="primary")
            train_status_md = gr.Markdown()
            train_comp_df = gr.Dataframe(label="نتائج المقارنة على بياناتك", interactive=False)

            gr.Markdown("### جرّب نماذجك الجديدة الآن")
            custom_test_box = gr.Textbox(
                label="اكتب جملة لاختبار النموذجين المتدربين على بياناتك",
                rtl=True,
            )
            custom_test_btn = gr.Button("اختبار الجملة")
            custom_test_df = gr.Dataframe(label="نتائج التنبؤ", interactive=False)

            trainer_file_in.change(
                fn=inspect_trainer_csv,
                inputs=trainer_file_in,
                outputs=[trainer_preview_df, tr_text_dd, tr_label_dd],
            )
            train_btn.click(
                fn=run_custom_training,
                inputs=[trainer_file_in, tr_text_dd, tr_label_dd],
                outputs=[train_status_md, train_comp_df, trainer_state],
            )
            custom_test_btn.click(
                fn=run_custom_prediction,
                inputs=[custom_test_box, trainer_state],
                outputs=custom_test_df,
            )


if __name__ == "__main__":
    demo.launch()