import os
import json
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
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
# إعداد الصفحة وإدارة الذاكرة (16GB RAM)
# ==========================================
st.set_page_config(
    page_title="Arabic Sentiment Arena",
    page_icon="🎯",
    layout="wide"
)


@st.cache_resource(max_entries=2)
def get_sentiment_model(model_key: str):
    """
    تحميل الموديلات عند الطلب مع الاحتفاظ بحد أقصى نموذجين في الرامات.
    """
    if model_key == "tfidf":
        return TfidfWrapper(repo_id=REPO_ID)
    elif model_key == "fasttext":
        return FastTextWrapper(repo_id=REPO_ID)
    elif model_key == "bilstm":
        return BiLstmWrapper(repo_id=REPO_ID)
    elif model_key == "camelbert":
        return CamelBertWrapper(repo_id=REPO_ID)
    raise ValueError(f"Unknown model key: {model_key}")


@st.cache_resource(max_entries=1)
def get_frozen_camelbert_encoder():
    """تحميل CAMeLBERT-Mix الأساسي لاستخراج الـ Embeddings في Tab 3."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_CAMELBERT_MODEL)
    model = AutoModel.from_pretrained(BASE_CAMELBERT_MODEL)
    model.to(torch.device("cpu"))
    model.eval()
    return tokenizer, model


def format_sentiment_badge(label: str) -> str:
    lbl = label.lower()
    if lbl in {"pos", "positive", "إيجابي", "1"}:
        return f"🟢 إيجابي ({label})"
    elif lbl in {"neg", "negative", "سلبي", "0"}:
        return f"🔴 سلبي ({label})"
    return f"🔵 {label}"


# ==========================================
# العنوان الرئيسي والتبويبات الثلاثة
# ==========================================
st.title("🎯 Arabic Sentiment Arena")
st.caption("منصة مقارنة وتحليل المشاعر العربية عبر 4 مدارس من تعلم الآلة والتعلم العميق والـ Transformers")

tab1, tab2, tab3 = st.tabs([
    "⚔️ ساحة التجربة الحية (Live Arena)",
    "📊 لوحة المقارنة المرجعية (Benchmark Dashboard)",
    "⚡ درّب نموذجك لحظياً (Custom Live Trainer)"
])

# ==========================================
# TAB 1: Live Arena + Batch CSV
# ==========================================
with tab1:
    st.subheader("اختبار التويتات العربية ومقارنة النماذج")

    sample_tweets = {
        "اكتب نصك الخاص...": "",
        "مثال إيجابي بالعامية + إيموجي": "الخدمة النهاردة كانت فوق الممتاز بجد والتعامل راقي جداً 😍👏",
        "مثال سلبي بأداة نفي (اختبار المعالجة)": "للأسف التجربة مش حلوة خالص والتوصيل اتأخر ومش هكررها تاني 😡",
        "مثال ساركازم / لهجة مختلطة": "يعني بعد انتظار ساعتين الأكل يوصل بارد! شابوه بجد على الاحترافية 🙄"
    }

    selected_example = st.selectbox("اختر مثالاً جاهزاً أو اكتب تويتة جديدة:", list(sample_tweets.keys()))
    user_tweet = st.text_area(
        "نص التويتة:",
        value=sample_tweets[selected_example],
        height=100,
        placeholder="اكتب أي جملة أو تويتة باللهجة المصرية أو الخليجية أو الفصحى..."
    )

    col_mode, col_select = st.columns([1, 2])
    with col_mode:
        compare_all = st.toggle("مقارنة النماذج الأربعة معاً", value=True)
    with col_select:
        single_model_key = st.selectbox(
            "اختر النموذج:",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda k: MODEL_OPTIONS[k],
            disabled=compare_all
        )

    if st.button("تحليل المشاعر الآن", type="primary"):
        if not user_tweet.strip():
            st.warning("من فضلك اكتب نص التويتة أولاً.")
        else:
            keys_to_run = list(MODEL_OPTIONS.keys()) if compare_all else [single_model_key]
            results = []
            with st.spinner("جاري التحليل..."):
                for key in keys_to_run:
                    wrapper = get_sentiment_model(key)
                    results.append(wrapper.predict(user_tweet))

            cols = st.columns(len(results))
            for col, res in zip(cols, results):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{res['model_name']}**")
                        st.markdown(f"### {format_sentiment_badge(res['label'])}")
                        st.metric("نسبة الثقة (Confidence)", f"{res['confidence'] * 100:.1f}%")
                        st.caption(f"⏱️ زمن الاستجابة: `{res['latency_ms']} ms`")

    st.divider()
    with st.expander("📂 تحليل ملف كامل (Batch CSV Prediction)"):
        batch_file = st.file_uploader("ارفع ملف CSV يحتوي على عمود النصوص:", type=["csv"], key="batch_csv")
        if batch_file is not None:
            batch_df = pd.read_csv(batch_file)
            st.write(f"عدد الصفوف في الملف: **{len(batch_df)}** (الحد الأقصى للمعالجة الدفعية: {MAX_BATCH_PREDICT_ROWS} صف)")

            b_col1, b_col2 = st.columns(2)
            with b_col1:
                batch_text_col = st.selectbox("اختر عمود النصوص:", batch_df.columns, key="batch_col")
            with b_col2:
                batch_model_key = st.selectbox(
                    "اختر النموذج للتنبؤ الجماعي:",
                    options=list(MODEL_OPTIONS.keys()),
                    format_func=lambda k: MODEL_OPTIONS[k],
                    key="batch_model"
                )

            if st.button("تشغيل التنبؤ على الملف"):
                subset_df = batch_df.head(MAX_BATCH_PREDICT_ROWS).copy()
                texts_list = subset_df[batch_text_col].fillna("").astype(str).tolist()

                with st.spinner("جاري معالجة الملف..."):
                    model_wrapper = get_sentiment_model(batch_model_key)
                    preds = model_wrapper.predict_batch(texts_list)

                subset_df["predicted_label"] = [p["label"] for p in preds]
                subset_df["confidence"] = [p["confidence"] for p in preds]

                st.dataframe(subset_df.head(20), use_container_width=True)
                csv_bytes = subset_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="⬇️ تحميل النتائج كـ CSV",
                    data=csv_bytes,
                    file_name=f"predictions_{batch_model_key}.csv",
                    mime="text/csv"
                )

# ==========================================
# TAB 2: Benchmark Dashboard
# ==========================================
with tab2:
    st.subheader("نتائج التقييم المرجعي الموحد على نفس مجموعة الاختبار (11,520 تويتة)")

    if not os.path.exists(BENCHMARK_FILE_PATH):
        st.error(f"ملف النتائج غير موجود في المسار `{BENCHMARK_FILE_PATH}`.")
    else:
        with open(BENCHMARK_FILE_PATH, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

        bench_df = pd.DataFrame.from_dict(bench_data, orient="index").reset_index(drop=True)

        # بطاقات ملخصة لأفضل الأرقام
        best_acc_row = bench_df.loc[bench_df["accuracy"].idxmax()]
        best_f1_row = bench_df.loc[bench_df["macro_f1"].idxmax()]
        fastest_row = bench_df.loc[bench_df["avg_latency_ms"].idxmin()]

        m1, m2, m3 = st.columns(3)
        m1.metric("🏆 أعلى دقة (Accuracy)", f"{best_acc_row['accuracy'] * 100:.2f}%", best_acc_row["model_name"])
        m2.metric("🎯 أعلى Macro F1", f"{best_f1_row['macro_f1'] * 100:.2f}%", best_f1_row["model_name"])
        m3.metric("⚡ أسرع نموذج (CPU Latency)", f"{fastest_row['avg_latency_ms']} ms", fastest_row["model_name"])

        st.divider()
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            melted_scores = bench_df.melt(
                id_vars=["model_name"],
                value_vars=["accuracy", "macro_f1"],
                var_name="المقياس",
                value_name="النتيجة"
            )
            melted_scores["المقياس"] = melted_scores["المقياس"].map({
                "accuracy": "Accuracy",
                "macro_f1": "Macro F1-Score"
            })
            fig_acc = px.bar(
                melted_scores,
                x="model_name",
                y="النتيجة",
                color="المقياس",
                barmode="group",
                text_auto=".3f",
                title="مقارنة الدقة و Macro F1-Score بين المدارس الأربعة"
            )
            fig_acc.update_layout(yaxis_range=[0, 1.05], xaxis_title="", legend_title="")
            st.plotly_chart(fig_acc, use_container_width=True)

        with chart_col2:
            fig_lat = px.bar(
                bench_df.sort_values("avg_latency_ms"),
                x="avg_latency_ms",
                y="model_name",
                orientation="h",
                text_auto=".2f",
                title="متوسط زمن الاستجابة للتويتة الواحدة على CPU (ms - الأقل أسرع)"
            )
            fig_lat.update_layout(xaxis_title="Latency (ms)", yaxis_title="")
            st.plotly_chart(fig_lat, use_container_width=True)

        st.markdown("#### جدول الأرقام التفصيلي")
        st.dataframe(bench_df, use_container_width=True, hide_index=True)

# ==========================================
# TAB 3: Custom Live Auto-Trainer
# ==========================================
with tab3:
    st.subheader("درّب وقارن TF-IDF مع CAMeLBERT Embeddings على بياناتك الخاصة في ثوانٍ")
    st.info(
        f"شروط الملف: ملف CSV عربي يحتوي على عمود نص وعمود تصنيف، "
        f"بعدد صفوف بين **{TRAINER_MIN_ROWS}** و **{TRAINER_MAX_ROWS}** صف، وفئتين على الأقل."
    )

    custom_file = st.file_uploader("ارفع ملف التدريب (CSV):", type=["csv"], key="trainer_csv")

    if custom_file is not None:
        user_df = pd.read_csv(custom_file)
        st.write(f"تم تحميل الملف بنجاح — إجمالي الصفوف: **{len(user_df)}**")
        st.dataframe(user_df.head(5), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            t_col = st.selectbox("اختر عمود النص (Text):", user_df.columns, key="tr_text")
        with c2:
            default_lbl_idx = 1 if len(user_df.columns) > 1 else 0
            l_col = st.selectbox("اختر عمود التصنيف (Label):", user_df.columns, index=default_lbl_idx, key="tr_label")

        if st.button("🚀 ابدأ التدريب اللحظي والمقارنة", type="primary"):
            try:
                with st.spinner("جاري التحقق من البيانات واستخراج التمثيلات وتدريب النموذجين..."):
                    tok, enc = get_frozen_camelbert_encoder()
                    trained_bundle = train_dual_custom_models(user_df, t_col, l_col, tok, enc)
                    st.session_state["custom_trained_bundle"] = trained_bundle
                st.success(
                    f"تم التدريب بنجاح على {trained_bundle['dataset_info']['train_rows']} صف "
                    f"والاختبار على {trained_bundle['dataset_info']['val_rows']} صف!"
                )
            except GuardrailError as ge:
                st.error(f"⚠️ خطأ في بيانات الملف: {ge}")
            except Exception as ex:
                st.error(f"حدث خطأ غير متوقع أثناء التدريب: {ex}")

    if "custom_trained_bundle" in st.session_state:
        bundle = st.session_state["custom_trained_bundle"]
        st.divider()
        st.markdown("#### نتائج المقارنة على بياناتك")

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
            }
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown("#### جرّب نماذجك الجديدة الآن")
        test_custom_input = st.text_input("اكتب جملة لاختبار النموذجين المتدربين على بياناتك:")
        if test_custom_input.strip():
            tok, enc = get_frozen_camelbert_encoder()
            custom_preds = predict_custom_tweet(test_custom_input, bundle, tok, enc)
            p_cols = st.columns(2)
            for col, p in zip(p_cols, custom_preds):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{p['model_name']}**")
                        st.markdown(f"### {format_sentiment_badge(p['label'])}")
                        st.metric("نسبة الثقة", f"{p['confidence'] * 100:.1f}%")
                        st.caption(f"⏱️ زمن الاستجابة: `{p['latency_ms']} ms`")