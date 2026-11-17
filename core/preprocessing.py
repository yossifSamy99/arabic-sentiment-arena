import re
import pandas as pd

# أدوات النفي المستثناة تماماً من الحذف حتى لا ينعكس معنى الجملة
NEGATION_WORDS = {"لا", "مش", "غير", "ما", "لن", "لم", "ليس", "بلا"}

# قائمة Stopwords بعد توحيد حروفها لتتطابق مع النص المُطبّع
RAW_STOPWORDS = {
    "في", "من", "على", "إلى", "الى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "كان", "كانت", "ان", "أن", "إن", "و", "او", "أو", "ثم", "جدا", "جداً",
    "هو", "هي", "هم", "هن", "نحن", "انا", "أنا", "انت", "أنت", "التي", "الذي"
}


def _normalize_arabic(text: str) -> str:
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    # حذف التشكيل وحرف التطويل (الكشيدة)
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)
    return text


NORMALIZED_STOPWORDS = {_normalize_arabic(w) for w in RAW_STOPWORDS} - NEGATION_WORDS


def clean_classical(text: str) -> str:
    """
    مخصصة لـ: TF-IDF و FastText و Bi-LSTM.
    - تحذف الروابط، المنشنات، الأرقام، والرموز غير العربية.
    - توحد الحروف وتحذف التشكيل والتطويل.
    - تحذف Stopwords مع الإبقاء التام على أدوات النفي.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # 1. حذف الروابط والمنشنات
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)

    # 2. توحيد الحروف العربية وحذف التشكيل والتطويل
    text = _normalize_arabic(text)

    # 3. الإبقاء على الحروف العربية والمسافات فقط (حذف الأرقام والإنجليزية والرموز)
    text = re.sub(r"[^\u0621-\u064A\s]", " ", text)

    # 4. حذف Stopwords مع استثناء أدوات النفي
    tokens = [
        word for word in text.split()
        if word in NEGATION_WORDS or (word not in NORMALIZED_STOPWORDS and len(word) > 1)
    ]
    return " ".join(tokens)


def clean_transformer(text: str) -> str:
    """
    مخصصة لـ: CAMeLBERT-Mix.
    - تحذف الروابط والمنشنات والمسافات المكررة فقط.
    - تحافظ على الإيموجيز وعلامات الترقيم والـ Stopwords لفهم السياق واللهجات.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text