import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# =========================
# INFO STUDI KASUS (SALAMAN)
# =========================
APP_NAME = "Salaman"
APP_ID = "gov.disdukcapilkotabdg.salaman"
PLAYSTORE_URL = "https://play.google.com/store/apps/details?id=gov.disdukcapilkotabdg.salaman"

MODEL_PATH = "svm_sentiment_model_salaman.joblib"


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title=f"TA-13 | SVM Sentiment - {APP_NAME}",
    page_icon="💬",
    layout="centered"
)

st.title(f"💬 TA-13 — Analisis Sentimen Ulasan {APP_NAME} (Play Store) dengan SVM")
st.caption(f"App ID: {APP_ID}")
st.link_button("Buka halaman Play Store", PLAYSTORE_URL)


# =========================
# HELPERS
# =========================
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

def pred_to_text(v: int) -> str:
    return "Positif" if int(v) == 1 else "Negatif"

def normalize_label_series(s: pd.Series) -> pd.Series:
    """
    Konversi label jadi 0/1.
    Support:
    - numeric 0/1
    - string 'Negatif'/'Positif' (atau 'negative'/'positive')
    """
    if s.dtype.kind in "biufc":
        return s.astype(int)

    s2 = s.astype(str).str.strip().str.lower()
    mapping = {
        "0": 0, "1": 1,
        "negatif": 0, "negative": 0,
        "positif": 1, "positive": 1
    }
    return s2.map(mapping)

def plot_confusion_matrix(cm, labels=("Negatif (0)", "Positif (1)"), title="Confusion Matrix"):
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    return fig


# =========================
# LOAD MODEL
# =========================
try:
    model = load_model(MODEL_PATH)
    st.success(f"Model berhasil dimuat: {MODEL_PATH}")
except Exception as e:
    st.error(f"Gagal load model: {MODEL_PATH}\n\nDetail: {e}")
    st.stop()

st.caption("Label: 0 = Negatif, 1 = Positif")


# =========================
# SINGLE PREDICTION
# =========================
st.subheader("1) Prediksi 1 Teks")
text = st.text_area(
    "Masukkan teks ulasan",
    height=120,
    placeholder="Contoh: pelayanannya cepat dan sangat membantu"
)

if st.button("Prediksi Teks", use_container_width=True):
    if not text.strip():
        st.warning("Teks masih kosong.")
    else:
        pred = int(model.predict([text])[0])
        st.write("Hasil:")
        st.write(f"**Sentimen:** {pred_to_text(pred)}")
        st.write(f"**Label (0/1):** {pred}")

st.divider()


# =========================
# BATCH PREDICTION (CSV UPLOAD)
# =========================
st.subheader("2) Uji Batch (Upload CSV)")
st.write(
    "Upload CSV berisi kolom teks ulasan (X). "
    "Jika CSV juga punya kolom label asli (y), maka Confusion Matrix & Classification Report akan ditampilkan."
)

uploaded = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, sep=";")

    st.write("Preview data:")
    st.dataframe(df.head(10), use_container_width=True)

    cols = df.columns.tolist()
    if len(cols) == 0:
        st.warning("CSV tidak memiliki kolom.")
        st.stop()

    text_col = st.selectbox("Pilih kolom teks ulasan (X)", cols)
    label_options = ["(Tidak ada label)"] + cols
    label_col = st.selectbox("Pilih kolom label asli (y) — opsional", label_options)

    if st.button("Jalankan Prediksi Batch", use_container_width=True):
        X_batch = df[text_col].astype(str).fillna("").str.strip()
        X_batch = X_batch[X_batch != ""]

        if len(X_batch) == 0:
            st.warning("Kolom teks kosong semua.")
            st.stop()

        preds = model.predict(X_batch.tolist()).astype(int)

        out = df.copy()
        out["pred_label"] = np.nan
        out.loc[X_batch.index, "pred_label"] = preds
        out["pred_label"] = out["pred_label"].astype("Int64")
        out["pred_sentiment"] = out["pred_label"].apply(lambda x: pred_to_text(int(x)) if pd.notna(x) else None)

        st.success("Prediksi batch selesai ✅")
        st.write("Hasil (preview):")
        st.dataframe(out.head(20), use_container_width=True)

        csv_bytes = out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "Download Hasil Prediksi (CSV)",
            data=csv_bytes,
            file_name=f"batch_predictions_{APP_ID}.csv",
            mime="text/csv",
            use_container_width=True
        )

        if label_col != "(Tidak ada label)":
            y_true_raw = out[label_col]
            y_true = normalize_label_series(y_true_raw)

            valid_mask = y_true.notna() & out["pred_label"].notna()
            y_true_valid = y_true[valid_mask].astype(int)
            y_pred_valid = out.loc[valid_mask, "pred_label"].astype(int)

            if len(y_true_valid) == 0:
                st.warning("Kolom label tidak bisa dipetakan ke 0/1 (atau semuanya kosong). Confusion matrix tidak bisa dihitung.")
            else:
                st.subheader("3) Evaluasi (Confusion Matrix & Report)")
                acc = accuracy_score(y_true_valid, y_pred_valid)
                st.write(f"**Accuracy:** {acc:.4f}")

                cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1])
                fig = plot_confusion_matrix(cm, title=f"Confusion Matrix (Batch) - {APP_NAME}")
                st.pyplot(fig)

                st.write("Classification Report:")
                report = classification_report(y_true_valid, y_pred_valid, digits=4, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
        else:
            st.info("Confusion Matrix belum bisa ditampilkan karena kamu belum memilih kolom label asli (y).")

st.divider()
st.caption("Confusion Matrix hanya muncul jika CSV berisi label asli (0/1 atau Positif/Negatif).")
