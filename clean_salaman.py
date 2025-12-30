import re
import pandas as pd

INPUT_CSV = "salaman_reviews.csv"
OUTPUT_CSV = "salaman_reviews_clean.csv"

TEXT_COL = "content"
LABEL_COL = "label_sentiment"   # dari hasil scraping
MIN_WORDS = 2
DROP_DUPLICATES = True

STOPWORDS_FALLBACK = {
    "yang","dan","di","ke","dari","ini","itu","ada","atau","untuk","dengan","pada","saya","aku",
    "kamu","dia","mereka","kami","kita","nya","lah","kok","sih","nih","ya","iya","aja","deh",
    "dong","min","admin","aplikasi","app","ga","gak","nggak","tdk","tidak","bukan","udah",
    "sudah","belum","bgt","banget","bisa","dapat","dapet","kalo","kalau","karena","jadi",
    "sebagai","juga","lagi","lg","pun","akan","lebih","masih","sangat","harus","mau","mohon",
    "tolong","terima","kasih","terimakasih","halo","selamat"
}

HAS_SASTRAWI = False
stemmer = None
stopwords = set(STOPWORDS_FALLBACK)

try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    stop_factory = StopWordRemoverFactory()
    stopwords = set(stop_factory.get_stop_words()) | set(STOPWORDS_FALLBACK)

    stem_factory = StemmerFactory()
    stemmer = stem_factory.create_stemmer()

    HAS_SASTRAWI = True
except Exception:
    HAS_SASTRAWI = False


def basic_clean(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # hapus url
    text = re.sub(r"\S+@\S+", " ", text)           # hapus email
    text = re.sub(r"[^a-z\s]", " ", text)          # hapus angka, emoji, tanda baca
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    if not text:
        return ""
    tokens = [t for t in text.split() if t not in stopwords and len(t) > 1]
    return " ".join(tokens)


def do_stemming(text: str) -> str:
    if not text or stemmer is None:
        return text
    return stemmer.stem(text)


def clean_pipeline(text: str) -> str:
    text = basic_clean(text)
    text = remove_stopwords(text)
    text = do_stemming(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    df = pd.read_csv(INPUT_CSV)
    print("Loaded:", INPUT_CSV, "| shape:", df.shape)
    print("Pakai Sastrawi:", HAS_SASTRAWI)
    print("Kolom:", df.columns.tolist())

    if TEXT_COL not in df.columns:
        raise ValueError(f"Kolom '{TEXT_COL}' tidak ada.")
    if LABEL_COL not in df.columns:
        raise ValueError(f"Kolom '{LABEL_COL}' tidak ada.")

    df = df[df[LABEL_COL].notna()].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df = df[df[LABEL_COL].isin([0, 1])].copy()

    df["text"] = df[TEXT_COL].apply(clean_pipeline)

    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    df = df[(df["text"].astype(str).str.strip() != "") & (df["word_count"] >= MIN_WORDS)].copy()

    if DROP_DUPLICATES:
        df = df.drop_duplicates(subset=["text", LABEL_COL]).copy()

    out = df[["text", LABEL_COL]].rename(columns={LABEL_COL: "label"})
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("DONE ✅")
    print("Output:", OUTPUT_CSV)
    print("Jumlah data:", len(out))
    print("Distribusi label:")
    print(out["label"].value_counts())

    print("\nContoh 5 baris (RAW -> CLEAN):")
    for i in range(min(5, len(df))):
        raw = str(df.iloc[i][TEXT_COL])
        clean = str(df.iloc[i]["text"])
        print("-" * 70)
        print("RAW  :", raw[:300])
        print("CLEAN:", clean[:300])


if __name__ == "__main__":
    main()
