import time
from urllib.parse import urlparse, parse_qs

import pandas as pd
from google_play_scraper import Sort, reviews

# ✅ LINK APLIKASI PLAY STORE (SUDAH DIISI)
PLAYSTORE_URL = "https://play.google.com/store/apps/details?id=gov.disdukcapilkotabdg.salaman"

# OUTPUT
OUTPUT_CSV = "salaman_reviews.csv"

# SETTING
TARGET_COUNT = 1500     # target ambil review (sebelum buang netral)
LANG = "id"
COUNTRY = "id"
SLEEP_MS = 250          # jeda per request (ms)


def extract_app_id(playstore_url: str) -> str:
    qs = parse_qs(urlparse(playstore_url).query)
    app_id = qs.get("id", [None])[0]
    if not app_id:
        raise ValueError("URL tidak valid. Pastikan ada parameter '?id=' di link Play Store.")
    return app_id


def rating_to_label(score: int):
    # Label biner TA-13:
    # 1-2 = Negatif (0), 4-5 = Positif (1), 3 = Netral (dibuang)
    if score <= 2:
        return 0
    if score >= 4:
        return 1
    return None


def scrape_reviews(app_id: str, count: int) -> pd.DataFrame:
    all_rows = []
    token = None
    fetched = 0

    while fetched < count:
        batch_size = min(200, count - fetched)  # max per request

        result, token = reviews(
            app_id,
            lang=LANG,
            country=COUNTRY,
            sort=Sort.NEWEST,
            count=batch_size,
            continuation_token=token
        )

        if not result:
            break

        for r in result:
            score = int(r.get("score", 0))
            label = rating_to_label(score)

            all_rows.append({
                "app_id": app_id,
                "reviewId": r.get("reviewId"),
                "userName": r.get("userName"),
                "score": score,
                "content": r.get("content"),
                "thumbsUpCount": r.get("thumbsUpCount"),
                "at": r.get("at").isoformat() if r.get("at") else None,
                "replyContent": r.get("replyContent"),
                "repliedAt": r.get("repliedAt").isoformat() if r.get("repliedAt") else None,
                "appVersion": r.get("appVersion"),
                "label_sentiment": label  # 0 negatif, 1 positif, None netral
            })

        fetched += len(result)

        if token is None:
            break

        time.sleep(SLEEP_MS / 1000)

    df = pd.DataFrame(all_rows)

    # buang netral (score=3) agar biner, dan buang konten kosong
    df = df[df["label_sentiment"].notna()].copy()
    df["label_sentiment"] = df["label_sentiment"].astype(int)
    df = df[df["content"].notna() & (df["content"].astype(str).str.strip() != "")].copy()

    return df


if __name__ == "__main__":
    app_id = extract_app_id(PLAYSTORE_URL)
    print("[INFO] app_id:", app_id)

    df = scrape_reviews(app_id, TARGET_COUNT)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("[DONE] tersimpan:", OUTPUT_CSV)
    print("[DONE] jumlah data (setelah buang netral & kosong):", len(df))
    print(df.head(5).to_string(index=False))
