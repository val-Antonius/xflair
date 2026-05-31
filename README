## Nutmeg Prediction (xFlair)

Prediksi berapa kali seorang pemain flair akan melakukan nutmeg dalam satu pertandingan, dengan fokus pada kompetisi internasional (WC 2022, Euro 2024, Copa America 2024) sebagai basis menuju WC 2026. Project ini dibuat untuk portfolio data science dan demo aplikasi Streamlit.

### Goals & Scope
- Tujuan utama: membangun model prediksi nutmeg per pemain per pertandingan (pre-match, no leakage).
- Output: expected nutmegs + insight performa pemain flair untuk konteks scouting/analisis.
- Scope data: kompetisi internasional (WC 2022, Euro 2024, Copa America 2024) dari StatsBomb Open Data.
- Non-goals: prediksi in-play real-time dan analisis taktis video.

### Highlight
- Target: jumlah nutmeg per pemain per pertandingan (count data).
- Pendekatan final: Player-Match XGBoost (no leakage, rolling features).
- Dataset aktif: `data/processed/player_match_features.csv` (664 rows, 17 kolom).
- Model terpilih: XGBoost dengan MAE 0.3061, RMSE 0.4319, +27.9% vs baseline.

---

## Struktur Folder

```
nutmeg_prediction/
├── app.py
├── README
├── eda_statsbomb.ipynb
├── eda_wc2022sb.ipynb
├── fe_dribbles_clean.ipynb
├── fe_player_match.ipynb
├── modelling.ipynb
├── findplayer.ipynb
├── validation.ipynb
├── data/
│   ├── raw/
│   │   ├── wc2022_events.csv
│   │   └── all_events_combined.csv
│   ├── processed/
│   │   ├── wc2022sb_dribbles.csv
│   │   ├── wc2022sb_dribbles_clean.csv
│   │   ├── flair_player_profiles.csv
│   │   ├── flair_player_profiles_all.csv
│   │   ├── master_features.csv
│   │   ├── player_match_features.csv
│   │   └── final_predictions.csv
│   ├── streamlit_assets/
│   │   ├── calibration_curve.csv
│   │   ├── error_analysis.csv
│   │   ├── top_players_expected_vs_actual.csv
│   │   └── validation_summary.csv
│   ├── xgb_model.pkl
│   └── scaler.pkl
└── docs/
	├── DESIGN.md
	└── PROJECT_CONTEXT.md
```

Catatan: folder `data/raw/`, `data/processed/`, dan `docs/` tidak dipush ke repo.

---

## Data & Sumber

Sumber utama: StatsBomb Open Data (label `dribble_nutmeg: true`).

Kompetisi yang digunakan:
- FIFA World Cup 2022
- Euro 2024
- Copa America 2024

Catatan:
- Data mentah berada di `data/raw/` dan data yang sudah dibersihkan di `data/processed/`.
- Kedua folder ini diabaikan oleh git. Ikuti langkah download/ETL sendiri (lihat referensi StatsBomb Open Data).

---

## Fitur Utama

**Player profile (historical):**
- `career_nutmeg_rate`
- `career_dribble_success_rate`

**Rolling features (no leakage):**
- `rolling_nutmeg_rate_last5`
- `rolling_dribbles_last5`

**Match context:**
- `opponent_duel_win_rate`
- `match_number`

---

## Model

**Final approach:** Player-Match XGBoost
- Target: `match_nutmegs`
- Train/test: time-series split (train s/d 30 Juni 2024, test 1-15 Juli 2024)
- MAE: 0.3061
- RMSE: 0.4319
- Improvement vs baseline: 27.9%

Model dan scaler disimpan di:
- `data/xgb_model.pkl`
- `data/scaler.pkl`

---

## Menjalankan Project

### 1) Setup Environment

Disarankan menggunakan virtual environment di root project:

```bash
python -m venv .venv
```

Aktifkan:

```bash
# Windows PowerShell
\.venv\Scripts\Activate.ps1
```

Install dependencies (buat jika belum ada):

```bash
pip install -r requirements.txt
```

Catatan: jika `requirements.txt` belum ada, jalankan notebook satu kali lalu export:

```bash
pip freeze > requirements.txt
```

### 2) Jalankan Streamlit App

```bash
streamlit run app.py
```

Jika file data belum ada, siapkan dataset dari StatsBomb Open Data, lalu letakkan di `data/raw/` dan `data/processed/` sesuai struktur folder.

---

## Notebook Utama

- `fe_dribbles_clean.ipynb`: EDA dan cleaning dribbles.
- `fe_player_match.ipynb`: Feature engineering player-match (final).
- `modelling.ipynb`: Model awal (tautology). Tidak dipakai untuk final.
- `validation.ipynb`: Evaluasi, error analysis, dan visualisasi.

---

## Output Penting

- `data/processed/player_match_features.csv`: dataset final untuk training.
- `data/processed/final_predictions.csv`: hasil prediksi per player-match.
- `data/streamlit_assets/`: ringkasan visual untuk Streamlit.

Catatan: file di `data/raw/` dan `data/processed/` tidak disimpan ke repo.

---

## Lisensi

Belum ditentukan.

---

## Referensi

- StatsBomb Open Data: https://github.com/statsbomb/open-data
