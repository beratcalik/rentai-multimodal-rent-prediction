import pandas as pd
from pathlib import Path

DATASET_DIR = Path("dataset")
OUTPUT_FILE = DATASET_DIR / "dataset_export.xlsx"

parquet_files = [
    "listings.parquet",
    "images.parquet",
    "validation_report.parquet",
    "run_log.parquet",
    "train_ready_ml.parquet",
    "train_ready_multimodal.parquet",
]

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    for filename in parquet_files:
        file_path = DATASET_DIR / filename
        if file_path.exists():
            df = pd.read_parquet(file_path)

            sheet_name = filename.replace(".parquet", "")
            sheet_name = sheet_name[:31]  # Excel sheet isim limiti

            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Aktarıldı: {filename} -> sheet: {sheet_name}")
        else:
            print(f"Bulunamadı, atlandı: {filename}")

print(f"\nExcel oluşturuldu: {OUTPUT_FILE}")