# CLIP Image Embedding Report

## Summary

- Input dataset: `E:\rent-agent\dataset\train_ready_multimodal.parquet`
- Output parquet: `E:\rent-agent\dataset\clip_image_embeddings_smoketest.parquet`
- Encoder: `open_clip` ViT-B-16 / `laion2b_s34b_b88k`
- Image processing: model-native OpenCLIP preprocess
- Listing representation: mean pooled, max pooled, and mean+max concatenated embeddings
- Device: **cpu**
- Batch size: **64**
- Max images per listing: **16**
- Sure: **00:05:01** (301.46 saniye)

## Metrics

- Toplam ilan sayisi: **100**
- Embedding cikarilan ilan sayisi: **100**
- Skip edilen ilan sayisi: **0**
- Toplam kullanilan gorsel sayisi: **1,516**
- Ortalama kullanilan gorsel sayisi: **15.16**
- Embedding dimension: **512**
- Mean+max dimension: **1024**
- Hata veren / okunamayan gorsel sayisi: **0**