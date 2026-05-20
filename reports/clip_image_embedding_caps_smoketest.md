# CLIP Image Embedding Caps Report

## Summary

- Input dataset: `E:\rent-agent\dataset\train_ready_multimodal.parquet`
- Output parquet: `E:\rent-agent\dataset\clip_image_embeddings_caps_smoketest.parquet`
- Encoder: `open_clip` ViT-B-16 / `laion2b_s34b_b88k`
- Image processing: model-native OpenCLIP preprocess
- Listing representation: mean pooled, max pooled, and mean+max concatenated embeddings
- Tested image caps: **4, 8, 12, 16**
- Device: **cuda (NVIDIA GeForce RTX 4060 Laptop GPU)**
- Batch size: **64**
- Sure: **00:00:13** (13.12 saniye)

## Metrics

- Toplam ilan sayisi: **20**
- Embedding cikarilan ilan sayisi: **20**
- Skip edilen ilan sayisi: **0**
- Bir kez encode edilen toplam gorsel sayisi: **302**
- Embedding dimension: **512**
- Mean+max dimension: **1024**
- Hata veren / okunamayan gorsel sayisi: **0**

## Cap Breakdown

| image_cap | extracted_listings | total_used_images | avg_used_images |
| --- | --- | --- | --- |
| 4 | 20 | 80 | 4.00 |
| 8 | 20 | 160 | 8.00 |
| 12 | 20 | 235 | 11.75 |
| 16 | 20 | 302 | 15.10 |