# CLIP Image Embedding Caps Report

## Summary

- Input dataset: `E:\rent-agent\dataset\train_ready_multimodal.parquet`
- Output parquet: `E:\rent-agent\dataset\clip_image_embeddings_caps.parquet`
- Encoder: `open_clip` ViT-B-16 / `laion2b_s34b_b88k`
- Image processing: model-native OpenCLIP preprocess
- Listing representation: mean pooled, max pooled, and mean+max concatenated embeddings
- Tested image caps: **4, 8, 12, 16**
- Device: **cuda (NVIDIA GeForce RTX 4060 Laptop GPU)**
- Batch size: **64**
- Sure: **01:12:40** (4359.80 saniye)

## Metrics

- Toplam ilan sayisi: **6,389**
- Embedding cikarilan ilan sayisi: **6,389**
- Skip edilen ilan sayisi: **0**
- Bir kez encode edilen toplam gorsel sayisi: **99,148**
- Embedding dimension: **512**
- Mean+max dimension: **1024**
- Hata veren / okunamayan gorsel sayisi: **0**

## Cap Breakdown

| image_cap | extracted_listings | total_used_images | avg_used_images |
| --- | --- | --- | --- |
| 4 | 6,389 | 25,555 | 4.00 |
| 8 | 6,389 | 51,018 | 7.99 |
| 12 | 6,389 | 75,916 | 11.88 |
| 16 | 6,389 | 99,148 | 15.52 |