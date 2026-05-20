# SigLIP Image Embedding Report

## Summary

- Input dataset: `E:\rent-agent\dataset\train_ready_multimodal.parquet`
- Output parquet: `E:\rent-agent\dataset\siglip_image_embeddings_smoketest.parquet`
- Encoder: `transformers` / `google/siglip-base-patch16-224`
- Image processing: SigLIP AutoProcessor preprocess
- Listing representation: mean pooled, max pooled, and mean+max concatenated embeddings
- Device: **cuda (NVIDIA GeForce RTX 4060 Laptop GPU)**
- Batch size: **64**
- Max images per listing: **16**
- Sure: **00:00:18** (18.26 saniye)

## Metrics

- Toplam ilan sayisi: **20**
- Embedding cikarilan ilan sayisi: **20**
- Skip edilen ilan sayisi: **0**
- Toplam kullanilan gorsel sayisi: **302**
- Ortalama kullanilan gorsel sayisi: **15.10**
- Embedding dimension: **768**
- Mean+max dimension: **1536**
- Hata veren / okunamayan gorsel sayisi: **0**