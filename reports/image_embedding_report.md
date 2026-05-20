# Image Embedding Report

## Summary

- Input dataset: `E:\rent-agent\dataset\train_ready_multimodal.parquet`
- Output parquet: `E:\rent-agent\dataset\image_embeddings.parquet`
- Backbone: `torchvision.models.efficientnet_b0` with ImageNet pretrained weights
- Embedding strategy: classification head removed, per-image feature vector + listing-level mean pooling
- Device: **cuda (NVIDIA GeForce RTX 4060 Laptop GPU)**
- Batch size: **64**
- Max images per listing: **6**

## Metrics

- Toplam multimodal ilan sayisi: **6,389**
- Embedding cikarilan ilan sayisi: **6,389**
- Skip edilen ilan sayisi: **0**
- Ortalama kullanilan gorsel sayisi: **6.00**
- Embedding dimension: **1280**
- Bozuk / okunamayan gorsel sayisi: **0**
- Toplam kullanilan gorsel sayisi: **38,312**