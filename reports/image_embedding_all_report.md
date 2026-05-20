# Image Embedding All Report

## Summary

- Input dataset: `E:\rent-agent\dataset\train_ready_multimodal.parquet`
- Output parquet: `E:\rent-agent\dataset\image_embeddings_all.parquet`
- Backbone: `torchvision.models.efficientnet_b0` with ImageNet pretrained weights
- Embedding strategy: classification head removed, per-image feature vector + listing-level mean pooling
- Image usage policy: each listing uses all readable image paths from `valid_image_paths`
- Device: **cuda (NVIDIA GeForce RTX 4060 Laptop GPU)**
- Batch size: **64**
- Sure: **01:16:24** (4583.89 saniye)

## Metrics

- Toplam multimodal ilan sayisi: **6,389**
- Embedding cikarilan ilan sayisi: **6,389**
- Skip edilen ilan sayisi: **0**
- Ortalama kullanilan gorsel sayisi: **15.56**
- Toplam kullanilan gorsel sayisi: **99,392**
- Embedding dimension: **1280**
- Bozuk / okunamayan gorsel sayisi: **0**