# CLIP Fusion Caps Ablation Results

## Ozet

- Multimodal source: `E:\rent-agent\dataset\train_ready_multimodal.parquet`
- CLIP caps embedding source: `E:\rent-agent\dataset\clip_image_embeddings_caps.parquet`
- Kaydedilen model bundle: `E:\rent-agent\models\clip_fusion_caps_ablation_model.joblib`
- Join sonucu kalan ornek sayisi: **6,389**
- Denenen image_cap degerleri: **4, 8, 12, 16**
- Denenen representationlar: **clip_mean_embedding, clip_max_embedding, clip_meanmax_embedding**
- Representation dimensionlari: **clip_mean_embedding=512, clip_max_embedding=512, clip_meanmax_embedding=1024**
- Reducer: **PCA**
- Denenen image_dim degerleri: **16, 32, 64**
- En iyi kombinasyon: **image_cap=16 + clip_meanmax_embedding + XGBRegressor + image_dim=16**

## Validation Leaderboard

| image_cap | representation | image_dim | model | validation_mae | validation_rmse | validation_r2 | validation_mape |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | clip_meanmax_embedding | 16 | XGBRegressor | 4,339.4446 | 6,518.3182 | 0.8391 | 13.2759 |
| 16 | clip_mean_embedding | 16 | XGBRegressor | 4,339.9711 | 6,542.7251 | 0.8379 | 13.2530 |
| 16 | clip_max_embedding | 16 | XGBRegressor | 4,363.7483 | 6,382.2078 | 0.8458 | 13.2709 |
| 16 | clip_meanmax_embedding | 64 | XGBRegressor | 4,381.0104 | 6,518.6060 | 0.8391 | 13.2644 |
| 16 | clip_mean_embedding | 32 | XGBRegressor | 4,398.1466 | 6,609.3382 | 0.8346 | 13.3118 |
| 16 | clip_max_embedding | 32 | XGBRegressor | 4,401.9721 | 6,476.3191 | 0.8412 | 13.2843 |
| 16 | clip_meanmax_embedding | 32 | XGBRegressor | 4,413.3627 | 6,616.7916 | 0.8342 | 13.3131 |
| 16 | clip_mean_embedding | 64 | XGBRegressor | 4,421.3614 | 6,630.8109 | 0.8335 | 13.3984 |
| 12 | clip_max_embedding | 16 | XGBRegressor | 4,429.3160 | 6,486.0357 | 0.8407 | 13.4688 |
| 16 | clip_meanmax_embedding | 16 | LightGBMRegressor | 4,451.0215 | 6,771.2851 | 0.8264 | 13.3889 |
| 12 | clip_mean_embedding | 32 | XGBRegressor | 4,453.8241 | 6,476.9580 | 0.8411 | 13.5263 |
| 16 | clip_max_embedding | 64 | XGBRegressor | 4,459.9441 | 6,532.8902 | 0.8384 | 13.4515 |
| 12 | clip_meanmax_embedding | 32 | XGBRegressor | 4,469.9022 | 6,485.1217 | 0.8407 | 13.5302 |
| 16 | clip_max_embedding | 16 | LightGBMRegressor | 4,478.5454 | 6,666.3722 | 0.8317 | 13.4104 |
| 16 | clip_meanmax_embedding | 64 | LightGBMRegressor | 4,479.6791 | 6,733.0814 | 0.8283 | 13.4480 |
| 12 | clip_mean_embedding | 16 | XGBRegressor | 4,486.9484 | 6,714.7785 | 0.8293 | 13.5684 |
| 16 | clip_meanmax_embedding | 32 | LightGBMRegressor | 4,487.1543 | 6,676.0273 | 0.8312 | 13.5783 |
| 16 | clip_mean_embedding | 32 | LightGBMRegressor | 4,488.9892 | 6,732.9644 | 0.8283 | 13.4970 |
| 16 | clip_mean_embedding | 64 | LightGBMRegressor | 4,490.7516 | 6,698.4982 | 0.8301 | 13.4509 |
| 12 | clip_max_embedding | 32 | XGBRegressor | 4,493.0495 | 6,565.5522 | 0.8368 | 13.5964 |
| 4 | clip_meanmax_embedding | 16 | XGBRegressor | 4,495.7546 | 6,590.4218 | 0.8355 | 13.6109 |
| 16 | clip_mean_embedding | 16 | LightGBMRegressor | 4,502.0020 | 6,780.3725 | 0.8259 | 13.4434 |
| 4 | clip_max_embedding | 16 | XGBRegressor | 4,511.6829 | 6,574.6024 | 0.8363 | 13.7950 |
| 12 | clip_meanmax_embedding | 64 | XGBRegressor | 4,521.5202 | 6,691.4256 | 0.8304 | 13.6352 |
| 12 | clip_mean_embedding | 16 | LightGBMRegressor | 4,534.4683 | 6,742.4341 | 0.8278 | 13.6068 |
| 16 | clip_meanmax_embedding | 16 | HistGradientBoostingRegressor | 4,534.8868 | 6,961.2816 | 0.8165 | 13.5916 |
| 16 | clip_max_embedding | 32 | LightGBMRegressor | 4,539.4762 | 6,639.4253 | 0.8331 | 13.6233 |
| 4 | clip_meanmax_embedding | 32 | XGBRegressor | 4,551.3632 | 6,716.5959 | 0.8292 | 13.7329 |
| 16 | clip_max_embedding | 64 | LightGBMRegressor | 4,551.8423 | 6,796.6189 | 0.8251 | 13.5679 |
| 12 | clip_mean_embedding | 64 | XGBRegressor | 4,553.0378 | 6,718.7543 | 0.8291 | 13.7302 |
| 12 | clip_meanmax_embedding | 16 | XGBRegressor | 4,562.9228 | 6,752.3124 | 0.8273 | 13.8024 |
| 4 | clip_mean_embedding | 16 | XGBRegressor | 4,568.5486 | 6,691.2597 | 0.8305 | 13.8570 |
| 12 | clip_mean_embedding | 16 | HistGradientBoostingRegressor | 4,571.5333 | 6,930.7691 | 0.8181 | 13.7470 |
| 16 | clip_meanmax_embedding | 64 | HistGradientBoostingRegressor | 4,572.1760 | 6,869.3512 | 0.8213 | 13.6539 |
| 16 | clip_mean_embedding | 16 | HistGradientBoostingRegressor | 4,576.0997 | 6,911.6404 | 0.8191 | 13.6470 |
| 12 | clip_mean_embedding | 32 | LightGBMRegressor | 4,579.4258 | 6,657.3677 | 0.8322 | 13.7842 |
| 8 | clip_max_embedding | 32 | XGBRegressor | 4,582.1804 | 6,618.3020 | 0.8341 | 14.0622 |
| 16 | clip_mean_embedding | 32 | HistGradientBoostingRegressor | 4,584.9153 | 6,853.5894 | 0.8221 | 13.7701 |
| 8 | clip_max_embedding | 64 | XGBRegressor | 4,585.0480 | 6,632.3642 | 0.8334 | 13.9975 |
| 4 | clip_max_embedding | 64 | XGBRegressor | 4,591.2751 | 6,698.6925 | 0.8301 | 13.9977 |
| 12 | clip_max_embedding | 64 | XGBRegressor | 4,593.4476 | 6,685.1594 | 0.8308 | 13.8742 |
| 4 | clip_meanmax_embedding | 16 | LightGBMRegressor | 4,599.2837 | 6,922.2925 | 0.8185 | 13.8410 |
| 8 | clip_max_embedding | 16 | XGBRegressor | 4,600.6722 | 6,564.5068 | 0.8368 | 14.0936 |
| 16 | clip_mean_embedding | 64 | HistGradientBoostingRegressor | 4,600.7608 | 6,931.1615 | 0.8181 | 13.7826 |
| 8 | clip_meanmax_embedding | 64 | XGBRegressor | 4,602.4259 | 6,637.8907 | 0.8331 | 13.8993 |
| 4 | clip_mean_embedding | 64 | XGBRegressor | 4,603.8006 | 6,742.5994 | 0.8278 | 13.9413 |
| 4 | clip_mean_embedding | 32 | XGBRegressor | 4,607.2978 | 6,807.5322 | 0.8245 | 13.8922 |
| 16 | clip_meanmax_embedding | 32 | HistGradientBoostingRegressor | 4,607.6327 | 6,922.0864 | 0.8186 | 13.8481 |
| 4 | clip_max_embedding | 32 | XGBRegressor | 4,609.5770 | 6,726.1354 | 0.8287 | 13.9869 |
| 12 | clip_meanmax_embedding | 64 | LightGBMRegressor | 4,611.5499 | 6,868.8258 | 0.8213 | 13.7720 |
| 12 | clip_mean_embedding | 64 | LightGBMRegressor | 4,611.7819 | 6,742.6953 | 0.8278 | 13.8501 |
| 16 | clip_max_embedding | 16 | HistGradientBoostingRegressor | 4,612.9110 | 6,806.4665 | 0.8246 | 13.7947 |
| 12 | clip_meanmax_embedding | 32 | LightGBMRegressor | 4,614.4690 | 6,809.7863 | 0.8244 | 13.8670 |
| 12 | clip_mean_embedding | 32 | HistGradientBoostingRegressor | 4,615.8359 | 6,817.3549 | 0.8240 | 13.9063 |
| 4 | clip_mean_embedding | 64 | LightGBMRegressor | 4,617.1827 | 6,954.2024 | 0.8169 | 13.9640 |
| 12 | clip_max_embedding | 64 | LightGBMRegressor | 4,618.8612 | 6,861.8363 | 0.8217 | 13.8142 |
| 4 | clip_mean_embedding | 16 | LightGBMRegressor | 4,628.8916 | 7,035.2642 | 0.8126 | 13.9150 |
| 4 | clip_meanmax_embedding | 64 | XGBRegressor | 4,631.9658 | 6,927.4091 | 0.8183 | 13.8864 |
| 16 | clip_max_embedding | 64 | HistGradientBoostingRegressor | 4,637.2330 | 6,886.5521 | 0.8204 | 13.9220 |
| 16 | clip_max_embedding | 32 | HistGradientBoostingRegressor | 4,644.5190 | 6,937.1711 | 0.8178 | 13.8842 |
| 8 | clip_mean_embedding | 64 | XGBRegressor | 4,647.5893 | 6,703.5450 | 0.8298 | 14.0369 |
| 12 | clip_max_embedding | 16 | LightGBMRegressor | 4,649.3055 | 6,891.6633 | 0.8201 | 13.9969 |
| 4 | clip_meanmax_embedding | 64 | LightGBMRegressor | 4,649.6862 | 6,934.4706 | 0.8179 | 13.9863 |
| 4 | clip_max_embedding | 16 | LightGBMRegressor | 4,650.1590 | 6,925.4807 | 0.8184 | 13.9742 |
| 8 | clip_meanmax_embedding | 64 | LightGBMRegressor | 4,652.2115 | 6,977.5309 | 0.8156 | 13.9705 |
| 8 | clip_meanmax_embedding | 32 | XGBRegressor | 4,656.0011 | 6,790.4324 | 0.8254 | 14.0666 |
| 8 | clip_meanmax_embedding | 16 | XGBRegressor | 4,657.0341 | 6,735.3551 | 0.8282 | 14.1500 |
| 8 | clip_mean_embedding | 32 | XGBRegressor | 4,658.6460 | 6,760.3667 | 0.8269 | 14.0669 |
| 12 | clip_max_embedding | 16 | HistGradientBoostingRegressor | 4,659.6723 | 6,985.0088 | 0.8152 | 14.0290 |
| 12 | clip_meanmax_embedding | 16 | LightGBMRegressor | 4,669.1729 | 6,909.8603 | 0.8192 | 13.9529 |
| 4 | clip_meanmax_embedding | 16 | HistGradientBoostingRegressor | 4,681.9784 | 7,116.1704 | 0.8082 | 14.0082 |
| 12 | clip_max_embedding | 32 | LightGBMRegressor | 4,683.0915 | 6,869.7068 | 0.8213 | 14.0620 |
| 4 | clip_mean_embedding | 64 | HistGradientBoostingRegressor | 4,687.6085 | 7,043.8367 | 0.8121 | 14.1494 |
| 8 | clip_mean_embedding | 16 | XGBRegressor | 4,691.8567 | 6,748.5659 | 0.8275 | 14.2052 |
| 4 | clip_meanmax_embedding | 32 | LightGBMRegressor | 4,704.4731 | 7,080.1844 | 0.8102 | 14.0788 |
| 12 | clip_mean_embedding | 64 | HistGradientBoostingRegressor | 4,706.1687 | 6,926.4765 | 0.8183 | 14.1355 |
| 4 | clip_mean_embedding | 32 | HistGradientBoostingRegressor | 4,707.0140 | 7,014.1871 | 0.8137 | 14.0206 |
| 4 | clip_max_embedding | 64 | LightGBMRegressor | 4,708.4780 | 6,968.4641 | 0.8161 | 14.3489 |
| 8 | clip_mean_embedding | 16 | LightGBMRegressor | 4,713.7384 | 6,982.8240 | 0.8154 | 14.2178 |
| 8 | clip_mean_embedding | 64 | LightGBMRegressor | 4,714.3337 | 6,936.9457 | 0.8178 | 14.1252 |
| 4 | clip_meanmax_embedding | 32 | HistGradientBoostingRegressor | 4,719.5347 | 7,142.7842 | 0.8068 | 14.1550 |
| 4 | clip_max_embedding | 16 | HistGradientBoostingRegressor | 4,723.0166 | 7,165.1187 | 0.8056 | 14.1818 |
| 12 | clip_max_embedding | 32 | HistGradientBoostingRegressor | 4,725.6637 | 6,892.3484 | 0.8201 | 14.1809 |
| 4 | clip_mean_embedding | 16 | HistGradientBoostingRegressor | 4,726.0664 | 7,163.4294 | 0.8057 | 14.1604 |
| 4 | clip_mean_embedding | 32 | LightGBMRegressor | 4,731.2535 | 7,046.8967 | 0.8120 | 14.1296 |
| 8 | clip_mean_embedding | 32 | LightGBMRegressor | 4,732.6059 | 6,981.1069 | 0.8154 | 14.2327 |
| 8 | clip_max_embedding | 64 | LightGBMRegressor | 4,739.6811 | 6,905.6654 | 0.8194 | 14.4702 |
| 12 | clip_meanmax_embedding | 32 | HistGradientBoostingRegressor | 4,742.8672 | 7,036.1182 | 0.8125 | 14.1687 |
| 12 | clip_meanmax_embedding | 64 | HistGradientBoostingRegressor | 4,748.2083 | 7,191.0756 | 0.8042 | 14.0849 |
| 4 | clip_max_embedding | 32 | LightGBMRegressor | 4,759.7094 | 7,059.6222 | 0.8113 | 14.3971 |
| 12 | clip_meanmax_embedding | 16 | HistGradientBoostingRegressor | 4,762.6875 | 7,249.2374 | 0.8010 | 14.1104 |
| 12 | clip_max_embedding | 64 | HistGradientBoostingRegressor | 4,769.1243 | 7,096.8879 | 0.8093 | 14.1842 |
| 8 | clip_mean_embedding | 32 | HistGradientBoostingRegressor | 4,772.3992 | 7,007.5802 | 0.8140 | 14.2526 |
| 8 | clip_max_embedding | 16 | LightGBMRegressor | 4,773.9574 | 6,984.8990 | 0.8152 | 14.4125 |
| 8 | clip_max_embedding | 32 | LightGBMRegressor | 4,777.4344 | 6,970.4294 | 0.8160 | 14.5025 |
| 8 | clip_meanmax_embedding | 32 | LightGBMRegressor | 4,792.3788 | 7,155.7636 | 0.8061 | 14.2449 |
| 8 | clip_mean_embedding | 16 | HistGradientBoostingRegressor | 4,794.0154 | 7,150.7590 | 0.8064 | 14.4085 |
| 8 | clip_meanmax_embedding | 64 | HistGradientBoostingRegressor | 4,796.4500 | 7,113.4236 | 0.8084 | 14.3213 |
| 8 | clip_mean_embedding | 64 | HistGradientBoostingRegressor | 4,796.8213 | 7,215.7207 | 0.8028 | 14.2619 |
| 4 | clip_meanmax_embedding | 64 | HistGradientBoostingRegressor | 4,799.0072 | 7,174.0573 | 0.8051 | 14.4380 |
| 8 | clip_meanmax_embedding | 16 | LightGBMRegressor | 4,800.0550 | 7,114.8017 | 0.8083 | 14.3422 |
| 8 | clip_max_embedding | 32 | HistGradientBoostingRegressor | 4,819.8062 | 7,059.3306 | 0.8113 | 14.6041 |
| 4 | clip_max_embedding | 32 | HistGradientBoostingRegressor | 4,831.1895 | 7,307.8232 | 0.7978 | 14.5281 |
| 8 | clip_meanmax_embedding | 32 | HistGradientBoostingRegressor | 4,835.8579 | 7,236.5188 | 0.8017 | 14.4951 |
| 8 | clip_max_embedding | 16 | HistGradientBoostingRegressor | 4,860.7508 | 7,125.9086 | 0.8077 | 14.6739 |
| 4 | clip_max_embedding | 64 | HistGradientBoostingRegressor | 4,866.0785 | 7,283.9118 | 0.7991 | 14.6959 |
| 8 | clip_max_embedding | 64 | HistGradientBoostingRegressor | 4,875.3431 | 7,191.6636 | 0.8041 | 14.7055 |
| 8 | clip_meanmax_embedding | 16 | HistGradientBoostingRegressor | 4,903.7323 | 7,327.0308 | 0.7967 | 14.5975 |

## Cap Bazli Skorlar

| image_cap | best_representation | best_model | best_image_dim | best_validation_mae | best_validation_rmse | best_validation_r2 | best_validation_mape |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | clip_meanmax_embedding | XGBRegressor | 16 | 4,495.7546 | 6,590.4218 | 0.8355 | 13.6109 |
| 8 | clip_max_embedding | XGBRegressor | 32 | 4,582.1804 | 6,618.3020 | 0.8341 | 14.0622 |
| 12 | clip_max_embedding | XGBRegressor | 16 | 4,429.3160 | 6,486.0357 | 0.8407 | 13.4688 |
| 16 | clip_meanmax_embedding | XGBRegressor | 16 | 4,339.4446 | 6,518.3182 | 0.8391 | 13.2759 |

## Representation Bazli Skorlar

| representation | best_image_cap | best_model | best_image_dim | best_validation_mae | best_validation_rmse | best_validation_r2 | best_validation_mape |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clip_mean_embedding | 16 | XGBRegressor | 16 | 4,339.9711 | 6,542.7251 | 0.8379 | 13.2530 |
| clip_max_embedding | 16 | XGBRegressor | 16 | 4,363.7483 | 6,382.2078 | 0.8458 | 13.2709 |
| clip_meanmax_embedding | 16 | XGBRegressor | 16 | 4,339.4446 | 6,518.3182 | 0.8391 | 13.2759 |

## Image Dim Bazli Skorlar

| image_dim | best_image_cap | best_representation | best_model | best_validation_mae | best_validation_rmse | best_validation_r2 | best_validation_mape |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 16 | clip_meanmax_embedding | XGBRegressor | 4,339.4446 | 6,518.3182 | 0.8391 | 13.2759 |
| 32 | 16 | clip_mean_embedding | XGBRegressor | 4,398.1466 | 6,609.3382 | 0.8346 | 13.3118 |
| 64 | 16 | clip_meanmax_embedding | XGBRegressor | 4,381.0104 | 6,518.6060 | 0.8391 | 13.2644 |

## Final Test Sonuclari

| Metric | Value |
| --- | --- |
| MAE | 4,346.62 |
| RMSE | 6,441.28 |
| R2 | 0.8345 |
| MAPE (%) | 12.95 |

## Matched Baseline ve Previous CLIP Best Karsilastirmasi

| row | mae | rmse | r2 | mape |
| --- | --- | --- | --- | --- |
| Matched tabular baseline | 4,700.0787 | 7,057.7587 | 0.8013 | 13.7867 |
| Previous CLIP best | 4,381.5600 | 6,359.8100 | 0.8387 | 13.1000 |
| Best CLIP cap model | 4,346.6193 | 6,441.2836 | 0.8345 | 12.9463 |
| Improvement vs matched baseline | 353.4594 | 616.4750 | 0.0332 | 0.8404 |
| Improvement vs previous CLIP best | 34.9407 | -81.4736 | -0.0042 | 0.1537 |

## District Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Akyurt | 5 | 3,156.1552 | 2,007.9992 | 14.7552 | 10.8640 | 1,148.1560 | 3.8912 |
| Pursaklar | 10 | 5,075.2632 | 4,341.7141 | 15.7168 | 12.9942 | 733.5491 | 2.7226 |
| Çankaya | 379 | 6,277.1725 | 5,561.6106 | 15.2434 | 13.6780 | 715.5619 | 1.5653 |
| Altındağ | 33 | 4,002.8446 | 3,538.2641 | 15.8675 | 13.6985 | 464.5805 | 2.1689 |
| Gölbaşı | 56 | 3,835.0616 | 3,422.2551 | 11.8777 | 9.9762 | 412.8065 | 1.9015 |
| Etimesgut | 82 | 3,632.4278 | 3,251.6871 | 11.7475 | 11.0785 | 380.7407 | 0.6690 |
| Keçiören | 142 | 3,827.3778 | 3,603.4710 | 13.6167 | 12.9012 | 223.9067 | 0.7154 |
| Sincan | 58 | 3,421.4291 | 3,443.9251 | 13.3003 | 13.6709 | -22.4960 | -0.3706 |
| Mamak | 75 | 2,936.1545 | 3,012.4807 | 10.8235 | 11.1966 | -76.3262 | -0.3730 |
| Yenimahalle | 90 | 4,086.4141 | 4,337.7071 | 11.4672 | 12.2459 | -251.2930 | -0.7788 |
| Çubuk | 15 | 2,087.9365 | 2,519.0405 | 13.8520 | 17.0009 | -431.1040 | -3.1489 |
| Polatlı | 12 | 3,214.2595 | 4,371.9486 | 16.4078 | 18.2191 | -1,157.6890 | -1.8114 |

## Price Range Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-20k TRY | 84 | 4,130.3212 | 3,929.6850 | 24.3314 | 23.0467 | 200.6362 | 1.2847 |
| 20k-30k TRY | 372 | 2,905.2472 | 2,882.7568 | 11.9290 | 11.7813 | 22.4903 | 0.1476 |
| 30k-40k TRY | 266 | 4,224.3421 | 3,872.8646 | 12.4604 | 11.4050 | 351.4776 | 1.0554 |
| 40k-50k TRY | 122 | 5,594.9134 | 5,306.8410 | 12.8582 | 12.1833 | 288.0724 | 0.6748 |
| 50k-75k TRY | 84 | 8,840.6085 | 7,048.1305 | 15.2758 | 12.2640 | 1,792.4780 | 3.0119 |
| 75k+ TRY | 31 | 17,122.9349 | 16,008.6870 | 18.5068 | 17.6356 | 1,114.2479 | 0.8712 |

## m2 Range Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-75 m2 | 152 | 3,504.8093 | 3,284.7246 | 12.3153 | 11.7161 | 220.0847 | 0.5991 |
| 75-100 m2 | 150 | 4,117.2768 | 3,939.8535 | 14.2892 | 13.9045 | 177.4233 | 0.3847 |
| 100-125 m2 | 273 | 3,954.3141 | 3,774.4189 | 13.6953 | 13.1244 | 179.8952 | 0.5709 |
| 125-150 m2 | 235 | 4,205.7464 | 3,704.1705 | 12.9367 | 11.4512 | 501.5759 | 1.4855 |
| 150-200 m2 | 103 | 8,025.6704 | 7,283.0401 | 17.2320 | 15.9679 | 742.6304 | 1.2641 |
| 200+ m2 | 46 | 10,055.0157 | 9,284.8301 | 14.1807 | 13.7027 | 770.1855 | 0.4780 |

## Used Image Count Analizi

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1-4 images | 1 | 35.1493 | 9,098.4062 | 0.0469 | 12.1312 | -9,063.2570 | -12.0843 |
| 5-8 images | 9 | 1,999.3300 | 2,009.3628 | 7.9406 | 8.7837 | -10.0328 | -0.8431 |
| 9-12 images | 43 | 5,249.6558 | 4,292.6226 | 19.1167 | 15.8128 | 957.0331 | 3.3039 |
| 13-16 images | 906 | 4,705.9726 | 4,367.1550 | 13.6070 | 12.8526 | 338.8175 | 0.7544 |

## Image Branch Ablation

- Final best-cap CLIP fusion MAE: **4346.62**
- Ablation sonrasi MAE: **5643.97**
- Ablation sonrasi RMSE: **8304.47**
- Ablation sonrasi R2: **0.7249**
- Ablation sonrasi MAPE: **16.90%**
- Ablation notu: Reduced image block standardized ve PCA ile olusturuldugu icin ablasyon testinde son image block sifira cekildi; bu yaklasim reduced image branch katkisini ayirmak icin kullanildi.

## En Yuksek Hata Yapan 20 Ilan

| listing_id | district | neighborhood | rooms | m2_gross | actual_price_try | baseline_prediction | fusion_prediction | fusion_residual | fusion_abs_error | fusion_ape_pct | abs_error_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hepsiemlak:149516-72 | Çankaya | Gaziosmanpaşa Mah. | 2+1 | 100.0000 | 99,000.0000 | 56,922.9254 | 47,343.8516 | -51,656.1484 | 51,656.1484 | 52.1779 | -9,579.0738 |
| hepsiemlak:130551-1404 | Çankaya | Yukarı Dikmen Mah. | 5+1 | 305.0000 | 149,000.0000 | 130,170.0458 | 111,101.3672 | -37,898.6328 | 37,898.6328 | 25.4353 | -19,068.6786 |
| hepsiemlak:145733-213 | Çankaya | Cevizlidere Mah. | 5+1 | 375.0000 | 78,000.0000 | 96,425.8913 | 113,562.3047 | 35,562.3047 | 35,562.3047 | 45.5927 | -17,136.4134 |
| hepsiemlak:162654-112 | Çankaya | Çiğdem Mah. | 4+1 | 200.0000 | 120,000.0000 | 74,196.3756 | 84,473.9062 | -35,526.0938 | 35,526.0938 | 29.6051 | 10,277.5306 |
| hepsiemlak:5665-1536 | Çankaya | Namık Kemal Mah. | 3+0 | 139.0000 | 83,000.0000 | 57,041.3502 | 48,879.7070 | -34,120.2930 | 34,120.2930 | 41.1088 | -8,161.6432 |
| hepsiemlak:5978-6315 | Çankaya | Cevizlidere Mah. | 4+1 | 190.0000 | 80,000.0000 | 46,478.5771 | 47,879.2344 | -32,120.7656 | 32,120.7656 | 40.1510 | 1,400.6573 |
| hepsiemlak:0-46238403 | Çankaya | Kavaklıdere Mah. | 3+2 | 150.0000 | 19,000.0000 | 55,611.1481 | 47,173.6484 | 28,173.6484 | 28,173.6484 | 148.2824 | 8,437.4997 |
| hepsiemlak:61156-6048 | Çankaya | Büyükesat Mah. | 2+1 | 170.0000 | 40,000.0000 | 62,619.8139 | 68,035.0781 | 28,035.0781 | 28,035.0781 | 70.0877 | -5,415.2642 |
| hepsiemlak:156606-360 | Keçiören | Ovacık Mah. | 4+1 | 230.0000 | 50,000.0000 | 63,554.6823 | 77,646.5859 | 27,646.5859 | 27,646.5859 | 55.2932 | -14,091.9036 |
| hepsiemlak:7677-4917 | Çankaya | Mustafa Kemal Mah. | 3+1 | 145.0000 | 75,000.0000 | 42,669.2486 | 48,382.3867 | -26,617.6133 | 26,617.6133 | 35.4902 | 5,713.1382 |
| hepsiemlak:8315-16683 | Çankaya | Kızılırmak Mah. | 4+1 | 185.0000 | 85,000.0000 | 59,649.0568 | 58,653.1406 | -26,346.8594 | 26,346.8594 | 30.9963 | -995.9162 |
| hepsiemlak:165177-42 | Çankaya | Prof. Dr. Ahmet Taner Kışlalı Mah. | 4+1 | 200.0000 | 93,000.0000 | 62,194.5871 | 68,810.7969 | -24,189.2031 | 24,189.2031 | 26.0099 | 6,616.2097 |
| hepsiemlak:16602-1909 | Çankaya | Ayrancı Mah. | 1+1 | 50.0000 | 52,000.0000 | 34,228.3998 | 28,986.7500 | -23,013.2500 | 23,013.2500 | 44.2563 | -5,241.6498 |
| hepsiemlak:38897-1578 | Çankaya | Namık Kemal Mah. | 3+1 | 155.0000 | 85,000.0000 | 93,543.4714 | 63,033.4570 | -21,966.5430 | 21,966.5430 | 25.8430 | -13,423.0715 |
| hepsiemlak:56844-1171 | Çankaya | Remzi Oğuz Arık Mah. | 1+1 | 85.0000 | 52,000.0000 | 25,329.4443 | 30,927.4902 | -21,072.5098 | 21,072.5098 | 40.5241 | 5,598.0459 |
| hepsiemlak:37602-2791 | Etimesgut | Aşağıyurtçu Mah. | 4+1 | 180.0000 | 42,000.0000 | 63,423.8385 | 61,492.4180 | 19,492.4180 | 19,492.4180 | 46.4105 | 1,931.4205 |
| hepsiemlak:31332-1974 | Keçiören | Basınevleri Mah. | 3+1 | 169.0000 | 56,000.0000 | 36,525.1977 | 37,373.8555 | -18,626.1445 | 18,626.1445 | 33.2610 | 848.6578 |
| hepsiemlak:43014-3617 | Çankaya | Oran Mah. | 4+1 | 175.0000 | 180,000.0000 | 128,045.0401 | 162,110.5781 | -17,889.4219 | 17,889.4219 | 9.9386 | 34,065.5380 |
| hepsiemlak:155679-108 | Gölbaşı | Taşpınar Mah. | 3+1 | 230.0000 | 52,000.0000 | 67,633.3731 | 69,879.7969 | 17,879.7969 | 17,879.7969 | 34.3842 | -2,246.4238 |
| hepsiemlak:9800-2550 | Çankaya | Emek Mah. | 3+1 | 145.0000 | 65,000.0000 | 39,001.4751 | 47,472.9883 | -17,527.0117 | 17,527.0117 | 26.9646 | 8,471.5132 |

## Model Durumu

- `XGBRegressor`: calisti; denenen kombinasyonlar arasinda cap=12|clip_max_embedding@16, cap=12|clip_max_embedding@32, cap=12|clip_max_embedding@64, cap=12|clip_mean_embedding@16, cap=12|clip_mean_embedding@32, cap=12|clip_mean_embedding@64, cap=12|clip_meanmax_embedding@16, cap=12|clip_meanmax_embedding@32 ...
- `LightGBMRegressor`: calisti; denenen kombinasyonlar arasinda cap=12|clip_max_embedding@16, cap=12|clip_max_embedding@32, cap=12|clip_max_embedding@64, cap=12|clip_mean_embedding@16, cap=12|clip_mean_embedding@32, cap=12|clip_mean_embedding@64, cap=12|clip_meanmax_embedding@16, cap=12|clip_meanmax_embedding@32 ...
- `HistGradientBoostingRegressor`: calisti; denenen kombinasyonlar arasinda cap=12|clip_max_embedding@16, cap=12|clip_max_embedding@32, cap=12|clip_max_embedding@64, cap=12|clip_mean_embedding@16, cap=12|clip_mean_embedding@32, cap=12|clip_mean_embedding@64, cap=12|clip_meanmax_embedding@16, cap=12|clip_meanmax_embedding@32 ...

## Sonuc Yorumu

- matched tabular baseline'a gore MAE tarafinda 353.46 TRY iyilesme var; onceki CLIP best referansina gore MAE 34.94 TRY daha iyi; en iyi validation kombinasyonu yine image_cap=16 ile geldi; yani daha dusuk cap'ler mevcut CLIP kurulumunu gecemedi; image branch ablasyonda MAE 1297.35 kadar kotulestigi icin CLIP image sinyali modele net pozitif katki veriyor.