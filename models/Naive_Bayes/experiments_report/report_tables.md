
### T_cfg: cấu hình

| Mô hình | n-gram | min_df | α | fit_prior | Số đặc trưng (train / train+val) |
|---|---|---|---|---|---|
| BoW, chọn theo val | (1, 2) | 2 | 0.5 | False | 4204 / 5151 |
| TF-IDF, chọn theo val (R0) | (1, 2) | 2 | 0.01 | False | 4204 / 5151 |
| BoW, chọn theo CV | (1, 2) | 2 | 0.3 | False | 4204 / 5151 |
| TF-IDF, chọn theo CV (R1) | (1, 2) | 3 | 0.1 | False | 2256 / 2746 |

### T_tvt: macro-F1 / accuracy

| Mô hình | Train | Val | Test, fit trên train | Test, fit trên train+val |
|---|---|---|---|---|
| BoW, chọn theo val | 0.8982 / 0.9142 | 0.6041 / 0.6861 | 0.5412 / 0.6725 | 0.5698 / 0.6969 |
| TF-IDF, chọn theo val (R0) | 0.9525 / 0.9598 | 0.6147 / 0.6958 | 0.5440 / 0.6760 | 0.5618 / 0.6934 |
| BoW, chọn theo CV | 0.9116 / 0.9242 | 0.5982 / 0.6796 | 0.5499 / 0.6760 | 0.5692 / 0.6864 |
| TF-IDF, chọn theo CV (R1) | 0.8878 / 0.9027 | 0.5813 / 0.6505 | 0.5290 / 0.6411 | 0.5395 / 0.6585 |

### T_b_test

| Mô hình | Accuracy | Macro-P | Macro-R | Macro-F1 |
|---|---|---|---|---|
| BoW, chọn theo val | 0.6969 | 0.5812 | 0.5621 | 0.5698 |
| TF-IDF, chọn theo val (R0) | 0.6934 | 0.5748 | 0.5540 | 0.5618 |
| BoW, chọn theo CV | 0.6864 | 0.5720 | 0.5669 | 0.5692 |
| TF-IDF, chọn theo CV (R1) | 0.6585 | 0.5354 | 0.5444 | 0.5395 |

### T_b_cmp

| a | b | f1_a | f1_b | delta | ci95_lo | ci95_hi | p_boot | a_right_b_wrong | a_wrong_b_right | p_mcnemar_exact |
|---|---|---|---|---|---|---|---|---|---|---|
| BoW_val|trainval | TFIDF_val|trainval | 0.5698 | 0.5618 | 0.008 | -0.0368 | 0.0525 | 0.7404 | 12 | 11 | 1.0 |
| BoW_cv|trainval | TFIDF_cv|trainval | 0.5692 | 0.5395 | 0.0297 | -0.0207 | 0.0787 | 0.2478 | 22 | 14 | 0.243 |

### T_film

| Mô hình | Accuracy | Macro-P | Macro-R | Macro-F1 | CI95 |
|---|---|---|---|---|---|
| Luôn đoán Positive | 0.6585 |  |  | 0.2647 |  |
| R0, fit trên train | 0.6760 | 0.5602 | 0.5329 | 0.5440 | [0.4723, 0.6127] |
| R0, fit trên train+val | 0.6934 | 0.5748 | 0.5540 | 0.5618 | [0.4907, 0.6295] |
| R1, fit trên train+val | 0.6585 | 0.5354 | 0.5444 | 0.5395 | [0.4699, 0.6051] |
| BoW chọn theo val, fit trên train+val | 0.6969 | 0.5812 | 0.5621 | 0.5698 | [0.4990, 0.6376] |
| BoW chọn theo CV, fit trên train+val | 0.6864 | 0.5720 | 0.5669 | 0.5692 | [0.4997, 0.6363] |
| `X_old_bal`, fit trên train+val | 0.7038 | 0.6045 | 0.5656 | 0.5789 | [0.5066, 0.6458] |
| R0v2, fit trên train+val | 0.6969 | 0.5979 | 0.5668 | 0.5772 | [0.5076, 0.6432] |
| **NB cuối (V2)**, fit trên train+val | 0.6934 | 0.5872 | 0.5962 | 0.5899 | [0.5220, 0.6537] |

### T_final_perclass

| Lớp | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.444 | 0.538 | 0.487 | 52 |
| Neutral | 0.488 | 0.457 | 0.472 | 46 |
| Positive | 0.829 | 0.794 | 0.811 | 189 |

### T_cv

| Biến thể | Đối chứng | Macro-F1 CV | Δ | CI95 của Δ | Kết luận |
|---|---|---|---|---|---|
| Tune nested CV (R1) | R0 | 0.6128 ± 0.0065 | +0.0200 | [+0.0079, +0.0327] | Chấp nhận |
| Trọng số count (BoW) | R1 | 0.6039 ± 0.0108 | -0.0089 | [-0.0211, +0.0028] | Không đủ bằng chứng |
| Trọng số binary | R1 | 0.6073 ± 0.0094 | -0.0055 | [-0.0165, +0.0052] | Không đủ bằng chứng |
| ComplementNB | R1 | 0.5991 ± 0.0077 | -0.0137 | [-0.0235, -0.0044] | Kém hơn |
| NB-LR | R1 | 0.5640 ± 0.0114 | -0.0488 | [-0.0669, -0.0309] | Kém hơn |
| Đầu vào `Clean Text Old` | R1 | 0.6201 ± 0.0059 | +0.0073 | [-0.0075, +0.0219] | Không đủ bằng chứng |
| Thêm token meta từ RAW | R1 | 0.6130 ± 0.0063 | +0.0002 | [-0.0052, +0.0056] | Không đủ bằng chứng |
| Đánh dấu phủ định | R1 | 0.6102 ± 0.0060 | -0.0026 | [-0.0116, +0.0058] | Không đủ bằng chứng |
| Char n-gram | R1 | 0.5989 ± 0.0045 | -0.0140 | [-0.0320, +0.0040] | Không đủ bằng chứng |
| Word + char n-gram | R1 | 0.6140 ± 0.0041 | +0.0012 | [-0.0152, +0.0175] | Không đủ bằng chứng |
| Chọn đặc trưng chi²/MI | R1 | 0.6046 ± 0.0042 | -0.0082 | [-0.0152, -0.0013] | Kém hơn |
| Class weight balanced | R1 | 0.6212 ± 0.0107 | +0.0084 | [-0.0020, +0.0181] | Không đủ bằng chứng |
| Oversample | R1 | 0.6152 ± 0.0064 | +0.0024 | [-0.0072, +0.0117] | Không đủ bằng chứng |
| Offset log-prior | R1 | 0.6079 ± 0.0054 | -0.0049 | [-0.0130, +0.0034] | Không đủ bằng chứng |
| Lọc nhãn nghi nhiễu | R1 | 0.5980 ± 0.0096 | -0.0148 | [-0.0251, -0.0046] | Kém hơn |
| Augmentation | R1 | 0.6152 ± 0.0093 | +0.0024 | [-0.0055, +0.0102] | Không đủ bằng chứng |
| Tune nested CV (R1) | R0 | 0.6128 ± 0.0065 | +0.0200 | [+0.0079, +0.0327] | Chấp nhận |
| Làm sạch lại `clean_v2` | R1 | 0.6241 ± 0.0045 | +0.0113 | [-0.0029, +0.0256] | Không đủ bằng chứng |
| `clean_v2` + class weight balanced | V1 | 0.6411 ± 0.0068 | +0.0170 | [+0.0073, +0.0269] | Chấp nhận |
| V2 + pool UIT-VSMEC | V2 | 0.6449 ± 0.0117 | +0.0038 | [-0.0043, +0.0118] | Không đủ bằng chứng |
| V2 + pool UIT-VSFC | V2 | 0.6345 ± 0.0093 | -0.0066 | [-0.0172, +0.0043] | Không đủ bằng chứng |
| V2 + cả hai pool | V2 | 0.6453 ± 0.0033 | +0.0042 | [-0.0071, +0.0153] | Không đủ bằng chứng |
| V2 + 8 nhãn sửa | V2 | 0.6386 ± 0.0101 | -0.0025 | [-0.0078, +0.0027] | Không đủ bằng chứng |

### T_v2_r0

| a | b | f1_a | f1_b | delta | ci95_lo | ci95_hi | p_boot |
|---|---|---|---|---|---|---|---|
| V2 | R0 | 0.6411 | 0.5928 | 0.0483 | 0.0306 | 0.0664 | 0.0002 |

### V2 chọn cấu hình

alpha {0.3: 13, 0.1: 10, 1.0: 2} | min_df {1.0: 21, 3.0: 2, 2.0: 2} | max_order {2.0: 25} | fit_prior {False: 15, True: 10}


### T_tv2

| Mô hình | Accuracy | Macro-P | Macro-R | Macro-F1 | CI95 |
|---|---|---|---|---|---|
| R0v2 | 0.3791 | 0.3782 | 0.3761 | 0.3644 | [0.3381, 0.3908] |
| NB_v2 | 0.4027 | 0.3970 | 0.3995 | 0.3892 | [0.3618, 0.4157] |
| Hybrid_v2 | 0.3989 | 0.3941 | 0.3994 | 0.3869 | [0.3602, 0.4133] |
| Bert_v2 | 0.5614 | 0.5625 | 0.5716 | 0.5499 | [0.5220, 0.5763] |

### T_tv2_cmp

| So sánh | Δ macro-F1 | CI95 của Δ | p Holm | A đúng B sai / A sai B đúng | p McNemar |
|---|---|---|---|---|---|
| NB_v2 so với R0v2 | +0.0249 | [+0.0049, +0.0443] | 0.035 | 104 / 73 | 0.024 |
| Hybrid_v2 so với NB_v2 | -0.0024 | [-0.0171, +0.0121] | 0.746 | 42 / 47 | 0.672 |
| Bert_v2 so với Hybrid_v2 | +0.1630 | [+0.1320, +0.1937] | <0.001 | 337 / 124 | <0.001 |

### T_film_ref

| Mô hình | Accuracy | Macro-F1 | CI95 |
|---|---|---|---|
| R0v2 | 0.6969 | 0.5772 | [0.5076, 0.6432] |
| NB_v2 | 0.6934 | 0.5899 | [0.5220, 0.6537] |
| Hybrid_v2 | 0.7003 | 0.5973 | [0.5273, 0.6621] |
| Bert_v2 | 0.7422 | 0.6486 | [0.5807, 0.7124] |

### T_bert_cv

| variant | macro_f1_mean | macro_f1_std | f1_negative | f1_neutral | f1_positive | accuracy | macro_f1_ci95_lo | macro_f1_ci95_hi |
|---|---|---|---|---|---|---|---|---|
| V2 | 0.6411 | 0.0068 | 0.5793 | 0.5502 | 0.7937 | 0.6969 | 0.6191 | 0.6625 |
| LR1 | 0.6416 | 0.0115 | 0.5663 | 0.5573 | 0.8011 | 0.702 | 0.6189 | 0.6642 |
| HV2 | 0.6481 | 0.0085 | 0.5859 | 0.5558 | 0.8025 | 0.7062 | 0.6259 | 0.6699 |
| P1 | 0.6754 | 0.0099 | 0.582 | 0.6327 | 0.8115 | 0.7258 | 0.6531 | 0.6969 |
| P2 | 0.6727 | 0.0061 | 0.5679 | 0.6201 | 0.8302 | 0.7387 | 0.6496 | 0.6961 |
| PE | 0.6978 | 0.01 | 0.6156 | 0.6473 | 0.8306 | 0.7482 | 0.6762 | 0.7193 |

### T_bert_cmp

| a | b | f1_a | f1_b | delta | ci95_lo | ci95_hi | p_boot | accepted | p_holm |
|---|---|---|---|---|---|---|---|---|---|
| P1 | HV2 | 0.6754 | 0.6481 | 0.0273 | 0.0039 | 0.0503 | 0.0212 | True | 0.0424 |
| P2 | P1 | 0.6727 | 0.6754 | -0.0027 | -0.0221 | 0.0172 | 0.7904 | False | 0.7904 |
| PE | P1 | 0.6978 | 0.6754 | 0.0224 | 0.0127 | 0.0321 | 0.0002 | True | 0.0012 |
| PE | HV2 | 0.6978 | 0.6481 | 0.0497 | 0.0309 | 0.0679 | 0.0002 | True | 0.0012 |
| P1 | V2 | 0.6754 | 0.6411 | 0.0343 | 0.0106 | 0.0579 | 0.0064 | True | 0.0192 |
| PE | V2 | 0.6978 | 0.6411 | 0.0567 | 0.0372 | 0.0765 | 0.0002 | True | 0.0012 |

### T_splitD

| Mô hình | Accuracy | Macro-P | Macro-R | Macro-F1 | CI95 macro-F1 | Macro-AUC | CI95 macro-AUC |
|---|---|---|---|---|---|---|---|
| NB BoW | 0.6866 | 0.6037 | 0.5866 | 0.5937 | [0.5374, 0.6474] | 0.7521 | [0.7098, 0.7957] |
| NB TF-IDF | 0.6986 | 0.6224 | 0.6508 | 0.6281 | [0.5762, 0.6774] | 0.8332 | [0.7983, 0.8669] |
| SVM BoW | 0.6986 | 0.6237 | 0.6102 | 0.6148 | [0.5595, 0.6669] | 0.7835 | [0.7416, 0.8244] |
| SVM TF-IDF | 0.7249 | 0.6556 | 0.6438 | 0.6469 | [0.5918, 0.6974] | 0.8150 | [0.7788, 0.8497] |

### T_splitD_cmp

| So sánh | Δ macro-F1 | CI95 của Δ | p Holm | A đúng B sai / A sai B đúng | p McNemar | Δ macro-AUC | CI95 của Δ AUC |
|---|---|---|---|---|---|---|---|
| NB BoW so với SVM BoW | -0.0211 | [-0.0726, +0.0310] | 0.818 | 36 / 41 | 0.649 | -0.0314 | [-0.0678, +0.0065] |
| NB TF-IDF so với SVM TF-IDF | -0.0189 | [-0.0634, +0.0248] | 0.818 | 26 / 37 | 0.207 | +0.0182 | [-0.0008, +0.0375] |
| NB TF-IDF so với NB BoW | +0.0344 | [-0.0074, +0.0762] | 0.459 | 30 / 25 | 0.590 | +0.0810 | [+0.0507, +0.1103] |
| SVM TF-IDF so với SVM BoW | +0.0321 | [-0.0078, +0.0730] | 0.459 | 24 / 13 | 0.099 | +0.0314 | [+0.0156, +0.0496] |

### T_splitD_pc

| Mô hình | Negative (P) | Negative (R) | Negative (F1) | Neutral (P) | Neutral (R) | Neutral (F1) | Positive (P) | Positive (R) | Positive (F1) |
|---|---|---|---|---|---|---|---|---|---|
| NB BoW | 0.521 | 0.500 | 0.510 | 0.514 | 0.432 | 0.469 | 0.777 | 0.828 | 0.802 |
| NB TF-IDF | 0.514 | 0.743 | 0.608 | 0.514 | 0.432 | 0.469 | 0.840 | 0.777 | 0.807 |
| SVM BoW | 0.550 | 0.446 | 0.493 | 0.526 | 0.568 | 0.546 | 0.795 | 0.816 | 0.805 |
| SVM TF-IDF | 0.575 | 0.622 | 0.597 | 0.586 | 0.466 | 0.519 | 0.806 | 0.844 | 0.824 |

### T_splitD_auc

| Mô hình | Negative | Neutral | Positive | Macro-AUC |
|---|---|---|---|---|
| NB BoW | 0.779 | 0.702 | 0.775 | 0.752 |
| NB TF-IDF | 0.881 | 0.761 | 0.858 | 0.833 |
| SVM BoW | 0.826 | 0.725 | 0.799 | 0.784 |
| SVM TF-IDF | 0.864 | 0.727 | 0.855 | 0.815 |

### T_splitH

| Mô hình | Accuracy | Macro-P | Macro-R | Macro-F1 | CI95 macro-F1 | Macro-AUC | CI95 macro-AUC |
|---|---|---|---|---|---|---|---|
| NB BoW | 0.6516 | 0.5287 | 0.5329 | 0.5292 | [0.4625, 0.5956] | 0.7549 | [0.6990, 0.8091] |
| NB TF-IDF | 0.6585 | 0.5612 | 0.5832 | 0.5681 | [0.5001, 0.6321] | 0.8156 | [0.7675, 0.8603] |
| SVM BoW | 0.6516 | 0.5332 | 0.5434 | 0.5356 | [0.4683, 0.6010] | 0.7500 | [0.6943, 0.8033] |
| SVM TF-IDF | 0.6899 | 0.5768 | 0.5501 | 0.5612 | [0.4901, 0.6296] | 0.7829 | [0.7332, 0.8315] |

### T_splitH_cmp

| So sánh | Δ macro-F1 | CI95 của Δ | p Holm | A đúng B sai / A sai B đúng | p McNemar | Δ macro-AUC | CI95 của Δ AUC |
|---|---|---|---|---|---|---|---|
| NB BoW so với SVM BoW | -0.0064 | [-0.0764, +0.0642] | 1.000 | 32 / 32 | 1.000 | +0.0048 | [-0.0383, +0.0493] |
| NB TF-IDF so với SVM TF-IDF | +0.0068 | [-0.0508, +0.0658] | 1.000 | 20 / 29 | 0.253 | +0.0327 | [+0.0049, +0.0605] |
| NB TF-IDF so với NB BoW | +0.0388 | [-0.0100, +0.0898] | 0.515 | 17 / 15 | 0.860 | +0.0607 | [+0.0277, +0.0935] |
| SVM TF-IDF so với SVM BoW | +0.0256 | [-0.0244, +0.0755] | 0.971 | 20 / 9 | 0.061 | +0.0329 | [+0.0106, +0.0560] |

### T_splitH_pc

| Mô hình | Negative (P) | Negative (R) | Negative (F1) | Neutral (P) | Neutral (R) | Neutral (F1) | Positive (P) | Positive (R) | Positive (F1) |
|---|---|---|---|---|---|---|---|---|---|
| NB BoW | 0.426 | 0.500 | 0.460 | 0.375 | 0.326 | 0.349 | 0.785 | 0.772 | 0.779 |
| NB TF-IDF | 0.414 | 0.558 | 0.475 | 0.457 | 0.457 | 0.457 | 0.813 | 0.735 | 0.772 |
| SVM BoW | 0.426 | 0.385 | 0.404 | 0.373 | 0.478 | 0.419 | 0.801 | 0.767 | 0.784 |
| SVM TF-IDF | 0.478 | 0.423 | 0.449 | 0.474 | 0.391 | 0.429 | 0.778 | 0.836 | 0.806 |

### T_splitH_auc

| Mô hình | Negative | Neutral | Positive | Macro-AUC |
|---|---|---|---|---|
| NB BoW | 0.745 | 0.759 | 0.760 | 0.755 |
| NB TF-IDF | 0.820 | 0.801 | 0.825 | 0.816 |
| SVM BoW | 0.730 | 0.755 | 0.766 | 0.750 |
| SVM TF-IDF | 0.764 | 0.777 | 0.808 | 0.783 |

### T_nodup

| a | b | f1_a | f1_b | delta | ci95_lo | ci95_hi | p_boot | a_right_b_wrong | a_wrong_b_right | p_mcnemar_exact | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NB BoW | SVM BoW | 0.5903 | 0.6113 | -0.021 | -0.0743 | 0.0324 | 0.4376 | 36 | 41 | 0.6488 | 0.7684 |
| NB TF-IDF | SVM TF-IDF | 0.6254 | 0.6449 | -0.0195 | -0.0651 | 0.0257 | 0.3842 | 26 | 37 | 0.2074 | 0.7684 |
| NB TF-IDF | NB BoW | 0.6254 | 0.5903 | 0.0351 | -0.0072 | 0.0779 | 0.1006 | 30 | 25 | 0.5901 | 0.4008 |
| SVM TF-IDF | SVM BoW | 0.6449 | 0.6113 | 0.0336 | -0.0063 | 0.0748 | 0.1002 | 24 | 13 | 0.0989 | 0.4008 |

### T_nodup_res

| model | accuracy | macro_f1 | ci95_lo | ci95_hi | n |
|---|---|---|---|---|---|
| NB BoW | 0.6767 | 0.5903 | 0.5343 | 0.644 | 399 |
| NB TF-IDF | 0.6892 | 0.6254 | 0.5719 | 0.6764 | 399 |
| SVM BoW | 0.6892 | 0.6113 | 0.5555 | 0.6636 | 399 |
| SVM TF-IDF | 0.7168 | 0.6449 | 0.5913 | 0.6976 | 399 |

### Cấu hình C

- D|NB BoW: {'config': {'kind': 'bow', 'max_order': 3, 'min_df': 2, 'sublinear': True, 'fit_prior': False, 'balanced': False, 'alpha': 0.3}, 'cv_macro_f1': 0.607589613353279}
- D|NB TF-IDF: {'config': {'kind': 'tfidf', 'max_order': 2, 'min_df': 1, 'sublinear': True, 'fit_prior': True, 'balanced': True, 'alpha': 0.3}, 'cv_macro_f1': 0.6252338254623264}
- D|SVM BoW: {'config': "Pipeline(steps=[('countvectorizer', CountVectorizer(ngram_range=(1, 2))),\n                ('onevsoneclassifier',\n                 OneVsOneClassifier(estimator=LinearSVC(C=0.03,\n                                                        class_weight='balanced',\n                                                        random_state=18)))])", 'published_macro_f1': 0.6147934637435395}
- D|SVM TF-IDF: {'config': "Pipeline(steps=[('tfidfvectorizer', TfidfVectorizer(ngram_range=(1, 2))),\n                ('onevsoneclassifier',\n                 OneVsOneClassifier(estimator=LinearSVC(C=0.3,\n                                                        class_weight='balanced',\n                                                        max_iter=10000,\n                                                        random_state=18)))])", 'published_macro_f1': 0.64693914003026}
- H|NB BoW: {'config': {'kind': 'bow', 'max_order': 2, 'min_df': 2, 'sublinear': True, 'fit_prior': True, 'balanced': True, 'alpha': 1.0}, 'cv_macro_f1': 0.6148167335321166}
- H|NB TF-IDF: {'config': {'kind': 'tfidf', 'max_order': 2, 'min_df': 1, 'sublinear': True, 'fit_prior': True, 'balanced': True, 'alpha': 0.3}, 'cv_macro_f1': 0.629089722367207}
- H|SVM BoW: {'config': {'ngram': 2, 'min_df': 2, 'sublinear': False, 'c': 0.03, 'cw': 'balanced'}, 'cv_macro_f1': 0.5966144135222468}
- H|SVM TF-IDF: {'config': {'ngram': 2, 'min_df': 1, 'sublinear': True, 'c': 0.3, 'cw': 'balanced'}, 'cv_macro_f1': 0.6239265502436637}

