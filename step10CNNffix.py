#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:57:29 2026======================================================================
STEP 10C-V3: MULTITASK 2D-CNN
======================================================================

Loading training CNN inputs...
Training input shape: (3196, 4, 20, 1)

Loading training volatility targets...
Training target shape: (3196,)

Loading training metadata...

Loading test data for verification only...
Test input shape: (1047, 4, 20, 1)
Test target shape: (1047,)

IMPORTANT: Test data remains untouched.

----------------------------------------------------------------------
INPUT VALIDATION
----------------------------------------------------------------------
PASS: Input dimensions and lengths are consistent.

Checking for NaN and Inf values...
PASS: No invalid values detected.

----------------------------------------------------------------------
CHRONOLOGICAL TRAIN / VALIDATION SPLIT
----------------------------------------------------------------------
Training observations:   1552
Validation observations: 1644
Training period: 2006-05-09 00:00:00 → 2013-12-31 00:00:00
Validation period: 2014-01-02 00:00:00 → 2020-12-31 00:00:00
PASS: Training and validation are chronologically separated.

----------------------------------------------------------------------
SPIKE DISTRIBUTION
----------------------------------------------------------------------
Training non-spikes: 1508
Training spikes:     44
Training spike rate: 2.8351%

Validation non-spikes: 1615
Validation spikes:     29
Validation spike rate: 1.7640%

Frozen spike threshold: 0.10927724

----------------------------------------------------------------------
VOLATILITY TARGET TRANSFORMATION
----------------------------------------------------------------------
Training original mean: 0.11388297
Training original maximum: 41.22689438
Training log1p mean: 0.02656609
Training log1p maximum: 3.74305725

----------------------------------------------------------------------
BALANCED CLASS WEIGHTS
----------------------------------------------------------------------
Non-spike weight: 0.514589
Spike weight:     17.636364

PASS: Sample weights created.

======================================================================
BUILDING MULTITASK 2D-CNN
======================================================================
Model: "multitask_2d_cnn"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ cnn_input           │ (None, 4, 20, 1)  │          0 │ -                 │
│ (InputLayer)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_block1       │ (None, 4, 20, 32) │        224 │ cnn_input[0][0]   │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ maxpool_block1      │ (None, 2, 10, 32) │          0 │ conv2d_block1[0]… │
│ (MaxPooling2D)      │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_block2       │ (None, 2, 10, 64) │     12,352 │ maxpool_block1[0… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ flatten_shared      │ (None, 1280)      │          0 │ conv2d_block2[0]… │
│ (Flatten)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ shared_dense        │ (None, 64)        │     81,984 │ flatten_shared[0… │
│ (Dense)             │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ shared_dropout      │ (None, 64)        │          0 │ shared_dense[0][… │
│ (Dropout)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ volatility_hidden   │ (None, 32)        │      2,080 │ shared_dropout[0… │
│ (Dense)             │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ spike_hidden        │ (None, 32)        │      2,080 │ shared_dropout[0… │
│ (Dense)             │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ volatility (Dense)  │ (None, 1)         │         33 │ volatility_hidde… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ spike_probability   │ (None, 1)         │         33 │ spike_hidden[0][… │
│ (Dense)             │                   │            │                   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 98,786 (385.88 KB)
 Trainable params: 98,786 (385.88 KB)
 Non-trainable params: 0 (0.00 B)

======================================================================
STARTING MULTITASK CNN TRAINING
======================================================================
Epoch 1/100
48/49 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.7533 - spike_probability_auc: 0.3335 - spike_probability_loss: 1.0427 - spike_probability_pr_auc: 0.0209 - spike_probability_precision: 0.0097 - spike_probability_recall: 0.0071 - volatility_loss: 0.0780 - volatility_mae: 0.2991                 
Epoch 1: val_spike_probability_auc improved from None to 0.34437, saving model to models/cnn_2d_multitask_model.keras

Epoch 1: finished saving model to models/cnn_2d_multitask_model.keras
49/49 ━━━━━━━━━━━━━━━━━━━━ 6s 34ms/step - loss: 0.7478 - spike_probability_auc: 0.3669 - spike_probability_loss: 1.0281 - spike_probability_pr_auc: 0.0202 - spike_probability_precision: 0.0039 - spike_probability_recall: 0.0227 - volatility_loss: 0.0780 - volatility_mae: 0.3162 - val_loss: 0.5126 - val_spike_probability_auc: 0.3444 - val_spike_probability_loss: 0.6926 - val_spike_probability_pr_auc: 0.0131 - val_spike_probability_precision: 0.0097 - val_spike_probability_recall: 0.4138 - val_volatility_loss: 0.0927 - val_volatility_mae: 0.3911 - learning_rate: 0.0010
Epoch 2/100
47/49 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.4618 - spike_probability_auc: 0.3822 - spike_probability_loss: 0.6458 - spike_probability_pr_auc: 0.0227 - spike_probability_precision: 0.0212 - spike_probability_recall: 0.1492 - volatility_loss: 0.0324 - volatility_mae: 0.1861                 
Epoch 2: val_spike_probability_auc improved from 0.34437 to 0.35817, saving model to models/cnn_2d_multitask_model.keras

Epoch 2: finished saving model to models/cnn_2d_multitask_model.keras
49/49 ━━━━━━━━━━━━━━━━━━━━ 1s 23ms/step - loss: 0.5005 - spike_probability_auc: 0.3717 - spike_probability_loss: 0.6968 - spike_probability_pr_auc: 0.0235 - spike_probability_precision: 0.0130 - spike_probability_recall: 0.1591 - volatility_loss: 0.0343 - volatility_mae: 0.1930 - val_loss: 0.5070 - val_spike_probability_auc: 0.3582 - val_spike_probability_loss: 0.7093 - val_spike_probability_pr_auc: 0.0125 - val_spike_probability_precision: 0.0144 - val_spike_probability_recall: 0.7931 - val_volatility_loss: 0.0347 - val_volatility_mae: 0.1700 - learning_rate: 0.0010
Epoch 3/100
48/49 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - loss: 0.4632 - spike_probability_auc: 0.3603 - spike_probability_loss: 0.6544 - spike_probability_pr_auc: 0.0272 - spike_probability_precision: 0.0225 - spike_probability_recall: 0.2645 - volatility_loss: 0.0170 - volatility_mae: 0.0646                     
Epoch 3: val_spike_probability_auc improved from 0.35817 to 0.53496, saving model to models/cnn_2d_multitask_model.keras

Epoch 3: finished saving model to models/cnn_2d_multitask_model.keras
49/49 ━━━━━━━━━━━━━━━━━━━━ 1s 26ms/step - loss: 0.4965 - spike_probability_auc: 0.4576 - spike_probability_loss: 0.6971 - spike_probability_pr_auc: 0.0414 - spike_probability_precision: 0.0258 - spike_probability_recall: 0.4773 - volatility_loss: 0.0204 - volatility_mae: 0.0960 - val_loss: 0.5219 - val_spike_probability_auc: 0.5350 - val_spike_probability_loss: 0.7334 - val_spike_probability_pr_auc: 0.0201 - val_spike_probability_precision: 0.0161 - val_spike_probability_recall: 0.8966 - val_volatility_loss: 0.0282 - val_volatility_mae: 0.1163 - learning_rate: 0.0010
Epoch 4/100
48/49 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.4763 - spike_probability_auc: 0.3441 - spike_probability_loss: 0.6737 - spike_probability_pr_auc: 0.0218 - spike_probability_precision: 0.0155 - spike_probability_recall: 0.2701 - volatility_loss: 0.0157 - volatility_mae: 0.0452                     
Epoch 4: val_spike_probability_auc improved from 0.53496 to 0.73142, saving model to models/cnn_2d_multitask_model.keras

Epoch 4: finished saving model to models/cnn_2d_multitask_model.keras
49/49 ━━━━━━━━━━━━━━━━━━━━ 1s 25ms/step - loss: 0.5094 - spike_probability_auc: 0.3815 - spike_probability_loss: 0.7166 - spike_probability_pr_auc: 0.0317 - spike_probability_precision: 0.0164 - spike_probability_recall: 0.3409 - volatility_loss: 0.0174 - volatility_mae: 0.0674 - val_loss: 0.5367 - val_spike_probability_auc: 0.7314 - val_spike_probability_loss: 0.7552 - val_spike_probability_pr_auc: 0.0432 - val_spike_probability_precision: 0.0177 - val_spike_probability_recall: 1.0000 - val_volatility_loss: 0.0261 - val_volatility_mae: 0.0863 - learning_rate: 0.0010
Epoch 5/100
44/49 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4583 - spike_probability_auc: 0.4133 - spike_probability_loss: 0.6480 - spike_probability_pr_auc: 0.0287 - spike_probability_precision: 0.0216 - spike_probability_recall: 0.5030 - volatility_loss: 0.0155 - volatility_mae: 0.0413                     
Epoch 5: val_spike_probability_auc did not improve from 0.73142
49/49 ━━━━━━━━━━━━━━━━━━━━ 1s 16ms/step - loss: 0.4835 - spike_probability_auc: 0.5935 - spike_probability_loss: 0.6801 - spike_probability_pr_auc: 0.0450 - spike_probability_precision: 0.0282 - spike_probability_recall: 0.7500 - volatility_loss: 0.0172 - volatility_mae: 0.0654 - val_loss: 0.5209 - val_spike_probability_auc: 0.6714 - val_spike_probability_loss: 0.7331 - val_spike_probability_pr_auc: 0.0405 - val_spike_probability_precision: 0.0195 - val_spike_probability_recall: 0.9310 - val_volatility_loss: 0.0254 - val_volatility_mae: 0.0723 - learning_rate: 0.0010
Epoch 6/100
47/49 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.4375 - spike_probability_auc: 0.4900 - spike_probability_loss: 0.6184 - spike_probability_pr_auc: 0.0484 - spike_probability_precision: 0.0332 - spike_probability_recall: 0.3727 - volatility_loss: 0.0153 - volatility_mae: 0.0306                     
Epoch 6: val_spike_probability_auc improved from 0.73142 to 0.74393, saving model to models/cnn_2d_multitask_model.keras

Epoch 6: finished saving model to models/cnn_2d_multitask_model.keras
49/49 ━━━━━━━━━━━━━━━━━━━━ 1s 25ms/step - loss: 0.4691 - spike_probability_auc: 0.6157 - spike_probability_loss: 0.6604 - spike_probability_pr_auc: 0.0766 - spike_probability_precision: 0.0306 - spike_probability_recall: 0.6364 - volatility_loss: 0.0159 - volatility_mae: 0.0425 - val_loss: 0.5829 - val_spike_probability_auc: 0.7439 - val_spike_probability_loss: 0.8215 - val_spike_probability_pr_auc: 0.0445 - val_spike_probability_precision: 0.0176 - val_spike_probability_recall: 0.9655 - val_volatility_loss: 0.0247 - val_volatility_mae: 0.0511 - learning_rate: 0.0010
Epoch 7/100
43/49 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4482 - spike_probability_auc: 0.4963 - spike_probability_loss: 0.6338 - spike_probability_pr_auc: 0.0391 - spike_probability_precision: 0.0332 - spike_probability_recall: 0.5751 - volatility_loss: 0.0152 - volatility_mae: 0.0287                         
Epoch 7: val_spike_probability_auc did not improve from 0.74393
49/49 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - loss: 0.4612 - spike_probability_auc: 0.6445 - spike_probability_loss: 0.6489 - spike_probability_pr_auc: 0.0684 - spike_probability_precision: 0.0348 - spike_probability_recall: 0.7045 - volatility_loss: 0.0163 - volatility_mae: 0.0472 - val_loss: 0.5526 - val_spike_probability_auc: 0.7344 - val_spike_probability_loss: 0.7783 - val_spike_probability_pr_auc: 0.0441 - val_spike_probability_precision: 0.0212 - val_spike_probability_recall: 0.9310 - val_volatility_loss: 0.0248 - val_volatility_mae: 0.0541 - learning_rate: 0.0010
Epoch 8/100
45/49 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4288 - spike_probability_auc: 0.5722 - spike_probability_loss: 0.6060 - spike_probability_pr_auc: 0.0394 - spike_probability_precision: 0.0385 - spike_probability_recall: 0.4935 - volatility_loss: 0.0154 - volatility_mae: 0.0288                     
Epoch 8: val_spike_probability_auc improved from 0.74393 to 0.74899, saving model to models/cnn_2d_multitask_model.keras

Epoch 8: finished saving model to models/cnn_2d_multitask_model.keras
49/49 ━━━━━━━━━━━━━━━━━━━━ 1s 15ms/step - loss: 0.4581 - spike_probability_auc: 0.6700 - spike_probability_loss: 0.6443 - spike_probability_pr_auc: 0.0558 - spike_probability_precision: 0.0392 - spike_probability_recall: 0.6818 - volatility_loss: 0.0166 - volatility_mae: 0.0513 - val_loss: 0.5413 - val_spike_probability_auc: 0.7490 - val_spike_probability_loss: 0.7618 - val_spike_probability_pr_auc: 0.0419 - val_spike_probability_precision: 0.0241 - val_spike_probability_recall: 0.9310 - val_volatility_loss: 0.0251 - val_volatility_mae: 0.0613 - learning_rate: 0.0010
Epoch 9/100
45/49 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4003 - spike_probability_auc: 0.5978 - spike_probability_loss: 0.5654 - spike_probability_pr_auc: 0.0581 - spike_probability_precision: 0.0436 - spike_probability_recall: 0.4042 - volatility_loss: 0.0153 - volatility_mae: 0.0275                         
Epoch 9: val_spike_probability_auc did not improve from 0.74899
49/49 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - loss: 0.4296 - spike_probability_auc: 0.7286 - spike_probability_loss: 0.6035 - spike_probability_pr_auc: 0.0926 - spike_probability_precision: 0.0493 - spike_probability_recall: 0.7045 - volatility_loss: 0.0159 - volatility_mae: 0.0423 - val_loss: 0.4058 - val_spike_probability_auc: 0.6655 - val_spike_probability_loss: 0.5682 - val_spike_probability_pr_auc: 0.0369 - val_spike_probability_precision: 0.0344 - val_spike_probability_recall: 0.4138 - val_volatility_loss: 0.0245 - val_volatility_mae: 0.0459 - learning_rate: 0.0010
Epoch 10/100
47/49 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.4577 - spike_probability_auc: 0.5803 - spike_probability_loss: 0.6473 - spike_probability_pr_auc: 0.0845 - spike_probability_precision: 0.0620 - spike_probability_recall: 0.4833 - volatility_loss: 0.0153 - volatility_mae: 0.0267                         
Epoch 10: val_spike_probability_auc did not improve from 0.74899
49/49 ━━━━━━━━━━━━━━━━━━━━ 1s 23ms/step - loss: 0.4617 - spike_probability_auc: 0.7735 - spike_probability_loss: 0.6490 - spike_probability_pr_auc: 0.1142 - spike_probability_precision: 0.0667 - spike_probability_recall: 0.7955 - volatility_loss: 0.0156 - volatility_mae: 0.0373 - val_loss: 0.4666 - val_spike_probability_auc: 0.7141 - val_spike_probability_loss: 0.6551 - val_spike_probability_pr_auc: 0.0404 - val_spike_probability_precision: 0.0351 - val_spike_probability_recall: 0.6207 - val_volatility_loss: 0.0244 - val_volatility_mae: 0.0406 - learning_rate: 0.0010
Epoch 11/100
49/49 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - loss: 0.3749 - spike_probability_auc: 0.6644 - spike_probability_loss: 0.5289 - spike_probability_pr_auc: 0.0789 - spike_probability_precision: 0.0555 - spike_probability_recall: 0.5833 - volatility_loss: 0.0153 - volatility_mae: 0.0263                        
Epoch 11: val_spike_probability_auc improved from 0.74899 to 0.75684, saving model to models/cnn_2d_multitask_model.keras

Epoch 11: finished saving model to models/cnn_2d_multitask_model.keras
49/49 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - loss: 0.3951 - spike_probability_auc: 0.7879 - spike_probability_loss: 0.5540 - spike_probability_pr_auc: 0.1112 - spike_probability_precision: 0.0645 - spike_probability_recall: 0.7727 - volatility_loss: 0.0154 - volatility_mae: 0.0319 - val_loss: 0.4516 - val_spike_probability_auc: 0.7568 - val_spike_probability_loss: 0.6325 - val_spike_probability_pr_auc: 0.0494 - val_spike_probability_precision: 0.0378 - val_spike_probability_recall: 0.7586 - val_volatility_loss: 0.0244 - val_volatility_mae: 0.0369 - learning_rate: 0.0010
Epoch 12/100
48/49 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - loss: 0.3594 - spike_probability_auc: 0.6666 - spike_probability_loss: 0.5069 - spike_probability_pr_auc: 0.1230 - spike_probability_precision: 0.0561 - spike_probability_recall: 0.5305 - volatility_loss: 0.0153 - volatility_mae: 0.0254                        
Epoch 12: val_spike_probability_auc did not improve from 0.75684
49/49 ━━━━━━━━━━━━━━━━━━━━ 1s 23ms/step - loss: 0.3405 - spike_probability_auc: 0.8511 - spike_probability_loss: 0.4754 - spike_probability_pr_auc: 0.1782 - spike_probability_precision: 0.0839 - spike_probability_recall: 0.7955 - volatility_loss: 0.0153 - volatility_mae: 0.0275 - val_loss: 0.3590 - val_spike_probability_auc: 0.7320 - val_spike_probability_loss: 0.4985 - val_spike_probability_pr_auc: 0.0440 - val_spike_probability_precision: 0.0361 - val_spike_probability_recall: 0.4483 - val_volatility_loss: 0.0244 - val_volatility_mae: 0.0335 - learning_rate: 0.0010
Epoch 13/100
49/49 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - loss: 0.3467 - spike_probability_auc: 0.6889 - spike_probability_loss: 0.4887 - spike_probability_pr_auc: 0.1546 - spike_probability_precision: 0.0630 - spike_probability_recall: 0.5993 - volatility_loss: 0.0153 - volatility_mae: 0.0252                        
Epoch 13: val_spike_probability_auc did not improve from 0.75684
49/49 ━━━━━━━━━━━━━━━━━━━━ 2s 31ms/step - loss: 0.3160 - spike_probability_auc: 0.8654 - spike_probability_loss: 0.4404 - spike_probability_pr_auc: 0.2265 - spike_probability_precision: 0.0914 - spike_probability_recall: 0.8409 - volatility_loss: 0.0153 - volatility_mae: 0.0269 - val_loss: 0.2723 - val_spike_probability_auc: 0.7135 - val_spike_probability_loss: 0.3756 - val_spike_probability_pr_auc: 0.0423 - val_spike_probability_precision: 0.0433 - val_spike_probability_recall: 0.3448 - val_volatility_loss: 0.0244 - val_volatility_mae: 0.0334 - learning_rate: 0.0010
Epoch 14/100
49/49 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.3094 - spike_probability_auc: 0.7303 - spike_probability_loss: 0.4354 - spike_probability_pr_auc: 0.1259 - spike_probability_precision: 0.0723 - spike_probability_recall: 0.5814 - volatility_loss: 0.0153 - volatility_mae: 0.0253                         
Epoch 14: val_spike_probability_auc did not improve from 0.75684
49/49 ━━━━━━━━━━━━━━━━━━━━ 1s 15ms/step - loss: 0.3088 - spike_probability_auc: 0.8683 - spike_probability_loss: 0.4301 - spike_probability_pr_auc: 0.1662 - spike_probability_precision: 0.0882 - spike_probability_recall: 0.7955 - volatility_loss: 0.0153 - volatility_mae: 0.0270 - val_loss: 0.3209 - val_spike_probability_auc: 0.7332 - val_spike_probability_loss: 0.4436 - val_spike_probability_pr_auc: 0.0459 - val_spike_probability_precision: 0.0396 - val_spike_probability_recall: 0.5172 - val_volatility_loss: 0.0244 - val_volatility_mae: 0.0335 - learning_rate: 0.0010
Epoch 15/100
46/49 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.3353 - spike_probability_auc: 0.6974 - spike_probability_loss: 0.4724 - spike_probability_pr_auc: 0.1015 - spike_probability_precision: 0.0623 - spike_probability_recall: 0.7219 - volatility_loss: 0.0153 - volatility_mae: 0.0253                         
Epoch 15: val_spike_probability_auc did not improve from 0.75684
49/49 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 0.3157 - spike_probability_auc: 0.8604 - spike_probability_loss: 0.4400 - spike_probability_pr_auc: 0.1388 - spike_probability_precision: 0.0847 - spike_probability_recall: 0.7955 - volatility_loss: 0.0153 - volatility_mae: 0.0271 - val_loss: 0.4249 - val_spike_probability_auc: 0.7400 - val_spike_probability_loss: 0.5917 - val_spike_probability_pr_auc: 0.0483 - val_spike_probability_precision: 0.0318 - val_spike_probability_recall: 0.5862 - val_volatility_loss: 0.0244 - val_volatility_mae: 0.0335 - learning_rate: 0.0010
Epoch 16/100
49/49 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.3543 - spike_probability_auc: 0.7286 - spike_probability_loss: 0.4994 - spike_probability_pr_auc: 0.1172 - spike_probability_precision: 0.0592 - spike_probability_recall: 0.7324 - volatility_loss: 0.0153 - volatility_mae: 0.0262                     
Epoch 16: val_spike_probability_auc did not improve from 0.75684
49/49 ━━━━━━━━━━━━━━━━━━━━ 1s 25ms/step - loss: 0.3362 - spike_probability_auc: 0.8603 - spike_probability_loss: 0.4688 - spike_probability_pr_auc: 0.1608 - spike_probability_precision: 0.0840 - spike_probability_recall: 0.7727 - volatility_loss: 0.0153 - volatility_mae: 0.0275 - val_loss: 0.6762 - val_spike_probability_auc: 0.7395 - val_spike_probability_loss: 0.9495 - val_spike_probability_pr_auc: 0.0491 - val_spike_probability_precision: 0.0275 - val_spike_probability_recall: 0.7931 - val_volatility_loss: 0.0244 - val_volatility_mae: 0.0334 - learning_rate: 0.0010
Epoch 17/100
46/49 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - loss: 0.5260 - spike_probability_auc: 0.5886 - spike_probability_loss: 0.7449 - spike_probability_pr_auc: 0.0465 - spike_probability_precision: 0.0346 - spike_probability_recall: 0.8002 - volatility_loss: 0.0153 - volatility_mae: 0.0258                         
Epoch 17: val_spike_probability_auc did not improve from 0.75684

Epoch 17: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
49/49 ━━━━━━━━━━━━━━━━━━━━ 2s 37ms/step - loss: 0.3725 - spike_probability_auc: 0.7972 - spike_probability_loss: 0.5202 - spike_probability_pr_auc: 0.0789 - spike_probability_precision: 0.0558 - spike_probability_recall: 0.9091 - volatility_loss: 0.0153 - volatility_mae: 0.0275 - val_loss: 0.2797 - val_spike_probability_auc: 0.6776 - val_spike_probability_loss: 0.3853 - val_spike_probability_pr_auc: 0.0515 - val_spike_probability_precision: 0.0361 - val_spike_probability_recall: 0.3793 - val_volatility_loss: 0.0244 - val_volatility_mae: 0.0339 - learning_rate: 0.0010
Epoch 18/100
49/49 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - loss: 0.3313 - spike_probability_auc: 0.7245 - spike_probability_loss: 0.4666 - spike_probability_pr_auc: 0.1289 - spike_probability_precision: 0.0571 - spike_probability_recall: 0.8239 - volatility_loss: 0.0153 - volatility_mae: 0.0267                        
Epoch 18: val_spike_probability_auc did not improve from 0.75684
49/49 ━━━━━━━━━━━━━━━━━━━━ 1s 27ms/step - loss: 0.2862 - spike_probability_auc: 0.8922 - spike_probability_loss: 0.3981 - spike_probability_pr_auc: 0.1888 - spike_probability_precision: 0.0870 - spike_probability_recall: 0.9545 - volatility_loss: 0.0154 - volatility_mae: 0.0278 - val_loss: 0.2307 - val_spike_probability_auc: 0.6855 - val_spike_probability_loss: 0.3157 - val_spike_probability_pr_auc: 0.0510 - val_spike_probability_precision: 0.0419 - val_spike_probability_recall: 0.3103 - val_volatility_loss: 0.0244 - val_volatility_mae: 0.0334 - learning_rate: 5.0000e-04
Epoch 19/100
46/49 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - loss: 0.3300 - spike_probability_auc: 0.7019 - spike_probability_loss: 0.4649 - spike_probability_pr_auc: 0.1346 - spike_probability_precision: 0.0640 - spike_probability_recall: 0.6943 - volatility_loss: 0.0151 - volatility_mae: 0.0254                        
Epoch 19: val_spike_probability_auc did not improve from 0.75684
49/49 ━━━━━━━━━━━━━━━━━━━━ 1s 24ms/step - loss: 0.2921 - spike_probability_auc: 0.8844 - spike_probability_loss: 0.4065 - spike_probability_pr_auc: 0.2192 - spike_probability_precision: 0.1000 - spike_probability_recall: 0.8409 - volatility_loss: 0.0152 - volatility_mae: 0.0271 - val_loss: 0.2633 - val_spike_probability_auc: 0.7032 - val_spike_probability_loss: 0.3615 - val_spike_probability_pr_auc: 0.0483 - val_spike_probability_precision: 0.0426 - val_spike_probability_recall: 0.3793 - val_volatility_loss: 0.0244 - val_volatility_mae: 0.0334 - learning_rate: 5.0000e-04
Epoch 20/100
47/49 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - loss: 0.3054 - spike_probability_auc: 0.7419 - spike_probability_loss: 0.4298 - spike_probability_pr_auc: 0.1648 - spike_probability_precision: 0.0656 - spike_probability_recall: 0.8375 - volatility_loss: 0.0153 - volatility_mae: 0.0255                         
Epoch 20: val_spike_probability_auc did not improve from 0.75684
49/49 ━━━━━━━━━━━━━━━━━━━━ 1s 28ms/step - loss: 0.2586 - spike_probability_auc: 0.9111 - spike_probability_loss: 0.3591 - spike_probability_pr_auc: 0.2642 - spike_probability_precision: 0.1005 - spike_probability_recall: 0.9545 - volatility_loss: 0.0153 - volatility_mae: 0.0270 - val_loss: 0.2450 - val_spike_probability_auc: 0.7094 - val_spike_probability_loss: 0.3356 - val_spike_probability_pr_auc: 0.0476 - val_spike_probability_precision: 0.0446 - val_spike_probability_recall: 0.3448 - val_volatility_loss: 0.0244 - val_volatility_mae: 0.0333 - learning_rate: 5.0000e-04
Epoch 21/100
47/49 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - loss: 0.2771 - spike_probability_auc: 0.7599 - spike_probability_loss: 0.3894 - spike_probability_pr_auc: 0.1786 - spike_probability_precision: 0.0865 - spike_probability_recall: 0.8051 - volatility_loss: 0.0153 - volatility_mae: 0.0252                         
Epoch 21: val_spike_probability_auc did not improve from 0.75684
49/49 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - loss: 0.2321 - spike_probability_auc: 0.9309 - spike_probability_loss: 0.3217 - spike_probability_pr_auc: 0.2837 - spike_probability_precision: 0.1289 - spike_probability_recall: 0.9318 - volatility_loss: 0.0153 - volatility_mae: 0.0268 - val_loss: 0.2559 - val_spike_probability_auc: 0.7099 - val_spike_probability_loss: 0.3510 - val_spike_probability_pr_auc: 0.0390 - val_spike_probability_precision: 0.0439 - val_spike_probability_recall: 0.3448 - val_volatility_loss: 0.0244 - val_volatility_mae: 0.0332 - learning_rate: 5.0000e-04
Epoch 22/100
48/49 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - loss: 0.2714 - spike_probability_auc: 0.7609 - spike_probability_loss: 0.3812 - spike_probability_pr_auc: 0.1733 - spike_probability_precision: 0.0816 - spike_probability_recall: 0.6375 - volatility_loss: 0.0153 - volatility_mae: 0.0253                        
Epoch 22: val_spike_probability_auc did not improve from 0.75684
49/49 ━━━━━━━━━━━━━━━━━━━━ 2s 31ms/step - loss: 0.2293 - spike_probability_auc: 0.9325 - spike_probability_loss: 0.3176 - spike_probability_pr_auc: 0.2742 - spike_probability_precision: 0.1283 - spike_probability_recall: 0.8864 - volatility_loss: 0.0153 - volatility_mae: 0.0269 - val_loss: 0.3282 - val_spike_probability_auc: 0.7162 - val_spike_probability_loss: 0.4530 - val_spike_probability_pr_auc: 0.0446 - val_spike_probability_precision: 0.0379 - val_spike_probability_recall: 0.3793 - val_volatility_loss: 0.0244 - val_volatility_mae: 0.0333 - learning_rate: 5.0000e-04
Epoch 23/100
49/49 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - loss: 0.2768 - spike_probability_auc: 0.7722 - spike_probability_loss: 0.3888 - spike_probability_pr_auc: 0.1933 - spike_probability_precision: 0.0829 - spike_probability_recall: 0.8418 - volatility_loss: 0.0153 - volatility_mae: 0.0252                        
Epoch 23: val_spike_probability_auc did not improve from 0.75684

Epoch 23: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
49/49 ━━━━━━━━━━━━━━━━━━━━ 2s 31ms/step - loss: 0.2282 - spike_probability_auc: 0.9235 - spike_probability_loss: 0.3162 - spike_probability_pr_auc: 0.2707 - spike_probability_precision: 0.1280 - spike_probability_recall: 0.9773 - volatility_loss: 0.0153 - volatility_mae: 0.0267 - val_loss: 0.2564 - val_spike_probability_auc: 0.6967 - val_spike_probability_loss: 0.3515 - val_spike_probability_pr_auc: 0.0350 - val_spike_probability_precision: 0.0370 - val_spike_probability_recall: 0.2759 - val_volatility_loss: 0.0244 - val_volatility_mae: 0.0332 - learning_rate: 5.0000e-04
Epoch 24/100
49/49 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - loss: 0.2365 - spike_probability_auc: 0.7943 - spike_probability_loss: 0.3313 - spike_probability_pr_auc: 0.2192 - spike_probability_precision: 0.0941 - spike_probability_recall: 0.8418 - volatility_loss: 0.0153 - volatility_mae: 0.0255                         
Epoch 24: val_spike_probability_auc did not improve from 0.75684
49/49 ━━━━━━━━━━━━━━━━━━━━ 1s 28ms/step - loss: 0.2040 - spike_probability_auc: 0.9440 - spike_probability_loss: 0.2819 - spike_probability_pr_auc: 0.3249 - spike_probability_precision: 0.1378 - spike_probability_recall: 0.9773 - volatility_loss: 0.0153 - volatility_mae: 0.0269 - val_loss: 0.2280 - val_spike_probability_auc: 0.6792 - val_spike_probability_loss: 0.3114 - val_spike_probability_pr_auc: 0.0317 - val_spike_probability_precision: 0.0281 - val_spike_probability_recall: 0.1724 - val_volatility_loss: 0.0244 - val_volatility_mae: 0.0332 - learning_rate: 2.5000e-04
Epoch 25/100
47/49 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - loss: 0.2189 - spike_probability_auc: 0.7970 - spike_probability_loss: 0.3062 - spike_probability_pr_auc: 0.2221 - spike_probability_precision: 0.1145 - spike_probability_recall: 0.8352 - volatility_loss: 0.0153 - volatility_mae: 0.0253                         
Epoch 25: val_spike_probability_auc did not improve from 0.75684
49/49 ━━━━━━━━━━━━━━━━━━━━ 1s 27ms/step - loss: 0.1878 - spike_probability_auc: 0.9556 - spike_probability_loss: 0.2590 - spike_probability_pr_auc: 0.3279 - spike_probability_precision: 0.1713 - spike_probability_recall: 0.9773 - volatility_loss: 0.0153 - volatility_mae: 0.0268 - val_loss: 0.2396 - val_spike_probability_auc: 0.6718 - val_spike_probability_loss: 0.3279 - val_spike_probability_pr_auc: 0.0323 - val_spike_probability_precision: 0.0276 - val_spike_probability_recall: 0.1724 - val_volatility_loss: 0.0244 - val_volatility_mae: 0.0332 - learning_rate: 2.5000e-04
Epoch 26/100
47/49 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.2220 - spike_probability_auc: 0.7872 - spike_probability_loss: 0.3106 - spike_probability_pr_auc: 0.2278 - spike_probability_precision: 0.0994 - spike_probability_recall: 0.7386 - volatility_loss: 0.0153 - volatility_mae: 0.0252                         
Epoch 26: val_spike_probability_auc did not improve from 0.75684
49/49 ━━━━━━━━━━━━━━━━━━━━ 1s 15ms/step - loss: 0.1889 - spike_probability_auc: 0.9540 - spike_probability_loss: 0.2606 - spike_probability_pr_auc: 0.3375 - spike_probability_precision: 0.1522 - spike_probability_recall: 0.9545 - volatility_loss: 0.0153 - volatility_mae: 0.0268 - val_loss: 0.2325 - val_spike_probability_auc: 0.6466 - val_spike_probability_loss: 0.3178 - val_spike_probability_pr_auc: 0.0301 - val_spike_probability_precision: 0.0312 - val_spike_probability_recall: 0.1724 - val_volatility_loss: 0.0244 - val_volatility_mae: 0.0332 - learning_rate: 2.5000e-04
Epoch 26: early stopping
Restoring model weights from the end of the best epoch: 11.

Loading best saved CNN model...

Generating validation predictions...

----------------------------------------------------------------------
VALIDATION FORECAST SUMMARY
----------------------------------------------------------------------
Validation observations: 1644
Actual spikes: 29
Actual spike rate: 1.7640%
Predicted spikes: 582
Predicted spike rate: 35.4015%

Classification cutoff: 0.5
Volatility spike threshold: 0.10927724

----------------------------------------------------------------------
CONTINUOUS VOLATILITY FORECAST
----------------------------------------------------------------------
Mean actual volatility: 1.18682778
Mean predicted volatility: 0.00400153
Maximum actual volatility: 1246.77575684
Maximum predicted volatility: 0.03105108

----------------------------------------------------------------------
SPIKE PROBABILITY DISTRIBUTION
----------------------------------------------------------------------
50% percentile: 0.416463
75% percentile: 0.550973
90% percentile: 0.618342
95% percentile: 0.661546
99% percentile: 0.979206
100% percentile: 0.999058

----------------------------------------------------------------------
FIRST 20 VALIDATION FORECASTS
----------------------------------------------------------------------
target_date  actual_volatility  predicted_volatility  actual_spike  predicted_spike_probability  predicted_spike
 2014-01-02           0.000167              0.006660             0                     0.331513                0
 2014-01-06           0.000000              0.006042             0                     0.321877                0
 2014-01-07           0.000000              0.013589             0                     0.366718                0
 2014-01-08           0.000000              0.023180             0                     0.404502                0
 2014-01-09           0.000000              0.019671             0                     0.399797                0
 2014-01-10           0.000000              0.013478             0                     0.375569                0
 2014-01-13           0.000167              0.015277             0                     0.374397                0
 2014-01-15           0.000000              0.012768             0                     0.369752                0
 2014-01-16           0.000000              0.015977             0                     0.391805                0
 2014-01-17           0.000000              0.014514             0                     0.391658                0
 2014-01-20           0.000000              0.019191             0                     0.399207                0
 2014-01-21           0.000042              0.013440             0                     0.370112                0
 2014-01-22           0.000000              0.010367             0                     0.364122                0
 2014-01-23           0.000000              0.007894             0                     0.345400                0
 2014-01-24           0.000000              0.011718             0                     0.379479                0
 2014-01-27           0.000000              0.010316             0                     0.366659                0
 2014-01-28           0.000000              0.007946             0                     0.361452                0
 2014-01-29           0.000000              0.003135             0                     0.296863                0
 2014-01-30           0.000000              0.004588             0                     0.319407                0
 2014-01-31           0.000000              0.003132             0                     0.292064                0

Saved validation forecasts: results/cnn_2d_multitask_validation_predictions.csv
Saved training history: results/cnn_2d_multitask_training_history.csv
Saved model configuration: results/cnn_2d_multitask_config.json

======================================================================
STEP 10C-V3 FINISHED SUCCESSFULLY
======================================================================

Best model:
models/cnn_2d_multitask_model.keras

Training history:
results/cnn_2d_multitask_training_history.csv

Validation predictions:
results/cnn_2d_multitask_validation_predictions.csv

Configuration:
results/cnn_2d_multitask_config.json

FINAL TEST SET REMAINS UNTOUCHED.

Next stage: evaluate the multitask CNN validation performance before Step 10D.

In [13]: 

@author: setup
"""
# ============================================================
# STEP 10C-V3
# MULTITASK MULTIVARIATE 2D-CNN
#
# PRIMARY OBJECTIVE:
#   Forecast whether the next observation is a volatility spike
#
# SECONDARY OBJECTIVE:
#   Forecast the continuous next-period FX volatility
#
# DATA SPLIT:
#   TRAIN      : 2006-05-09 -> 2013-12-31
#   VALIDATION : 2014-01-02 -> 2020-12-31
#   TEST       : 2021-01-04 -> 2025-10-29
#
# TEST SET REMAINS COMPLETELY UNTOUCHED
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import Input, Model

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATA_DIR = "data_cleaned"
RESULTS_DIR = "results"
MODEL_DIR = "models"


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

TRAIN_X_FILE = os.path.join(
    DATA_DIR,
    "cnn_X_train_scaled.npy"
)

TRAIN_Y_FILE = os.path.join(
    DATA_DIR,
    "cnn_y_train_volatility.npy"
)

TRAIN_META_FILE = os.path.join(
    DATA_DIR,
    "cnn_train_metadata.csv"
)


TEST_X_FILE = os.path.join(
    DATA_DIR,
    "cnn_X_test_scaled.npy"
)

TEST_Y_FILE = os.path.join(
    DATA_DIR,
    "cnn_y_test_volatility.npy"
)

TEST_META_FILE = os.path.join(
    DATA_DIR,
    "cnn_test_metadata.csv"
)


# ------------------------------------------------------------
# Output files
# ------------------------------------------------------------

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "cnn_2d_multitask_model.keras"
)

HISTORY_FILE = os.path.join(
    RESULTS_DIR,
    "cnn_2d_multitask_training_history.csv"
)

VALIDATION_FILE = os.path.join(
    RESULTS_DIR,
    "cnn_2d_multitask_validation_predictions.csv"
)

CONFIG_FILE = os.path.join(
    RESULTS_DIR,
    "cnn_2d_multitask_config.json"
)


# ============================================================
# 3. PROJECT SETTINGS
# ============================================================

RANDOM_SEED = 42

VALIDATION_START = pd.Timestamp(
    "2014-01-01"
)

VALIDATION_END = pd.Timestamp(
    "2020-12-31"
)


# Frozen project spike threshold
SPIKE_THRESHOLD = 0.10927724


# Maximum number of epochs
EPOCHS = 100

BATCH_SIZE = 32

LEARNING_RATE = 0.001


# ------------------------------------------------------------
# Multi-task loss weights
#
# 30% continuous volatility forecasting
# 70% spike classification
# ------------------------------------------------------------

VOLATILITY_LOSS_WEIGHT = 0.30

SPIKE_LOSS_WEIGHT = 0.70


# ============================================================
# 4. REPRODUCIBILITY
# ============================================================

np.random.seed(
    RANDOM_SEED
)

tf.random.set_seed(
    RANDOM_SEED
)


# ============================================================
# 5. CREATE DIRECTORIES
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 6. HEADER
# ============================================================

print("=" * 70)
print("STEP 10C-V3: MULTITASK 2D-CNN")
print("=" * 70)


# ============================================================
# 7. LOAD TRAINING DATA
# ============================================================

print("\nLoading training CNN inputs...")

X_train_full = np.load(
    TRAIN_X_FILE
)

print(
    f"Training input shape: "
    f"{X_train_full.shape}"
)


print("\nLoading training volatility targets...")

y_vol_full = np.load(
    TRAIN_Y_FILE
)

print(
    f"Training target shape: "
    f"{y_vol_full.shape}"
)


print("\nLoading training metadata...")

train_meta = pd.read_csv(
    TRAIN_META_FILE
)

train_meta["target_date"] = pd.to_datetime(
    train_meta["target_date"]
)

train_meta = train_meta.reset_index(
    drop=True
)


# ============================================================
# 8. LOAD TEST DATA
# ============================================================
#
# We load the test set only to verify dimensions.
#
# It will NOT participate in:
#
#   - model fitting
#   - validation
#   - early stopping
#   - model selection
#   - threshold selection
#
# ============================================================

print("\nLoading test data for verification only...")

X_test = np.load(
    TEST_X_FILE
)

y_test_vol = np.load(
    TEST_Y_FILE
)

test_meta = pd.read_csv(
    TEST_META_FILE
)

test_meta["target_date"] = pd.to_datetime(
    test_meta["target_date"]
)

test_meta = test_meta.reset_index(
    drop=True
)


print(
    f"Test input shape: "
    f"{X_test.shape}"
)

print(
    f"Test target shape: "
    f"{y_test_vol.shape}"
)

print(
    "\nIMPORTANT: Test data remains untouched."
)


# ============================================================
# 9. INPUT CONSISTENCY CHECKS
# ============================================================

print("\n" + "-" * 70)
print("INPUT VALIDATION")
print("-" * 70)


if X_train_full.ndim != 4:

    raise ValueError(
        "Training CNN input must have 4 dimensions."
    )


if X_test.ndim != 4:

    raise ValueError(
        "Test CNN input must have 4 dimensions."
    )


expected_input_shape = (
    4,
    20,
    1
)


if X_train_full.shape[1:] != expected_input_shape:

    raise ValueError(
        "Training input must have shape "
        "(samples, 4, 20, 1)."
    )


if X_test.shape[1:] != expected_input_shape:

    raise ValueError(
        "Test input must have shape "
        "(samples, 4, 20, 1)."
    )


if len(X_train_full) != len(y_vol_full):

    raise ValueError(
        "Training X and y lengths do not match."
    )


if len(X_test) != len(y_test_vol):

    raise ValueError(
        "Test X and y lengths do not match."
    )


if len(X_train_full) != len(train_meta):

    raise ValueError(
        "Training metadata length does not match training data."
    )


if len(X_test) != len(test_meta):

    raise ValueError(
        "Test metadata length does not match test data."
    )


print(
    "PASS: Input dimensions and lengths are consistent."
)


# ============================================================
# 10. INVALID VALUE CHECKS
# ============================================================

print("\nChecking for NaN and Inf values...")


if np.isnan(X_train_full).any():

    raise ValueError(
        "NaN detected in training inputs."
    )


if np.isinf(X_train_full).any():

    raise ValueError(
        "Inf detected in training inputs."
    )


if np.isnan(X_test).any():

    raise ValueError(
        "NaN detected in test inputs."
    )


if np.isinf(X_test).any():

    raise ValueError(
        "Inf detected in test inputs."
    )


if np.isnan(y_vol_full).any():

    raise ValueError(
        "NaN detected in training targets."
    )


if np.isnan(y_test_vol).any():

    raise ValueError(
        "NaN detected in test targets."
    )


if np.any(y_vol_full < 0):

    raise ValueError(
        "Negative training volatility detected."
    )


if np.any(y_test_vol < 0):

    raise ValueError(
        "Negative test volatility detected."
    )


print(
    "PASS: No invalid values detected."
)


# ============================================================
# 11. CHRONOLOGICAL TRAIN / VALIDATION SPLIT
# ============================================================
#
# Full model-development data:
#
#   2006 -> 2020
#
# Split into:
#
#   Training:
#       2006 -> 2013
#
#   Validation:
#       2014 -> 2020
#
# Final test:
#       2021 -> 2025
#
# ============================================================

train_dates_full = (
    train_meta["target_date"]
)


train_mask = (
    train_dates_full
    < VALIDATION_START
)


validation_mask = (
    (train_dates_full >= VALIDATION_START)
    &
    (train_dates_full <= VALIDATION_END)
)


train_indices = np.where(
    train_mask.to_numpy()
)[0]


validation_indices = np.where(
    validation_mask.to_numpy()
)[0]


# ------------------------------------------------------------
# Select training data
# ------------------------------------------------------------

X_train = X_train_full[
    train_indices
]

y_train_vol = y_vol_full[
    train_indices
]

train_dates = train_dates_full.iloc[
    train_indices
].reset_index(drop=True)


# ------------------------------------------------------------
# Select validation data
# ------------------------------------------------------------

X_validation = X_train_full[
    validation_indices
]

y_validation_vol = y_vol_full[
    validation_indices
]

validation_dates = train_dates_full.iloc[
    validation_indices
].reset_index(drop=True)


# ============================================================
# 12. DISPLAY CHRONOLOGICAL SPLIT
# ============================================================

print("\n" + "-" * 70)
print("CHRONOLOGICAL TRAIN / VALIDATION SPLIT")
print("-" * 70)


print(
    f"Training observations:   "
    f"{len(X_train)}"
)


print(
    f"Validation observations: "
    f"{len(X_validation)}"
)


print(
    f"Training period: "
    f"{train_dates.min()} "
    f"→ "
    f"{train_dates.max()}"
)


print(
    f"Validation period: "
    f"{validation_dates.min()} "
    f"→ "
    f"{validation_dates.max()}"
)


if train_dates.max() >= validation_dates.min():

    raise ValueError(
        "Training and validation periods overlap."
    )


print(
    "PASS: Training and validation are chronologically separated."
)


# ============================================================
# 13. CREATE SPIKE LABELS
# ============================================================
#
# Spike definition is frozen:
#
#       r² > 0.10927724
#
# ============================================================

y_train_spike = (
    y_train_vol > SPIKE_THRESHOLD
).astype(np.int32)


y_validation_spike = (
    y_validation_vol > SPIKE_THRESHOLD
).astype(np.int32)


# ============================================================
# 14. SPIKE DISTRIBUTION
# ============================================================

train_spikes = int(
    y_train_spike.sum()
)

train_nonspikes = int(
    len(y_train_spike) - train_spikes
)


validation_spikes = int(
    y_validation_spike.sum()
)

validation_nonspikes = int(
    len(y_validation_spike)
    - validation_spikes
)


print("\n" + "-" * 70)
print("SPIKE DISTRIBUTION")
print("-" * 70)


print(
    f"Training non-spikes: "
    f"{train_nonspikes}"
)


print(
    f"Training spikes:     "
    f"{train_spikes}"
)


print(
    f"Training spike rate: "
    f"{train_spikes / len(y_train_spike):.4%}"
)


print(
    f"\nValidation non-spikes: "
    f"{validation_nonspikes}"
)


print(
    f"Validation spikes:     "
    f"{validation_spikes}"
)


print(
    f"Validation spike rate: "
    f"{validation_spikes / len(y_validation_spike):.4%}"
)


print(
    f"\nFrozen spike threshold: "
    f"{SPIKE_THRESHOLD}"
)


if train_spikes == 0:

    raise ValueError(
        "No spike observations exist in the training set."
    )


if validation_spikes == 0:

    raise ValueError(
        "No spike observations exist in the validation set."
    )


# ============================================================
# 15. LOG1P VOLATILITY TARGET
# ============================================================
#
# Continuous target:
#
#       v = r²
#
# Transformation:
#
#       z = log(1 + v)
#
# ============================================================

print("\n" + "-" * 70)
print("VOLATILITY TARGET TRANSFORMATION")
print("-" * 70)


y_train_logvol = np.log1p(
    y_train_vol
)


y_validation_logvol = np.log1p(
    y_validation_vol
)


print(
    f"Training original mean: "
    f"{np.mean(y_train_vol):.8f}"
)


print(
    f"Training original maximum: "
    f"{np.max(y_train_vol):.8f}"
)


print(
    f"Training log1p mean: "
    f"{np.mean(y_train_logvol):.8f}"
)


print(
    f"Training log1p maximum: "
    f"{np.max(y_train_logvol):.8f}"
)


# ============================================================
# 16. CALCULATE BALANCED CLASS WEIGHTS
# ============================================================
#
# Instead of manually selecting a spike weight,
# compute balanced weights directly from training frequency.
#
# ============================================================

classes = np.array([
    0,
    1
])


balanced_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_spike
)


nonspike_weight = float(
    balanced_weights[0]
)


spike_weight = float(
    balanced_weights[1]
)


print("\n" + "-" * 70)
print("BALANCED CLASS WEIGHTS")
print("-" * 70)


print(
    f"Non-spike weight: "
    f"{nonspike_weight:.6f}"
)


print(
    f"Spike weight:     "
    f"{spike_weight:.6f}"
)


# ============================================================
# 17. CREATE SAMPLE WEIGHTS
# ============================================================
#
# We have two model outputs:
#
#   output 1 = volatility
#   output 2 = spike probability
#
# Therefore the sample-weight list follows the exact same
# order:
#
#   [volatility weights, spike weights]
#
# ============================================================

# Every observation has equal importance for regression
train_volatility_weights = np.ones(
    len(y_train_vol),
    dtype=np.float32
)


# Classification head receives balanced class weights
train_spike_weights = np.where(
    y_train_spike == 1,
    spike_weight,
    nonspike_weight
).astype(np.float32)


# For validation we do not artificially reweight observations.
# This keeps validation metrics easier to interpret.
validation_volatility_weights = np.ones(
    len(y_validation_vol),
    dtype=np.float32
)


validation_spike_weights = np.ones(
    len(y_validation_spike),
    dtype=np.float32
)


print(
    "\nPASS: Sample weights created."
)


# ============================================================
# 18. BUILD SHARED CNN
# ============================================================

print("\n" + "=" * 70)
print("BUILDING MULTITASK 2D-CNN")
print("=" * 70)


# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

inputs = Input(
    shape=(4, 20, 1),
    name="cnn_input"
)


# ------------------------------------------------------------
# Convolution block 1
# ------------------------------------------------------------

x = Conv2D(
    filters=32,
    kernel_size=(2, 3),
    activation="relu",
    padding="same",
    name="conv2d_block1"
)(inputs)


x = MaxPooling2D(
    pool_size=(2, 2),
    name="maxpool_block1"
)(x)


# ------------------------------------------------------------
# Convolution block 2
# ------------------------------------------------------------

x = Conv2D(
    filters=64,
    kernel_size=(2, 3),
    activation="relu",
    padding="same",
    name="conv2d_block2"
)(x)


# ------------------------------------------------------------
# Shared feature representation
# ------------------------------------------------------------

x = Flatten(
    name="flatten_shared"
)(x)


x = Dense(
    units=64,
    activation="relu",
    name="shared_dense"
)(x)


x = Dropout(
    rate=0.30,
    name="shared_dropout"
)(x)


# ============================================================
# 19. VOLATILITY REGRESSION HEAD
# ============================================================

volatility_branch = Dense(
    units=32,
    activation="relu",
    name="volatility_hidden"
)(x)


volatility_output = Dense(
    units=1,
    activation="softplus",
    name="volatility"
)(volatility_branch)


# ============================================================
# 20. SPIKE CLASSIFICATION HEAD
# ============================================================

spike_branch = Dense(
    units=32,
    activation="relu",
    name="spike_hidden"
)(x)


spike_output = Dense(
    units=1,
    activation="sigmoid",
    name="spike_probability"
)(spike_branch)


# ============================================================
# 21. CREATE MULTI-OUTPUT MODEL
# ============================================================
#
# IMPORTANT:
# The order is:
#
#   [volatility, spike_probability]
#
# Targets and sample weights below use exactly this order.
#
# ============================================================

model = Model(
    inputs=inputs,

    outputs=[
        volatility_output,
        spike_output
    ],

    name="multitask_2d_cnn"
)


# ============================================================
# 22. COMPILE MODEL
# ============================================================
#
# Output 1:
#     Huber regression loss
#
# Output 2:
#     Binary cross-entropy classification loss
#
# Overall:
#
#     0.30 × volatility loss
#     +
#     0.70 × spike loss
#
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)


model.compile(

    optimizer=optimizer,

    loss=[
        tf.keras.losses.Huber(),
        tf.keras.losses.BinaryCrossentropy()
    ],

    loss_weights=[
        VOLATILITY_LOSS_WEIGHT,
        SPIKE_LOSS_WEIGHT
    ],

    metrics=[
        [
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ],

        [
            tf.keras.metrics.AUC(
                name="auc",
                curve="ROC"
            ),

            tf.keras.metrics.AUC(
                name="pr_auc",
                curve="PR"
            ),

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
            )
        ]
    ]
)


# ============================================================
# 23. DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# 24. CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(

    MODEL_FILE,

    monitor="val_spike_probability_auc",

    mode="max",

    save_best_only=True,

    verbose=1
)


early_stopping = EarlyStopping(

    monitor="val_spike_probability_auc",

    mode="max",

    patience=15,

    min_delta=1e-4,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_spike_probability_auc",

    mode="max",

    factor=0.5,

    patience=6,

    min_lr=1e-6,

    verbose=1
)


# ============================================================
# 25. PREPARE TARGETS
# ============================================================
#
# EXACT MODEL OUTPUT ORDER:
#
#     1. volatility
#     2. spike_probability
#
# ============================================================

y_train_targets = [

    y_train_logvol,

    y_train_spike

]


y_validation_targets = [

    y_validation_logvol,

    y_validation_spike

]


# ============================================================
# 26. PREPARE SAMPLE WEIGHTS
# ============================================================

train_sample_weights = [

    train_volatility_weights,

    train_spike_weights

]


validation_sample_weights = [

    validation_volatility_weights,

    validation_spike_weights

]


# ============================================================
# 27. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("STARTING MULTITASK CNN TRAINING")
print("=" * 70)


history = model.fit(

    X_train,

    y_train_targets,

    sample_weight=train_sample_weights,

    validation_data=(

        X_validation,

        y_validation_targets,

        validation_sample_weights

    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    # Preserve chronological order
    shuffle=False,

    callbacks=[

        checkpoint,

        early_stopping,

        reduce_lr

    ],

    verbose=1
)


# ============================================================
# 28. LOAD BEST MODEL
# ============================================================

print("\nLoading best saved CNN model...")


best_model = tf.keras.models.load_model(
    MODEL_FILE
)


# ============================================================
# 29. VALIDATION PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")


validation_predictions = best_model.predict(

    X_validation,

    verbose=0
)


# Model output order:
#
#   validation_predictions[0]
#       = volatility
#
#   validation_predictions[1]
#       = spike probability


validation_pred_logvol = (
    validation_predictions[0]
    .reshape(-1)
)


validation_pred_spike_probability = (
    validation_predictions[1]
    .reshape(-1)
)


# ============================================================
# 30. CONVERT VOLATILITY BACK TO ORIGINAL SCALE
# ============================================================
#
# Model predicts:
#
#       log(1 + volatility)
#
# Therefore:
#
#       volatility = exp(prediction) - 1
#
# ============================================================

validation_pred_volatility = np.expm1(
    validation_pred_logvol
)


# Numerical safety
validation_pred_volatility = np.maximum(
    validation_pred_volatility,
    0.0
)


# ============================================================
# 31. CREATE BINARY SPIKE PREDICTION
# ============================================================
#
# The classification head gives:
#
#       P(spike)
#
# We use 0.50 as the initial classification cutoff.
#
# IMPORTANT:
# This 0.50 probability cutoff is different from the
# volatility threshold 0.10927724.
#
# ============================================================

CLASSIFICATION_CUTOFF = 0.50


validation_pred_spike = (
    validation_pred_spike_probability
    >= CLASSIFICATION_CUTOFF
).astype(np.int32)


# ============================================================
# 32. VALIDATION SUMMARY
# ============================================================

actual_spike_count = int(
    y_validation_spike.sum()
)


predicted_spike_count = int(
    validation_pred_spike.sum()
)


print("\n" + "-" * 70)
print("VALIDATION FORECAST SUMMARY")
print("-" * 70)


print(
    f"Validation observations: "
    f"{len(y_validation_vol)}"
)


print(
    f"Actual spikes: "
    f"{actual_spike_count}"
)


print(
    f"Actual spike rate: "
    f"{actual_spike_count / len(y_validation_vol):.4%}"
)


print(
    f"Predicted spikes: "
    f"{predicted_spike_count}"
)


print(
    f"Predicted spike rate: "
    f"{predicted_spike_count / len(y_validation_vol):.4%}"
)


print(
    f"\nClassification cutoff: "
    f"{CLASSIFICATION_CUTOFF}"
)


print(
    f"Volatility spike threshold: "
    f"{SPIKE_THRESHOLD}"
)


# ============================================================
# 33. CONTINUOUS FORECAST SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("CONTINUOUS VOLATILITY FORECAST")
print("-" * 70)


print(
    f"Mean actual volatility: "
    f"{np.mean(y_validation_vol):.8f}"
)


print(
    f"Mean predicted volatility: "
    f"{np.mean(validation_pred_volatility):.8f}"
)


print(
    f"Maximum actual volatility: "
    f"{np.max(y_validation_vol):.8f}"
)


print(
    f"Maximum predicted volatility: "
    f"{np.max(validation_pred_volatility):.8f}"
)


# ============================================================
# 34. PROBABILITY DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("SPIKE PROBABILITY DISTRIBUTION")
print("-" * 70)


for q in [
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00
]:

    print(
        f"{q:.0%} percentile: "
        f"{np.quantile(validation_pred_spike_probability, q):.6f}"
    )


# ============================================================
# 35. FIRST 20 VALIDATION FORECASTS
# ============================================================

print("\n" + "-" * 70)
print("FIRST 20 VALIDATION FORECASTS")
print("-" * 70)


validation_preview = pd.DataFrame({

    "target_date":
        validation_dates.iloc[:20],

    "actual_volatility":
        y_validation_vol[:20],

    "predicted_volatility":
        validation_pred_volatility[:20],

    "actual_spike":
        y_validation_spike[:20],

    "predicted_spike_probability":
        validation_pred_spike_probability[:20],

    "predicted_spike":
        validation_pred_spike[:20]

})


print(
    validation_preview.to_string(
        index=False
    )
)


# ============================================================
# 36. SAVE VALIDATION PREDICTIONS
# ============================================================

validation_results = pd.DataFrame({

    "target_date":
        validation_dates,

    "actual_volatility":
        y_validation_vol,

    "predicted_volatility":
        validation_pred_volatility,

    "actual_spike":
        y_validation_spike,

    "predicted_spike_probability":
        validation_pred_spike_probability,

    "predicted_spike":
        validation_pred_spike

})


validation_results.to_csv(
    VALIDATION_FILE,
    index=False
)


print(
    f"\nSaved validation forecasts: "
    f"{VALIDATION_FILE}"
)


# ============================================================
# 37. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)


history_df.index.name = "epoch"


history_df.reset_index(
    inplace=True
)


history_df["epoch"] = (
    history_df["epoch"] + 1
)


history_df.to_csv(
    HISTORY_FILE,
    index=False
)


print(
    f"Saved training history: "
    f"{HISTORY_FILE}"
)


# ============================================================
# 38. SAVE MODEL CONFIGURATION
# ============================================================

config = {

    "model_type":
        "Multitask Multivariate 2D-CNN",

    "input_shape":
        [4, 20, 1],

    "features": [

        "fx_log_return",

        "fx_return_squared",

        "brent_log_return",

        "reserves_log_change"

    ],

    "window_size":
        20,

    "continuous_target":
        "next_period_fx_return_squared",

    "continuous_target_transform":
        "log1p",

    "continuous_inverse_transform":
        "expm1",

    "spike_definition":
        "next_period_fx_return_squared > frozen threshold",

    "spike_threshold":
        SPIKE_THRESHOLD,

    "classification_cutoff":
        CLASSIFICATION_CUTOFF,

    "training_period":
        "2006-05-09 to 2013-12-31",

    "validation_period":
        "2014-01-02 to 2020-12-31",

    "final_test_period":
        "2021-01-04 to 2025-10-29",

    "volatility_loss":
        "Huber",

    "spike_loss":
        "BinaryCrossentropy",

    "volatility_loss_weight":
        VOLATILITY_LOSS_WEIGHT,

    "spike_loss_weight":
        SPIKE_LOSS_WEIGHT,

    "nonspike_class_weight":
        nonspike_weight,

    "spike_class_weight":
        spike_weight,

    "epochs_max":
        EPOCHS,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "random_seed":
        RANDOM_SEED,

    "shuffle":
        False,

    "architecture": [

        "Conv2D(32, kernel=(2,3), activation=relu)",

        "MaxPooling2D(pool=(2,2))",

        "Conv2D(64, kernel=(2,3), activation=relu)",

        "Flatten",

        "Dense(64, activation=relu)",

        "Dropout(0.30)",

        "Volatility Dense(32, activation=relu)",

        "Volatility output Dense(1, activation=softplus)",

        "Spike Dense(32, activation=relu)",

        "Spike output Dense(1, activation=sigmoid)"

    ]

}


with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        config,
        file,
        indent=4
    )


print(
    f"Saved model configuration: "
    f"{CONFIG_FILE}"
)


# ============================================================
# 39. FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("STEP 10C-V3 FINISHED SUCCESSFULLY")
print("=" * 70)


print(
    "\nBest model:"
)

print(
    MODEL_FILE
)


print(
    "\nTraining history:"
)

print(
    HISTORY_FILE
)


print(
    "\nValidation predictions:"
)

print(
    VALIDATION_FILE
)


print(
    "\nConfiguration:"
)

print(
    CONFIG_FILE
)


print(
    "\nFINAL TEST SET REMAINS UNTOUCHED."
)


print(
    "\nNext stage: evaluate the multitask CNN "
    "validation performance before Step 10D."
)
