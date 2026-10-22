# Apple Leaf Disease Classification — PlantVillage

## 1. Problem
Classify an apple leaf image into one of four classes:
Apple Scab, Black Rot, Cedar Apple Rust, or Healthy.

## 2. Dataset source
- Dataset: **PlantVillage**, Apple subset only.
- Downloaded from the official PlantVillage repository:
  https://github.com/spMohanty/PlantVillage-Dataset
- Exact folders used: `raw/color/Apple___Apple_scab`,
  `raw/color/Apple___Black_rot`, `raw/color/Apple___Cedar_apple_rust`,
  `raw/color/Apple___healthy`
- This repository contains the exact dataset used for training,
  so no separate download is needed to reproduce the work.

## 3. Original dataset (before split)
| Class | Images |
|---|---|
| Apple_scab | 630 |
| Black_rot | 621 |
| Cedar_apple_rust | 275 |
| Healthy | 1645 |
| **Total** | **3171** |

Original size: 51 MB. Images are 256×256 RGB colour images.

## 4. Data split
- Ratio: **80% train / 10% validation / 10% test**
- Method: stratified split, one class at a time, random seed **42**,
  so the same split is produced every time.
- Train is used only for training. Validation is used only after
  each epoch to check accuracy and to save the best model.
  Test is used only once, at the end, after training is finished.
- Where 10% is not a whole number, it is rounded to the nearest
  whole image, and train takes the remainder. That is why Cedar
  Apple Rust train is 219 instead of 220, and Healthy train is
  1317 instead of 1316.

| Class | Total | Train (original) | Validation | Test |
|---|---|---|---|---|
| Apple_scab | 630 | 504 | 63 | 63 |
| Black_rot | 621 | 497 | 62 | 62 |
| Cedar_apple_rust | 275 | 219 | 28 | 28 |
| Healthy | 1645 | 1317 | 164 | 164 |
| **Total** | **3171** | **2537** | **317** | **317** |

## 5. Class imbalance
Healthy (1645) is about 6 times larger than Cedar Apple Rust (275).
If nothing is done, the model learns Healthy better and ignores
Cedar Apple Rust, while overall accuracy still looks good.
The solution used here is augmentation of the training set only,
described in Section 6. Validation and test are never augmented.

## 6. Augmentation — how it was done, in detail
Augmentation is applied **only to the training set**, never to
validation or test. The goal is exactly **3000 images per class**
in training, so all classes become perfectly balanced.

### 6.1 Technique used for each augmented copy
Each augmented copy is made with a different random combination of:
- Horizontal flip and/or vertical flip (random)
- Rotation: −25 to +25 degrees (random)
- Zoom: 0.8 to 1.2 (random, centre crop / resize)
- Brightness: 0.8 to 1.2 (random)
- Contrast: 0.8 to 1.2 (random)
- Slight colour jitter: 0.85 to 1.15 (random, kept small so the
  disease colour — for example the orange of Cedar Rust and the
  dark spots of Scab — is not changed)
- Slight Gaussian noise (standard deviation 5)

Techniques NOT used: Cutout / Random Erasing (it could hide the
disease spot), Mixup / CutMix (it mixes two diseases and confuses
the label), and large colour change (it changes the disease colour).

### 6.2 How a class reaches exactly 3000 — the cycle method
For each class:
```
needed_augmented_copies = 3000 − original_train_images
```
Augmentation goes through the original images in order. In one
cycle, every original image gets exactly one new augmented copy,
each with a different random combination, so copies are not identical.
When one cycle finishes, the next cycle starts again from the first
image. Augmentation stops exactly when the class reaches 3000.
The last cycle may be partial.

Worked examples:
- **Cedar_apple_rust:** original 219, needed 2781.
  12 full cycles = 219 × 12 = 2628, remaining 153.
  First 153 originals get a 13th copy, the other 66 get 12 copies.
  153 × 13 + 66 × 12 = 2781. Total = 219 + 2781 = 3000.
- **Apple_scab:** original 504, needed 2496.
  4 full cycles = 2016, remaining 480 get a 5th copy.
  Total = 504 + 2496 = 3000.
- **Black_rot:** original 497, needed 2503.
  5 full cycles = 2485, remaining 18 get a 6th copy.
  Total = 497 + 2503 = 3000.
- **Healthy:** original 1317, needed 1683.
  1 full cycle = 1317, remaining 366 get a 2nd copy.
  Total = 1317 + 1683 = 3000.

Augmented files are named `<original_name>_aug_<cycle>.JPG`,
so original and augmented images can always be told apart.
The script that did this is `augment.py` (seed 42).

### 6.3 Final dataset in this folder
| Split | Apple_scab | Black_rot | Cedar_apple_rust | Healthy | Total |
|---|---|---|---|---|---|
| train | 3000 | 3000 | 3000 | 3000 | 12000 |
| validation | 63 | 62 | 28 | 164 | 317 |
| test | 63 | 62 | 28 | 164 | 317 |
| **Total in folder** | | | | | **12634** |

All 12634 images were checked: 0 corrupt, 0 empty, all open correctly.
A sample visual check was also done per class: disease spots remain
visible in augmented images, and augmented copies differ in angle,
flip and brightness.

Folder structure:
```
data/
  train/Apple_scab, Black_rot, Cedar_apple_rust, Healthy
  validation/Apple_scab, Black_rot, Cedar_apple_rust, Healthy
  test/Apple_scab, Black_rot, Cedar_apple_rust, Healthy
```

## 7. Preprocessing
- Resize every image to 224×224 pixels.
- Convert to tensor and normalise with ImageNet mean
  [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225],
  because the pretrained models were trained with those values.

## 8. Models — transfer learning, one by one
Top 7 models for plant disease classification, in this order:
1. ResNet18
2. ResNet50
3. EfficientNet-B0
4. DenseNet121
5. MobileNetV2
6. VGG16
7. EfficientNet-B3

Each model is first trained for **5 epochs**. Models with good
results are then trained for **30 epochs** for the final comparison.
The best models are then fused (feature fusion) as the novel approach.

### 8.1 Fine-tuning — what is trained and what is frozen
This is partial fine-tuning, NOT training only the classification head.
The classification head AND the last feature block are trained.
Earlier layers, which already learned general shapes, edges and
textures from ImageNet, stay frozen.

- ResNet18 / ResNet50: layer4 + fc trained; conv1, layer1, layer2, layer3 frozen.
- EfficientNet-B0 / B3: features.7, features.8 + classifier trained; rest frozen.
- DenseNet121: denseblock4 + classifier trained; rest frozen.
- MobileNetV2: features.18 + classifier trained; rest frozen.
- VGG16: features 28–30 + classifier trained; rest frozen.

### 8.2 Training settings
- Optimizer: Adam, learning rate 0.001
- Loss: Cross-Entropy
- Batch size: 32
- No early stopping: all epochs run fully. Validation is monitored
  only, and the best-validation model is saved after each epoch.
- Seeds fixed: Python 42, NumPy 42, PyTorch 42.

### 8.3 Reproducibility
The exact split and augmented dataset in this folder is the dataset
used for training, so retraining on this folder gives close results.
Retraining may still differ by about 1–2% in accuracy because of
hardware and floating-point differences. For an exactly identical
test result, use the saved best-model weights and run them on the
test set without retraining.

## 9. Evaluation and final results (to be filled after training)
For every model, and for the final fused model:
- Training loss and validation accuracy per epoch (training graphs)
- Test accuracy
- Per-class precision, recall and F1-score (classification report)
- Confusion matrix (which class is confused with which)
- Bar graph comparing all models by accuracy
- Train vs validation gap, to check overfitting / underfitting
- Model size and training time
The final model is the one whose results are natural in every way:
high accuracy, balanced per-class scores, and no overfitting
(train much higher than validation) or underfitting (both low).

## 10. Run the code yourself — instructions
This code is tested and correct. Run it yourself, on GPU or on CPU,
with the same augmented dataset provided in this folder. Do not
re-split or re-augment the dataset — use the `data/` folder exactly
as given here, because this is the exact dataset the results are
based on.

### 10.1 Quick test first (1 epoch)
Run the 1-epoch version first, to check that everything works on
your machine before full training:
```bash
pip install torch torchvision scikit-learn pillow numpy
python train_models_1epoch.py
```
This trains every model for 1 epoch only, and saves results in
`model_results_1epoch/`.

### 10.2 Full training — 5 epochs first
```bash
python train_models.py
```
This trains the top 7 models one by one, 5 epochs each, in this order:
ResNet18, ResNet50, EfficientNet-B0, DenseNet121, MobileNetV2,
VGG16, EfficientNet-B3. Results and best-model weights are saved in
`model_results/` after every model.

### 10.3 Final training — 30 epochs
For the models with good 5-epoch results, change `EPOCHS = 5` to
`EPOCHS = 30` in `train_models.py` and run it again for those models.
No early stopping is used — all epochs run fully.

### 10.4 GPU or CPU
- On GPU (for example Google Colab or Kaggle GPU): the same code
  runs as it is, much faster than on CPU.
- On CPU: the same code runs as it is, only slower. Nothing else
  needs to be changed.
- Set the number of epochs yourself in the code, as above. The
  dataset, split, augmentation, preprocessing, fine-tuning and
  evaluation settings stay exactly as described in this README.

## 11. Files in this folder
- `data/` — exact split + augmented dataset, 12634 images
- `train.py` — single-model (ResNet18) training script, 5 epochs
- `train_models.py` — top-7 models, one by one, 5 epochs each
- `train_models_1epoch.py` — same as train_models.py, but 1 epoch
  only, for a quick code test before full training
- `augment.py` — augmentation script that produced the train set
- `README.md` — this file
