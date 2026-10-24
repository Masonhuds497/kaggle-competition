# kaggle-competition
[README.md](https://github.com/user-attachments/files/33913890/README.md)
# ENGE707 Assessment 3: Facial Feature Classification Challenge

**Author:** Aly Eid (student number 20119696)
**Kaggle username:** alyeid02

Eight-class image classification of cat and dog facial features (CatEye, DogEye, CatFace, DogFace, CatMouth, DogMouth, CatNose, DogNose) with transfer learning.

## Result

| Run | Preprocessing | Validation accuracy | Validation macro-F1 | Kaggle score |
|---|---|---|---|---|
| 1 | Pad to square | 0.9975 | 0.9974 | 0.9201 |
| 2 (final) | Centre-crop to square | 0.9357 | 0.9231 | **0.9249** |

The padded model read the image shape (in the training set every Mouth image is wide and tall images are almost all Faces, but the test set does not follow this), so its validation score was inflated. Centre-cropping hides the shape and gives an honest validation score.

## Method

- Model: ConvNeXt-Tiny pretrained on ImageNet, fine-tuned end to end (27.8M parameters), new head with dropout 0.3 and a linear layer to 8 classes.
- Input: centre-crop to a square, resize to 160 x 160, ImageNet normalisation.
- Training augmentation: random resized crop, horizontal flip, rotation up to 15 degrees, colour jitter, down-up resize, Gaussian blur, random erasing.
- Optimiser: AdamW (weight decay 0.05), learning rate 1e-4 for the backbone and 1e-3 for the head, 1 epoch warm-up then cosine decay, label smoothing 0.1, gradient clipping, mixed precision, batch size 64, 8 epochs, early stopping (patience 5).
- Validation: stratified, group-aware split (6 folds, seed 42), grouped by file hash so duplicates stay together.

## Files

| File | Description |
|---|---|
| `ENGE707_facial_feature_pipeline.ipynb` | The full notebook: data exploration, preprocessing, model, training, evaluation and submission file. Runs top to bottom on Kaggle. |
| `requirements.txt` | Python packages needed. |
| `requirements_frozen.txt` | Exact package versions used for the final run. |
| `results.json` | Settings, final metrics, per-epoch history and environment of the final run. |
| `classification_report.csv` | Per-class precision, recall and F1 on the validation split. |
| `experiment_comparison.csv` | Architecture comparison (ResNet-50, EfficientNet-B0, from-scratch CNN, no-augmentation) with padding preprocessing. |
| `submission.csv` | Predictions for the 29,443 test images, as submitted to Kaggle. |
| `figures/` | Class distribution, sample images, image sizes, augmentation examples, training curves, confusion matrix, most confident errors, experiment comparison. |
| `ENGE707_Assessment3_Report.docx` | The written report. |

Large generated files (model weights, test probabilities and the image scan tables) are not included in the repository. The notebook recreates them.

Recommended settings: `IMG_SIZE = 160`, `FINAL_EPOCHS = 8`, `RUN_ABLATIONS = False`.
4. Run all cells (about 40 minutes). The notebook writes `outputs/submission.csv`, `outputs/results.json` and `outputs/figures/`.

All randomness uses seed 42, so repeated runs give identical results.
