# Indian Traffic Sign Detection using YOLOv8

## Overview

This project detects Indian traffic signs from road images using a YOLOv8s object-detection model and provides a local Streamlit interface for image-based analysis.

The project prioritizes **accuracy and reliability over real-time speed**. Image upload is the primary interaction mode; webcam support is intentionally optional.

## Project Objectives

- Build a clean and leakage-free Indian traffic-sign dataset.
- Train a YOLOv8s object detector for 89 traffic-sign classes.
- Evaluate the model on an independent held-out test set.
- Test robustness on negative images and real-world photographs.
- Build a practical Streamlit application with confidence-aware voice announcements.

## Dataset

The frozen clean dataset contains:

- Train: **9,575 images**
- Validation: **1,196 images**
- Test: **1,198 images**
- Total: **11,969 images**
- Classes: **89**
- Annotated objects: **12,400**
- Missing labels: **0**
- Invalid labels: **0**
- Duplicate groups: **0**
- Cross-split duplicate groups: **0**
- Negative/empty images: **387** (341 train, 23 validation, 23 test)

The dataset was frozen after cleaning to preserve a reliable baseline.

## Model and Training

Model: **YOLOv8s**

Training configuration:

- Epochs: 100
- Training image size: 640
- Batch size: 16
- Device: Tesla T4 during training
- Patience: 20
- Pretrained: Yes
- Parameters: 11,160,027
- GFLOPs: 28.6
- Approximate training time: 4.893 hours

The trained `best.pt` model was downloaded and retained as the project baseline.

## Validation Results

- Precision: **0.912**
- Recall: **0.870**
- mAP50: **0.915**
- mAP50-95: **0.849**

## Independent Test Results

The completely separate test set produced:

- Precision: **86.51%**
- Recall: **86.86%**
- mAP50: **91.18%**
- mAP50-95: **84.92%**

The closeness of validation and test performance did not indicate an obvious validation/test collapse.

## Negative-image Robustness Test

The test set contained 23 negative images.

At confidence threshold 0.25:

- Negative images: **23**
- Images with false detections: **0**
- False positives: **0**
- Clean-image rate: **100%**

This was an important reliability check because an earlier webcam implementation had produced unwanted detections when no traffic sign was present.

## Real-world Evaluation

More than **50 real-world images** were tested through the application.

The reported behavior was encouraging overall, with approximately **6–7 missed cases**. A recurring failure mode was yellow/black directional chevron signage.

Controlled resolution experiments showed that higher inference resolution can help with small signs:

- At 640, a 30-sign and a 40-sign could be missed in full-scene images.
- At 960, the same signs were recovered in controlled tests.
- At 1280, the 30 and 40 signs were correctly recognized with high confidence.

A particularly useful diagnostic was tight-crop inference:

- SPEED_LIMIT_40: **0.93**
- SPEED_LIMIT_30: **0.95**

This demonstrated that the model had learned these classes, while small signs in a full scene could still be difficult.

## Speed-limit Class Diagnostic

Training-object counts for selected speed-limit classes were:

| Class | Train objects |
|---|---:|
| SPEED_LIMIT_30 | 264 |
| SPEED_LIMIT_40 | 105 |
| SPEED_LIMIT_50 | 236 |
| SPEED_LIMIT_60 | 124 |
| SPEED_LIMIT_70 | 84 |
| SPEED_LIMIT_80 | 102 |
| SPEED_LIMIT_100 | 109 |

On 129 clean test images containing speed-limit signs, the diagnostic TP/FP/FN results at IoU >= 0.50 were:

| Class | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| SPEED_LIMIT_30 | 32 | 0 | 1 | 1.000 | 0.970 |
| SPEED_LIMIT_40 | 13 | 0 | 0 | 1.000 | 1.000 |
| SPEED_LIMIT_50 | 29 | 0 | 0 | 1.000 | 1.000 |
| SPEED_LIMIT_60 | 16 | 2 | 0 | 0.889 | 1.000 |
| SPEED_LIMIT_70 | 11 | 0 | 0 | 1.000 | 1.000 |
| SPEED_LIMIT_80 | 13 | 1 | 0 | 0.929 | 1.000 |
| SPEED_LIMIT_100 | 14 | 0 | 0 | 1.000 | 1.000 |

These are project diagnostics, not substitutes for the official mAP calculation.

## Direction/Chevron Limitation

The class mapping contains a `DIRECTION` class. The clean dataset contains only:

- Train: **5** objects
- Validation: **1** object
- Test: **1** object

The five inspected training examples were yellow/black directional chevron-style signs. This matches the recurring real-world failure pattern observed during manual testing.

Because the class is extremely underrepresented, the current baseline should be considered limited for this sign type. The dataset remains frozen rather than being altered retrospectively.

## Streamlit Application

The application uses the downloaded `best.pt` model and provides:

- Image upload
- Confidence threshold control in percentage form
- Inference-resolution selection (640/960/1280)
- Original image display
- Annotated detection image
- Detection summary
- Detection table with confidence percentages and bounding boxes
- Confidence-aware voice announcements
- Manual voice playback

The default inference resolution is **1280** because controlled experiments showed improved recognition of smaller signs compared with 640.

## Application Flow

```text
Upload road image
       |
       v
YOLOv8s inference at selected resolution
       |
       v
Traffic-sign detections
       |
       +----> Detection details
       |
       +----> Original + annotated images
       |
       +----> Voice announcement
```

## Voice Announcements

The application uses confidence-aware natural language. Examples include:

- High confidence: **"Stop sign detected. Confidence is very high."**
- High confidence: **"Speed limit 50 detected with high confidence."**
- Moderate confidence: **"A speed limit 50 may be present. Confidence is moderate."**
- Low confidence: **"Possible speed limit 50. Detection confidence is low."**

The spoken message is presented below the images and can be played manually.

## Running the Project

Project structure:

```text
IndianTrafficSignProject/
├── IndianTrafficSign_CLEAN/
├── best.pt
├── app.py
├── requirements.txt
└── app_backup_v1.py
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the Streamlit application:

```powershell
python -m streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, normally:

```text
http://localhost:8501
```

## Current Limitations

1. Very small or distant signs can be difficult in full-scene images.
2. Real-world images can differ from the training distribution.
3. The `DIRECTION` class is severely underrepresented (5/1/1 objects across train/validation/test).
4. CPU inference at 1280 is slower than lower-resolution inference.
5. The current system is accuracy-focused rather than optimized for real-time video.

## Future Work

The most direct future improvement is to expand the `DIRECTION` class with representative yellow/black directional-chevron examples and retrain a controlled Dataset V2 while preserving the current clean dataset and `best.pt` as the baseline.

Other future work may include:

- Region-based or two-stage handling of very small signs.
- Optional webcam frame analysis.
- More systematic field testing.
- Better calibration of confidence thresholds.
- Broader geographic and environmental coverage.

## Conclusion

The current project demonstrates a complete machine-learning workflow rather than only a detector demo: dataset cleaning, leakage prevention, YOLOv8 training, independent testing, negative-image robustness testing, real-world testing, error analysis, and deployment through Streamlit.

The current `best.pt` model is retained as a strong baseline. The most clearly identified limitation is the severely underrepresented `DIRECTION` class and the resulting weakness on yellow/black directional chevrons.
