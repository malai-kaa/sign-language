# Sign Language Recognition System

A computer vision-based application that translates hand-signed alphabets into text and converts the generated text into speech.

## Overview

This project implements a deep learning-based Sign Language Recognition System that identifies hand gestures representing individual alphabets. It enables users to construct words through recognized gestures and communicate the resulting text using text-to-speech functionality.

## Key Features

- **Hand Gesture Recognition:** Identifies sign language alphabet gestures.
- **Alphabet-to-Text Translation:** Converts recognized gestures into corresponding letters.
- **Interactive Word Formation:** Adds recognized letters to build words using a dedicated input control.
- **Text Editing:** Supports space insertion and backspace functionality.
- **Text-to-Speech Conversion:** Converts completed text into spoken output.

## Technical Stack

- Python
- TensorFlow / Keras
- MediaPipe
- OpenCV
- NumPy
- Scikit-learn

## Model

The system uses a trained deep learning model (`sign_model.h5`) for alphabet classification. MediaPipe supports hand tracking and gesture processing, while the trained model predicts the corresponding alphabet.

## Repository Contents

| File | Description |
|---|---|
| `sign_model.h5` | Trained recognition model |
| `sign language translator.ipynb` | Development notebook |
| `sign language translator.py` | Python implementation |

## Dataset

The model was trained using an ASL alphabet image dataset. The original dataset is not included in this repository.

## Future Enhancements

- Expand the supported sign vocabulary.
- Improve recognition accuracy.
- Enhance real-time prediction performance.
- Explore continuous sign language recognition.

