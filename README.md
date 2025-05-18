# Wildfire Detection using MobileNetV2 and RandomForest

## Introduction

This project is a wildfire detection system that uses MobileNetV2 for feature extraction and a RandomForest model for classification. It is designed for real-time wildfire detection, allowing users to upload images and instantly determine whether they contain fire or not.

## Problem Statement

Wildfires are catastrophic events that can lead to severe ecological, economic, and human losses. Early detection is critical to minimizing the damage. This project provides an efficient and scalable solution for detecting wildfires in images, which can be used for real-time monitoring.

## Methodology

1. **Data Collection and Preparation:**

   * Dataset Source: Kaggle Fire Detection Image Dataset.
   * Images labeled as "Fire" and "No Fire".
   * Split into Training (70%), Validation (15%), and Testing (15%) sets.

2. **Data Preprocessing:**

   * Images resized to 224x224 pixels.
   * Normalized pixel values for faster convergence.
   * Data augmentation for training set.

3. **Feature Extraction using MobileNetV2:**

   * Loaded pre-trained MobileNetV2 (ImageNet weights).
   * Removed fully connected layers, retaining convolutional base.
   * Applied Global Average Pooling for 1024-D feature vectors.

4. **Feature Classification with RandomForest:**

   * Trained RandomForestClassifier (100 trees, reproducible results).
   * Evaluated using validation and test sets.

5. **Model Deployment:**

   * Deployed using Streamlit Cloud.
   * Users can upload images for real-time wildfire detection.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/priyanshu596/forest_fire_detection.git
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app locally:

   ```bash
   streamlit run app.py
   ```

## Usage

* Visit the Streamlit app.
* Upload an image.
* The model will predict whether the image contains fire or not.

## Future Scope

* Enhancing model accuracy using more advanced architectures.
* Integrating video stream support for continuous monitoring.
* Deploying on a scalable cloud service for global access.

## Contributing

Feel free to open issues or create pull requests if you have any suggestions.

## License

This project is licensed under the MIT License.
