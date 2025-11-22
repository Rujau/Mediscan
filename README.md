neumonia Detection Using Deep Learning (CNN Model + Grad-CAM + Deployment)

A deep learning–based medical imaging project for automatic detection of Pneumonia from Chest X-ray images, using a custom Convolutional Neural Network (CNN), Grad-CAM visualization, and optional deployment using Flask / Gradio / TFLite mobile deployment.

🚀 Project Overview

This project aims to build an AI-powered system that can classify Chest X-Ray images as Normal or Pneumonia using Convolutional Neural Networks (CNNs). The trained model achieves strong performance and includes Grad-CAM heatmaps to explain model decisions visually. The system supports lightweight deployment to web, desktop, and mobile using TFLite quantization.

🧠 Key Features

✔️ Custom CNN architecture trained from scratch

✔️ Data Augmentation for improved generalization

✔️ Tested on the famous Chest X-Ray Pneumonia Dataset

✔️ Evaluation metrics: accuracy, confusion matrix, classification report

✔️ Grad-CAM visual explanations to show infected lung regions

✔️ Export to TensorFlow Lite (.tflite) for mobile deployment

✔️ Deployment options:

🌐 Flask Web App

⚡ Gradio Web Interface

📱 TFLite Android/iOS app-ready model

📂 Dataset

Chest X-Ray Pneumonia Dataset
Available on Kaggle:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Folder structure used:
/train
   NORMAL/
   PNEUMONIA/

/val
   NORMAL/
   PNEUMONIA/

/test
   NORMAL/
   PNEUMONIA/

Project Structure:
📦 Pneumonia-Detection
 ┣ 📂 dataset/
 ┣ 📂 models/
 ┃ ┗ pneumonia_model.h5
 ┣ 📂 deployment/
 ┃ ┣ flask_app/
 ┃ ┣ tflite_model/
 ┣ 📂 gradcam_outputs/
 ┣ 📄 notebook.ipynb   ← full training & Grad-CAM
 ┣ 📄 README.md
 ┗ 📄 requirements.txt
🛠️ Tech Stack

Python

TensorFlow / Keras

NumPy, Matplotlib, Seaborn

OpenCV

Scikit-Learn

Flask / Gradio

TensorFlow Lite

🔬 Model Architecture

Conv2D → BatchNorm → MaxPool

Conv2D → BatchNorm → MaxPool

Conv2D → BatchNorm → MaxPool

Flatten

Dense (128) + Dropout

Dense (1) Sigmoid

Total Params: ~11.1M

📈 Training Results

Best Accuracy Achieved: ≈ 80–85%

Test Accuracy: ≈ 81%

Good balance between false positives and false negatives

Grad-CAM highlights pneumonia-infected lung areas

Grad-CAM Explainability

The project includes a working Grad-CAM pipeline that produces heatmaps showing exactly where the model is focusing in X-ray images.

Example output:

gradcam_overlay.png


(You can add your own sample image after generating Grad-CAM)

📱 Deployment Options
1. Gradio (fastest)

Run this cell:

iface.launch(share=True)

2. Flask Web App

Upload images → returns prediction + Grad-CAM overlay.

3. TFLite Mobile Deployment

Export with:

converter = tf.lite.TFLiteConverter.from_keras_model(fixed_model)
tflite_model = converter.convert()

▶️ Getting Started
Install Requirements
pip install -r requirements.txt

Run Training Notebook
jupyter notebook notebook.ipynb

Run Flask App
python app.py
