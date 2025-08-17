# 🌿 Plant Leaf Disease Classification with Explainable AI (XAI)

This project is a Streamlit web application designed for the CSE 366 - Artificial Intelligence course. It allows users to classify plant leaf diseases by uploading an image. The application leverages several pre-trained deep learning models and provides visual explanations for the predictions using various Explainable AI (XAI) techniques.

## ✨ Features

* **Model Selection**: Choose from a variety of trained models through a simple dropdown menu.

* **Image Upload**: Upload your own JPG or PNG images of plant leaves for classification.

* **Prediction Display**: View the top-3 predicted classes with their corresponding probabilities.

* **XAI Visualizations**: Generate and compare five different visual explanations for each prediction:

  * Grad-CAM

  * Grad-CAM++

  * Eigen-CAM

  * Ablation-CAM

  * LIME-style

* **Downloadable Results**: Export all generated visualizations and a summary of the results as a single ZIP file.

## 🤖 Models

The application includes the following pre-trained models for classification:

* **LeafNet_CNN**: A custom Convolutional Neural Network.

* **EfficientNet_B0**

* **MobileNet_V3**

* **ResNet18**

* **ViT_Small** (Vision Transformer)

## 💻 Technologies Used

* **Python**

* **Streamlit**: For creating the interactive web application.

* **PyTorch**: As the deep learning framework.

* **Pytorch-grad-cam**: For generating CAM-based explanations.

* **Timm**: For Vision Transformer models.

* **Pillow, OpenCV, Matplotlib**: For image processing and visualization.

## ⚙️ Setup and Installation

To run this project locally, please follow these steps:

1. **Clone the repository**:

2. git clone <your-repository-url>
cd <your-repository-name>

2. **Create a virtual environment** (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate


3. **Install dependencies**:

pip install -r requirements.txt


4. **Place model weights**:
Download the model weights and place them in the root directory of the project. The required weight files are:

* `LeafNet_CNN.pth`

* `EfficientNet_B0_best.pth`

* `MobileNet_V3_best.pth`

* `ResNet18_best.pth`

* `ViT_Small_best.pth`

## 🚀 Usage

To start the application, run the following command in your terminal:

streamlit run app.py


## 👥 Team and Contributions

Please fill in your team members' names and their contributions to the project below:

* MD Nasibul Islam Sazid[2022-3-60-014]

* Sababa Fairoze Prionty[2022-3-60-229]

* Anha Dhali[2022-3-60-107]


