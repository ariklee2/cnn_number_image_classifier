# MNIST Image Classifier (PyTorch)

## Project Overview

This project implements a Convolutional Neural Network (CNN) in PyTorch to classify handwritten digits from the MNIST dataset. The model is trained on 28x28 grayscale images and predicts digits from 0–9.

The project demonstrates:
- Deep learning model construction using PyTorch
- CNN architecture design
- Training and evaluation pipelines
- Model persistence (saving and loading)
- Custom image inference

---

## Technologies Used

- Python 3
- PyTorch
- Torchvision
- PIL (Python Imaging Library)

---

## Dataset

The model is trained using the MNIST dataset:

- 60,000 training images
- 10,000 test images
- 10 digit classes (0–9)
- Image size: 28x28 (grayscale)

The dataset is automatically downloaded using `torchvision.datasets.MNIST`.

---

## Model Architecture

The Convolutional Neural Network consists of:

- Conv2D (32 filters, 3x3 kernel)
- ReLU activation
- Conv2D (64 filters, 3x3 kernel)
- ReLU activation
- Conv2D (64 filters, 3x3 kernel)
- ReLU activation
- Flatten layer
- Fully Connected (Linear) layer with 10 output classes

Architecture Flow:

Input (1x28x28)  
→ Conv2D  
→ ReLU  
→ Conv2D  
→ ReLU  
→ Conv2D  
→ ReLU  
→ Flatten  
→ Linear (10 outputs)

Loss Function: CrossEntropyLoss  
Optimizer: Adam (learning rate = 0.001)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/mnist-classifier.git
cd mnist-classifier
```

Install dependencies:

```bash
pip install torch torchvision pillow
```

---

## Training the Model

To train the model:

```bash
python main.py
```

The script will:
- Download the dataset (if not already present)
- Train the model for 10 epochs
- Display loss and accuracy per epoch
- Save the trained model as:

model_state.pt

---

## Running Inference on a Custom Image

To classify your own digit image:

1. Add your image file to the project directory (e.g., `img_3.jpg`)
2. Ensure the image:
   - Is grayscale
   - Is resized to 28x28 pixels (the script resizes automatically)
3. Run:

```bash
python main.py
```

If the image exists, the script will output:

Predicted Digit: X

---

## Project Structure

mnist-classifier/
│
├── data/                  # MNIST dataset (auto-downloaded)
├── model_state.pt         # Saved trained model weights
├── img_3.jpg              # Optional custom test image
├── main.py                # Training and inference script
└── README.md

---

## Results

After training, the model achieves approximately:

- ~98–99% accuracy on the MNIST test dataset (depending on training epochs)
