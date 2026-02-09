# Cats vs Dogs Image Classifier with PyTorch CNN

A custom Convolutional Neural Network (CNN) built from scratch to classify images of cats and dogs using the Kaggle Dogs vs Cats dataset.

## Project Overview
- Trained on a subset of the Kaggle Dogs vs Cats dataset
- Custom CNN architecture with 4 convolutional layers and dropout
- Data augmentation to improve generalization
- Training with Adam optimizer and CrossEntropyLoss

## Results
- Number of trainable parameters: 26,144,962
- Final Train Accuracy: 64.27%
- Best Test Accuracy: 65.00% (achieved at epoch 15)
- Final Average Loss: 0.5487

## How to Run

### 1. Install dependencies
Create a virtual environment (recommended) and install the required packages:

```bash
# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Or directly:
pip install torch torchvision torchaudio numpy matplotlib

### 2. Download the dataset
Download the dataset from Kaggle:
https://www.kaggle.com/datasets/salader/dogs-vs-cats

# Extract the zip file
Make sure you have these folders:
archive/train/ (with subfolders cats/ and dogs/)
archive/test/ (with subfolders cats/ and dogs/)

### 3. Run the training
Simply execute the script:
python main.py

# The script will:
Load and augment the data
Train the CNN for 20 epochs
Print loss per step and average loss per epoch
Evaluate train and test accuracy
Save the model as cat_dog_model_final.pth

## Model Architecture:
Conv layers: 3 → 32 → 64 → 128 → 256 filters (with padding)
MaxPooling after each conv
Dropout (0.3) after first fully connected layer
FC layers: 2561414 → 512 → 128 → 2

## Requirements:
See requirements.txt for exact versions.
Future Improvements
Use transfer learning (ResNet18 or EfficientNet) for higher accuracy
Add validation set and early stopping
Experiment with more advanced augmentation and hyperparameters
Made with ❤️ and PyTorch
Feel free to fork, star, or reach out!