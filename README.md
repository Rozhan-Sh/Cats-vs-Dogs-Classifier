# Cats vs Dogs Image Classifier with PyTorch CNN

A custom Convolutional Neural Network (CNN) built from scratch to classify images of cats and dogs using a subset of the Kaggle Dogs vs Cats dataset.

## Project Overview
- Custom CNN architecture with 4 convolutional layers and dropout
- Data augmentation techniques to improve generalization
- Trained with Adam optimizer and CrossEntropyLoss
- Evaluated on train and test sets

## Results
- Number of trainable parameters: 26,144,962
- Final Train Accuracy: 64.27%
- Best Test Accuracy: 65.00% (achieved at epoch 15)
- Final Average Loss: 0.5487

## How to Run

### 1. Install dependencies
Create a virtual environment (recommended) and install the required packages:

```bash
# Optional: Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install torch torchvision torchaudio numpy matplotlib

Or use the provided requirements file:
pip install -r requirements.txt
```
### 2. Download the dataset
Download the dataset from Kaggle:

https://www.kaggle.com/datasets/salader/dogs-vs-cats

Extract the zip file

Ensure the folder structure is:

archive/

├── train/
 
│  ├── cats/
 
│  └── dogs/
 
└── test/
 
   ├── cats/
     
   └── dogs/

### 3. Run the training
Execute the main script:
```bash
python main.py
```
The script will:
- Load and augment the training data
- Train the CNN for 20 epochs
- Print loss per step and average loss per epoch
- Evaluate accuracy on train and test sets
- Save the trained model as cat_dog_model_final.pth

## Model Architecture
Convolutional layers: 3 → 32 → 64 → 128 → 256 filters (with padding)

MaxPooling (2×2) after each conv layer

Dropout (0.3) after the first fully connected layer

Fully connected layers: 256×14×14 → 512 → 128 → 2 (2 classes: cat/dog)

## Requirements
Listed in requirements.txt:

torch

torchvision

torchaudio

numpy

matplotlib

---
### Future Improvements

Implement transfer learning (e.g., ResNet18 or EfficientNet) for higher accuracy

Add a validation set and early stopping

Experiment with advanced augmentation, learning rate scheduling, and hyperparameter tuning

Made with ❤️ and PyTorch

Feel free to fork, star, or reach out!



