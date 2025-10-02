# Support Vector Machines (SVM) for Classification

This project demonstrates the implementation of Support Vector Machines for both linear and non-linear classification tasks using scikit-learn.

## Objective

Use SVMs for linear and non-linear classification with decision boundary visualization and hyperparameter tuning.

## Features

- Binary classification using SVM
- Linear and RBF kernel implementation
- Decision boundary visualization for 2D data
- Hyperparameter tuning (C and gamma parameters)
- Cross-validation for performance evaluation
- Performance metrics and visualization

## Requirements

- Python 3.8+
- Required packages listed in requirements.txt

## Installation

### Option 1: Using Setup Script (Recommended)
```bash
python setup.py
```

### Option 2: Manual Installation
```bash
pip install -r requirements.txt
```

## Usage

Run the SVM classification script:
```bash
python svm_classification.py
```

## Project Structure

- `svm_classification.py`: Main SVM implementation
- `setup.py`: Setup script for easy installation and environment checking
- `requirements.txt`: Required Python packages
- `README.md`: Project documentation

## Output

- Decision boundary visualizations for linear and RBF kernels
- Performance metrics and confusion matrices
- Cross-validation results
- Hyperparameter tuning results with heatmap visualization

## Datasets

The project uses multiple datasets for comprehensive SVM analysis:

### 1. Breast Cancer Dataset (Primary - Real-world Data)
- **Source**: Wisconsin Diagnostic Breast Cancer Dataset (from Kaggle/UCI)
- **Features**: 30 real-valued features computed from digitized images of breast mass
- **Samples**: 569 instances
- **Classes**: Malignant (0) and Benign (1)
- **Purpose**: Real-world medical classification problem

### 2. Synthetic Datasets (For Comparison)
- **Linear Dataset**: Generated using make_classification (linearly separable)
- **Non-linear Dataset**: Generated using make_circles (concentric circles)
- **Purpose**: Demonstrate SVM behavior on different data patterns