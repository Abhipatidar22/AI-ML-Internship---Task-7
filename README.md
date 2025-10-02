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

## Dataset

The project uses synthetic binary classification datasets generated using scikit-learn's make_classification and make_circles functions, which create linearly separable and non-linearly separable data points for demonstration purposes.