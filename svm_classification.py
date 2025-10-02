import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification, make_circles, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
from matplotlib.colors import ListedColormap

class SVMClassifier:
    def __init__(self):
        self.linear_model = None
        self.rbf_model = None
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self):
        print("Loading and preparing datasets...")
        
        # Breast Cancer dataset (real-world data)
        breast_cancer = load_breast_cancer()
        X_breast_cancer = breast_cancer.data
        y_breast_cancer = breast_cancer.target
        
        # Select only 2 most important features for 2D visualization
        selector = SelectKBest(f_classif, k=2)
        X_breast_cancer_2d = selector.fit_transform(X_breast_cancer, y_breast_cancer)
        selected_features = selector.get_support(indices=True)
        
        print(f"Breast Cancer Dataset: {X_breast_cancer.shape[0]} samples, {X_breast_cancer.shape[1]} features")
        print(f"Selected 2 best features for visualization: {[breast_cancer.feature_names[i] for i in selected_features]}")
        print(f"Target classes: {breast_cancer.target_names} (0=malignant, 1=benign)")
        
        # Linear dataset (synthetic for comparison)
        X_linear, y_linear = make_classification(
            n_samples=200,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=42,
            n_clusters_per_class=1
        )
        
        # Non-linear dataset (circles)
        X_nonlinear, y_nonlinear = make_circles(
            n_samples=200,
            noise=0.1,
            factor=0.3,
            random_state=42
        )
        
        return (X_breast_cancer, y_breast_cancer), (X_breast_cancer_2d, y_breast_cancer), (X_linear, y_linear), (X_nonlinear, y_nonlinear)
    
    def train_svm_models(self, X, y):
        print("Training SVM models with linear and RBF kernels...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Linear SVM
        self.linear_model = SVC(kernel='linear', C=1.0, random_state=42)
        self.linear_model.fit(X_train_scaled, y_train)
        
        # Train RBF SVM
        self.rbf_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        self.rbf_model.fit(X_train_scaled, y_train)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def evaluate_models(self, X_train, X_test, y_train, y_test):
        print("Evaluating model performance...")
        
        # Predictions
        linear_pred = self.linear_model.predict(X_test)
        rbf_pred = self.rbf_model.predict(X_test)
        
        # Accuracy scores
        linear_accuracy = accuracy_score(y_test, linear_pred)
        rbf_accuracy = accuracy_score(y_test, rbf_pred)
        
        print(f"Linear SVM Accuracy: {linear_accuracy:.4f}")
        print(f"RBF SVM Accuracy: {rbf_accuracy:.4f}")
        
        # Classification reports
        print("\nLinear SVM Classification Report:")
        print(classification_report(y_test, linear_pred))
        
        print("\nRBF SVM Classification Report:")
        print(classification_report(y_test, rbf_pred))
        
        # Confusion matrices
        self.plot_confusion_matrices(y_test, linear_pred, rbf_pred)
        
        return linear_accuracy, rbf_accuracy
    
    def plot_confusion_matrices(self, y_test, linear_pred, rbf_pred):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Linear SVM confusion matrix
        cm_linear = confusion_matrix(y_test, linear_pred)
        sns.heatmap(cm_linear, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Linear SVM Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # RBF SVM confusion matrix
        cm_rbf = confusion_matrix(y_test, rbf_pred)
        sns.heatmap(cm_rbf, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title('RBF SVM Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_decision_boundary(self, X, y, title_suffix=""):
        print(f"Visualizing decision boundaries {title_suffix}...")
        
        # Scale the data for visualization
        X_scaled = self.scaler.fit_transform(X)
        
        # Create a mesh for plotting decision boundary
        h = 0.02
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Train models on full scaled data for visualization
        linear_model = SVC(kernel='linear', C=1.0, random_state=42)
        rbf_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        
        linear_model.fit(X_scaled, y)
        rbf_model.fit(X_scaled, y)
        
        # Create color maps
        colors = ['red', 'blue']
        cmap = ListedColormap(colors[:len(np.unique(y))])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Linear SVM decision boundary
        Z_linear = linear_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_linear = Z_linear.reshape(xx.shape)
        
        ax1.contourf(xx, yy, Z_linear, alpha=0.3, cmap=cmap)
        scatter1 = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=cmap, edgecolors='black')
        ax1.set_title(f'Linear SVM Decision Boundary {title_suffix}')
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        
        # RBF SVM decision boundary
        Z_rbf = rbf_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_rbf = Z_rbf.reshape(xx.shape)
        
        ax2.contourf(xx, yy, Z_rbf, alpha=0.3, cmap=cmap)
        scatter2 = ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=cmap, edgecolors='black')
        ax2.set_title(f'RBF SVM Decision Boundary {title_suffix}')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        
        plt.tight_layout()
        plt.show()
    
    def perform_cross_validation(self, X, y):
        print("Performing cross-validation...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Linear SVM cross-validation
        linear_scores = cross_val_score(
            SVC(kernel='linear', C=1.0, random_state=42), 
            X_scaled, y, cv=5
        )
        
        # RBF SVM cross-validation
        rbf_scores = cross_val_score(
            SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42), 
            X_scaled, y, cv=5
        )
        
        print(f"Linear SVM CV Scores: {linear_scores}")
        print(f"Linear SVM CV Mean: {linear_scores.mean():.4f} (+/- {linear_scores.std() * 2:.4f})")
        
        print(f"RBF SVM CV Scores: {rbf_scores}")
        print(f"RBF SVM CV Mean: {rbf_scores.mean():.4f} (+/- {rbf_scores.std() * 2:.4f})")
        
        return linear_scores, rbf_scores
    
    def tune_hyperparameters(self, X, y):
        print("Tuning hyperparameters...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Parameter grid for RBF SVM
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            SVC(kernel='rbf', random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_scaled, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Visualize parameter tuning results
        self.plot_parameter_tuning_results(grid_search)
        
        return grid_search.best_params_, grid_search.best_score_
    
    def plot_parameter_tuning_results(self, grid_search):
        results_df = pd.DataFrame(grid_search.cv_results_)
        
        # Create pivot table for heatmap
        pivot_table = results_df.pivot_table(
            values='mean_test_score',
            index='param_gamma',
            columns='param_C'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis')
        plt.title('Grid Search Results: Mean CV Score')
        plt.xlabel('C Parameter')
        plt.ylabel('Gamma Parameter')
        plt.show()

def main():
    print("Support Vector Machine Classification Task")
    print("=" * 50)
    
    # Initialize SVM classifier
    svm_classifier = SVMClassifier()
    
    # Load datasets
    (X_breast_cancer, y_breast_cancer), (X_breast_cancer_2d, y_breast_cancer_2d), (X_linear, y_linear), (X_nonlinear, y_nonlinear) = svm_classifier.load_and_prepare_data()
    
    print("\n1. Working with Breast Cancer Dataset (Real-world Data)")
    print("-" * 55)
    
    # Train models on breast cancer dataset (full features)
    X_train_bc, X_test_bc, y_train_bc, y_test_bc = svm_classifier.train_svm_models(X_breast_cancer, y_breast_cancer)
    
    # Evaluate models on breast cancer dataset
    svm_classifier.evaluate_models(X_train_bc, X_test_bc, y_train_bc, y_test_bc)
    
    # Cross-validation on breast cancer dataset
    svm_classifier.perform_cross_validation(X_breast_cancer, y_breast_cancer)
    
    print("\n2. Breast Cancer Dataset - 2D Visualization")
    print("-" * 45)
    
    # Reset scaler for 2D breast cancer dataset
    svm_classifier.scaler = StandardScaler()
    
    # Visualize decision boundaries for breast cancer dataset (2D)
    svm_classifier.visualize_decision_boundary(X_breast_cancer_2d, y_breast_cancer_2d, "(Breast Cancer - 2D)")
    
    print("\n3. Working with Synthetic Linear Dataset")
    print("-" * 42)
    
    # Reset scaler for linear dataset
    svm_classifier.scaler = StandardScaler()
    
    # Train models on linear dataset
    X_train_linear, X_test_linear, y_train_linear, y_test_linear = svm_classifier.train_svm_models(X_linear, y_linear)
    
    # Evaluate models on linear dataset
    svm_classifier.evaluate_models(X_train_linear, X_test_linear, y_train_linear, y_test_linear)
    
    # Visualize decision boundaries for linear dataset
    svm_classifier.visualize_decision_boundary(X_linear, y_linear, "(Synthetic Linear)")
    
    print("\n4. Working with Non-linear Dataset")
    print("-" * 35)
    
    # Reset scaler for non-linear dataset
    svm_classifier.scaler = StandardScaler()
    
    # Train models on non-linear dataset
    X_train_nonlinear, X_test_nonlinear, y_train_nonlinear, y_test_nonlinear = svm_classifier.train_svm_models(X_nonlinear, y_nonlinear)
    
    # Evaluate models on non-linear dataset
    svm_classifier.evaluate_models(X_train_nonlinear, X_test_nonlinear, y_train_nonlinear, y_test_nonlinear)
    
    # Visualize decision boundaries for non-linear dataset
    svm_classifier.visualize_decision_boundary(X_nonlinear, y_nonlinear, "(Non-linear Circles)")
    
    print("\n5. Hyperparameter Tuning on Breast Cancer Dataset")
    print("-" * 52)
    
    # Reset scaler for hyperparameter tuning
    svm_classifier.scaler = StandardScaler()
    
    # Tune hyperparameters on breast cancer dataset
    best_params, best_score = svm_classifier.tune_hyperparameters(X_breast_cancer, y_breast_cancer)
    
    print("\nTask completed successfully!")
    print("Check the generated plots for decision boundaries and performance metrics.")
    print("The breast cancer dataset provides real-world medical data for classification.")

if __name__ == "__main__":
    main()