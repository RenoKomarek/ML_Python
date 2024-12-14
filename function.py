# Prompt 1: "how do i make a pipeline to implement kfold and gridsearch alongside and ml model training like lr"
# ChatGPT, 2024, Prompt 1: https://chatgpt.com/share/675c2ac6-e4bc-8001-8b96-aa8f5c2a3d5e
# Prompt 2: "can you turn this into a class and show me how to use it"
# ChatGPT, 2024, Prompt 2: https://chatgpt.com/share/675c2ac6-e4bc-8001-8b96-aa8f5c2a3d5e

# General Libraries
import numpy as np
import pandas as pd

# Machine Learning Libraries
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier

# Imbalanced Data Handling
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class ModelPipeline:
    def __init__(self, model, param_grid, scoring='f1'):
        """
        Initialize the ModelPipeline with the model, hyperparameter grid, and scoring metric.

        Args:
            model: The machine learning model (e.g., KNN, Logistic Regression, etc.).
            param_grid: Dictionary of hyperparameters for GridSearchCV.
            scoring: Metric used for GridSearchCV. Default is 'f1'.
        """
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring

        # Resampling techniques
        self.smote = SMOTE(sampling_strategy=0.999, random_state=42)
        self.undersample = RandomUnderSampler(sampling_strategy=0.999, random_state=42)

        # Cross-validation
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        # Initialize pipeline
        self.pipeline = Pipeline(steps=[
            ('over', self.smote),
            ('under', self.undersample),
            ('model', self.model)
        ])

        self.grid_search = None
        self.best_model = None
        self.best_params = None

    def fit(self, X_train, y_train):
        """
        Fit the model using GridSearchCV.

        Args:
            X_train: Training feature set.
            y_train: Training labels.
        """
        self.grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=self.param_grid,
            cv=self.kfold,
            scoring=self.scoring,
            n_jobs=-1
        )
        self.grid_search.fit(X_train, y_train)
        self.best_model = self.grid_search.best_estimator_
        self.best_params = self.grid_search.best_params_

        print(f"Best Hyperparameters: {self.best_params}")

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evaluate the model on both training and test datasets.

        Args:
            X_train: Training feature set.
            y_train: Training labels.
            X_test: Test feature set.
            y_test: Test labels.
        """
        # Evaluate on training data
        y_train_pred = self.best_model.predict(X_train)
        print("Training Data Evaluation:")
        print(classification_report(y_train, y_train_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_train, y_train_pred))

        # Evaluate on test data
        y_test_pred = self.best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, self.best_model.named_steps['model'].predict_proba(X_test)[:, 1])

        print("\nTest Data Evaluation:")
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"ROC-AUC Score: {roc_auc}")

        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc
        }
