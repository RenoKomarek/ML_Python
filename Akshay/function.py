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
        self.smote = SMOTE(sampling_strategy=0.5, random_state=42)
        self.undersample = RandomUnderSampler(sampling_strategy=0.8, random_state=42)

        # Cross-validation
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        # Initialize pipeline
        self.pipeline = Pipeline(steps=[
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

# Function to evaluate resampling strategies and return a DataFrame of F1 scores
def evaluate_resampling_strategies(model_name, model, param_grid, X_train, y_train, X_test, y_test):
    results = {}

    # Case 1: Only SMOTE
    X_train_smote, y_train_smote = SMOTE(sampling_strategy=0.5, random_state=42).fit_resample(X_train, y_train)
    smote_pipeline = ModelPipeline(model=model, param_grid=param_grid, scoring='f1')
    smote_pipeline.fit(X_train_smote, y_train_smote)
    results['SMOTE'] = smote_pipeline.evaluate(X_train_smote, y_train_smote, X_test, y_test)["f1"]

    # Case 2: Only Undersample
    X_train_under, y_train_under = RandomUnderSampler(sampling_strategy=0.8, random_state=42).fit_resample(X_train, y_train)
    undersample_pipeline = ModelPipeline(model=model, param_grid=param_grid, scoring='f1')
    undersample_pipeline.fit(X_train_under, y_train_under)
    results['Undersample'] = undersample_pipeline.evaluate(X_train_under, y_train_under, X_test, y_test)["f1"]

    # Case 3: No Resampling
    no_resample_pipeline = ModelPipeline(model=model, param_grid=param_grid, scoring='f1')
    no_resample_pipeline.fit(X_train, y_train)
    results['No Resampling'] = no_resample_pipeline.evaluate(X_train, y_train, X_test, y_test)["f1"]

    results_df = pd.DataFrame([results], index=[model_name])
    return results_df
