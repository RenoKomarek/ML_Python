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

    def fit_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate the model in three scenarios: oversampling, undersampling, and no sampling.

        Args:
            X_train: Training feature set.
            y_train: Training labels.
            X_test: Test feature set.
            y_test: Test labels.

        Returns:
            A DataFrame with F1 scores for each scenario.
        """
        results = []

        # Define the sampling strategies
        sampling_strategies = {
            "Oversampling": [('over', self.smote)],
            "Undersampling": [('under', self.undersample)],
            "No Sampling": []
        }

        for strategy_name, steps in sampling_strategies.items():
            print(f"Evaluating strategy: {strategy_name}")

            # Create pipeline for the current strategy
            pipeline = Pipeline(steps=steps + [('model', self.model)])

            # Perform GridSearchCV
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=self.param_grid,
                cv=self.kfold,
                scoring=self.scoring,
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            # Evaluate on test data
            y_test_pred = best_model.predict(X_test)
            f1 = f1_score(y_test, y_test_pred)

            results.append({
                "Strategy": strategy_name,
                "Model": self.model.__class__.__name__,
                "Best Params": grid_search.best_params_,
                "F1 Score": f1
            })

        # Create and return a DataFrame with results
        results_df = pd.DataFrame(results)
        return results_df
