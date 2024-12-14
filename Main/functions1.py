# Prompt 1: "how do i make a pipeline to implement kfold and gridsearch alongside and ml model training like lr"
# ChatGPT, 2024, Prompt 1: https://chatgpt.com/share/675c2ac6-e4bc-8001-8b96-aa8f5c2a3d5e
# Prompt 2: "can you turn this into a class and show me how to use it"
# ChatGPT, 2024, Prompt 2: https://chatgpt.com/share/675c2ac6-e4bc-8001-8b96-aa8f5c2a3d5e

import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import f1_score, classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class ModelPipelineWithResampling:
    def __init__(self, model, param_grid, scoring='f1'):
        """
        Initialize the ModelPipelineWithResampling with the model, hyperparameter grid, and scoring metric.

        Args:
            model: The machine learning model (e.g., Logistic Regression, etc.).
            param_grid: Dictionary of hyperparameters for GridSearchCV.
            scoring: Metric used for GridSearchCV. Default is 'f1'.
        """
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring

        # Resampling techniques
        self.smote = SMOTE(sampling_strategy=1, random_state=42)
        self.undersample = RandomUnderSampler(sampling_strategy=1, random_state=42)

        # Cross-validation
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        # Attributes to store GridSearchCV and the best model
        self.grid_search = None
        self.best_model = None

    def fit_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate the model in three scenarios: oversampling, undersampling, and without resampling.

        Args:
            X_train: Training feature set.
            y_train: Training labels.
            X_test: Test feature set.
            y_test: Test labels.

        Returns:
            A dataframe containing F1 scores for each scenario.
        """
        results = []
        scenarios = {
            'oversample': [('over', self.smote), ('model', self.model)],
            'undersample': [('under', self.undersample), ('model', self.model)],
            'no_resample': [('model', self.model)]
        }

        for scenario, steps in scenarios.items():
            pipeline = Pipeline(steps=steps)
            self.grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=self.param_grid,
                cv=self.kfold,
                scoring=self.scoring,
                n_jobs=-1
            )

            self.grid_search.fit(X_train, y_train)
            self.best_model = self.grid_search.best_estimator_

            # Evaluate on test data
            y_test_pred = self.best_model.predict(X_test)
            f1 = f1_score(y_test, y_test_pred)

            results.append({
                'scenario': scenario,
                'f1_score': f1
            })

            print(f"Scenario: {scenario}")
            print(f"Best Parameters: {self.grid_search.best_params_}")
            print("Classification Report:")
            print(classification_report(y_test, y_test_pred))

        return pd.DataFrame(results)

    def get_best_model(self):
        """
        Retrieve the best model from the GridSearchCV after training.

        Returns:
            The best model found during GridSearchCV.
        """
        if self.best_model is None:
            raise ValueError("The model has not been trained yet. Call fit_and_evaluate() first.")
        return self.best_model