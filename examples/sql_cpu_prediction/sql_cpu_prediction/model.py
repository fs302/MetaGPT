## model.py
import pandas as pd
import numpy as np  # For potential future use in data manipulation and model evaluation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
from typing import Tuple
from preprocessor import Preprocessor  # Assuming preprocessor.py is in the same directory and contains the Preprocessor class

class Model:
    def __init__(self, model_path: str = "model.pkl"):
        self.model_path = model_path
        self.model = RandomForestClassifier()

    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Trains the RandomForest model using the provided features X and target y.
        The trained model is saved to the path specified in self.model_path and returns the accuracy of the model.

        :param X: DataFrame containing the features.
        :param y: Series containing the target.
        :return: Accuracy of the model as a float.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

    def save_model(self):
        """
        Saves the trained model to the path specified in self.model_path.
        """
        dump(self.model, self.model_path)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts the CPU computation cost for the given SQL queries.

        :param X: DataFrame containing the features for prediction.
        :return: A Series containing the predicted CPU costs.
        """
        return pd.Series(self.model.predict(X))

    def classify(self, X: pd.DataFrame, high_cost_threshold: int = 50) -> pd.Series:
        """
        Classifies the SQL queries into predefined categories based on their CPU computation cost.

        :param X: DataFrame containing the features for classification.
        :param high_cost_threshold: The threshold above which costs are considered 'High'.
        :return: A Series containing the classification of the CPU costs.
        """
        predictions = self.model.predict(X)
        return pd.Series(["High" if cost > high_cost_threshold else "Low" for cost in predictions])

# Example usage
if __name__ == "__main__":
    # Assuming the Preprocessor class is properly implemented in preprocessor.py
    preprocessor = Preprocessor()
    # Example SQL query, in practice, you would fetch this from your data source
    example_sql_query = "SELECT * FROM table WHERE condition=True"
    df = preprocessor.preprocess_sql(example_sql_query)

    # Example target variable, in practice, you would fetch this from your data source
    y = pd.Series([10, 20, 30])  # Dummy CPU costs

    model = Model()
    accuracy = model.train(df, y)
    print(f"Model trained with accuracy: {accuracy}")
    model.save_model()  # Save the model after training
    print(model.predict(df))
    print(model.classify(df))
