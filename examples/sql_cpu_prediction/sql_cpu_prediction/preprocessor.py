## preprocessor.py

import pandas as pd
from typing import Dict, Any

class Preprocessor:
    """
    Preprocessor class for converting SQL queries into a DataFrame format suitable for model training and prediction.
    """

    def extract_features(self, sql_query: str) -> Dict[str, Any]:
        """
        Extracts features from a SQL query string that are relevant for CPU cost prediction.

        Parameters:
        sql_query (str): The SQL query from which to extract features.

        Returns:
        Dict[str, Any]: A dictionary containing the extracted features of the SQL query.
        """
        features = {
            "query_length": len(sql_query),
            "select_count": sql_query.lower().count("select"),
            "join_count": sql_query.lower().count("join"),
            "subquery_count": sql_query.count("("),
            "aggregate_functions_count": sum(sql_query.lower().count(func) for func in ["sum", "count", "avg", "min", "max"])
        }
        return features

    def preprocess_sql(self, sql_query: str) -> pd.DataFrame:
        """
        Converts a SQL query string into a DataFrame by extracting features relevant for CPU cost prediction.

        Parameters:
        sql_query (str): The SQL query to preprocess.

        Returns:
        pd.DataFrame: A DataFrame containing the preprocessed features of the SQL query.
        """
        features = self.extract_features(sql_query)
        features_df = pd.DataFrame([features])
        return features_df

# Example usage
if __name__ == "__main__":
    preprocessor = Preprocessor()
    sample_query = "SELECT * FROM users INNER JOIN orders ON users.id = orders.user_id"
    processed_df = preprocessor.preprocess_sql(sample_query)
    print(processed_df)
