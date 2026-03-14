import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class Transformer:
    def __init__(self):
        self.DROP_COLUMNS = [
            "marital",
            "job",
            "education",
            "poutcome",
            "contact",
        ]
        self.BINARY_FEATURES = [
            "housing",
            "loan",
            "default",
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(self.DROP_COLUMNS, axis=1)
        df = self._map_binary_column_to_int(df)
        df = self._map_month_to_int(df)

        return df

    def _map_binary_column_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.BINARY_FEATURES:
            df[col] = df[col].map({"yes": 1, "no": 0})
        return df

    def _map_month_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        month_mapping = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        df["month"] = df["month"].map(month_mapping)

        return df
    
    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        # Separate the classes
        df_y0 = df[df["y"] == 0].copy()
        df_y1 = df[df["y"] == 1].copy()

        # Find the smaller class size
        min_size = len(df_y1)

        # Randomly sample from each class
        df_y0_balanced = df_y0.sample(n=min_size, random_state=42)

        # Concatenate back together
        df_balanced = pd.concat([df_y0_balanced, df_y1])

        # Shuffle the dataset
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

        return df_balanced
    
def train_model(df: pd.DataFrame, target_column: str) -> LogisticRegression:
    MODEL_PARAMS = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # Probability for class 1

    # Create a result DataFrame
    results_df = X_test.copy()
    results_df["prediction"] = predictions
    results_df["predicted_probability"] = probabilities
    results_df["y"] = y_test

    return model, results_df


def map_target_column(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df[target_column] = df[target_column].map({"yes": 1, "no": 0})
    return df