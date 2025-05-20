import numpy as np, warnings
from typing import List
warnings.filterwarnings("ignore")

from joblibspark import register_spark
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils import parallel_backend
from pyspark.sql.dataframe import DataFrame

register_spark()

class DecisionTree:
    """
    Cây quyết định CART – gọn nhẹ, train batch-wise.
    """
    def __init__(self, max_depth:int|None = None):
        self.model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=max_depth,
            random_state=42)

    def _prepare(self, df: DataFrame):
        X = np.array(df.select("image").collect()).reshape(-1, 3072)
        y = np.array(df.select("label").collect()).reshape(-1)
        return X, y

    def train(self, df: DataFrame) -> List:
        X, y = self._prepare(df)
        with parallel_backend("spark", n_jobs=8):
            self.model.fit(X, y)

        preds = self.model.predict(X)
        acc   = accuracy_score(y, preds)
        prec  = precision_score(y, preds, average="macro")
        rec   = recall_score(y, preds, average="macro")
        f1    = 2*prec*rec/(prec+rec)
        return preds, acc, prec, rec, f1

    def predict(self, df: DataFrame):
        X, y = self._prepare(df)
        preds = self.model.predict(X)

        acc  = accuracy_score(y, preds)
        prec = precision_score(y, preds, average="macro")
        rec  = recall_score(y, preds, average="macro")
        f1   = 2*prec*rec/(prec+rec)
        cm   = confusion_matrix(y, preds)
        return preds, acc, prec, rec, f1, cm
