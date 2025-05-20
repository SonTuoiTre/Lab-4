import warnings, numpy as np
from typing import List
from pyspark.sql.dataframe import DataFrame
from joblibspark import register_spark

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils import parallel_backend

warnings.filterwarnings("ignore")
register_spark()

class LogisticRegressionModel:
    """
    Logistic Regression đa lớp (solver='saga') – fit lại trên từng batch.
    """
    def __init__(self, max_iter:int = 1000, C:float = 1.0):
        self.model = LogisticRegression(
            penalty="l2",
            C=C,
            solver="saga",
            multi_class="multinomial",
            max_iter=max_iter,
            n_jobs=-1,
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
