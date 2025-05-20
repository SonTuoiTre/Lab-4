from typing import List
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from joblibspark import register_spark
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils import parallel_backend
from pyspark.sql.dataframe import DataFrame

register_spark()

class RandomForest:
    """
    Random-Forest đa lớp huấn luyện trên mỗi mini-batch Spark.
    """
    def __init__(self,
                 n_estimators: int = 200,
                 max_depth: int | None = None,
                 n_jobs: int = -1) -> None:

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=42)

    # ---------------- streaming helpers ---------------- #
    def _prepare(self, df: DataFrame):
        X = np.array(df.select("image").collect()).reshape(-1, 32*32*3)
        y = np.array(df.select("label").collect()).reshape(-1)
        return X, y

    def train(self, df: DataFrame) -> List:
        X, y = self._prepare(df)
        with parallel_backend("spark", n_jobs=8):
            self.model.fit(X, y)

        preds = self.model.predict(X)
        acc  = accuracy_score(y, preds)
        prec = precision_score(y, preds, average="macro")
        rec  = recall_score(y, preds, average="macro")
        f1   = 2*prec*rec/(prec+rec)

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