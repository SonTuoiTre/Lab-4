"""
Nhận DStream → huấn luyện mô hình theo mini-batch.
"""
import pyspark
from pyspark.context import SparkContext
from pyspark.streaming.context import StreamingContext
from pyspark.sql.context import SQLContext
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import StructType, StructField, IntegerType
from transforms import Transforms
from dataloader import DataLoader

# ---------- Spark run-time config ---------- #
class SparkConfig:
    appName        = "CIFAR-Stream"
    receivers      = 4
    host           = "local"          # master URL
    stream_host    = "localhost"
    port           = 6100
    batch_interval = 2                # seconds

# ---------- Trainer class ---------- #
class Trainer:
    def __init__(self,
                 model,
                 split: str,
                 spark_conf: SparkConfig,
                 transforms: Transforms):

        self.model      = model
        self.split      = split
        self.conf       = spark_conf
        self.transforms = transforms

        self.sc  = SparkContext(f"{self.conf.host}[{self.conf.receivers}]",
                                self.conf.appName)
        self.ssc = StreamingContext(self.sc, self.conf.batch_interval)
        self.sql = SQLContext(self.sc)

        self.loader = DataLoader(self.sc, self.ssc, self.sql,
                                 self.conf, self.transforms)

    # ------------- train loop ------------- #
    def train(self):
        stream = self.loader.parse_stream()
        stream.foreachRDD(self._train_batch)

        self.ssc.start()
        self.ssc.awaitTermination()

    # per-RDD callback
    def _train_batch(self, timestamp, rdd: pyspark.RDD):
        if rdd.isEmpty():
            return

        schema = StructType([
            StructField("image", VectorUDT(),  True),
            StructField("label", IntegerType(), True)])

        df = self.sql.createDataFrame(rdd, schema)
        preds, acc, prec, rec, f1 = self.model.train(df)

        print("-"*32, f"\nBatch @ {timestamp}")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        print(f"Batch size RDD: {rdd.count()}")
        print("-"*32)

    # predict-only flow (giữ lại nếu cần)
    # def predict(self): ...
