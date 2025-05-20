import json, numpy as np
from pyspark.streaming.context import StreamingContext
from pyspark.streaming.dstream import DStream
from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.ml.linalg import DenseVector
from transforms import Transforms
from trainer import SparkConfig

class DataLoader:
    def __init__(self,
                 sc: SparkContext,
                 ssc: StreamingContext,
                 sql_ctx: SQLContext,
                 spark_conf: SparkConfig,
                 transforms: Transforms):

        self.sc   = sc
        self.ssc  = ssc
        self.sql  = sql_ctx
        self.conf = spark_conf
        self.tfms = transforms

        self.stream = self.ssc.socketTextStream(
            hostname=self.conf.stream_host,
            port=self.conf.port)

    # ------------- build pipeline ------------- #
    def parse_stream(self) -> DStream:
        js_stream = self.stream.map(lambda line: json.loads(line))
        js_stream = js_stream.flatMap(lambda x: x.values())
        js_stream = js_stream.map(lambda x: list(x.values()))
        # x[:-1] = 3072 pixel,  x[-1] = label
        pixels    = js_stream.map(
            lambda x: [np.array(x[:-1]).reshape(3,32,32).transpose(1,2,0).astype(np.uint8),
                       int(x[-1])])
        pixels    = self.preprocess(pixels, self.tfms)
        return pixels

    @staticmethod
    def preprocess(stream: DStream, tfms: Transforms) -> DStream:
        stream = stream.map(
            lambda x: [tfms.transform(x[0]).reshape(-1).tolist(), x[1]])
        stream = stream.map(lambda x: [DenseVector(x[0]), x[1]])
        return stream