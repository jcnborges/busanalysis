import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from .sparketl import ETLSpark


class TrustProcessing:

    def __init__(self, date):
        self.etlspark = ETLSpark()
        self.date = date

    def __call__(self, *args, **kwargs):
        self.perform()

    def perform(self):
        vehicles = self.vehicles_ingestion(self.date)
        self.save(vehicles, "/data/trusted/vehicles")

        busstops = self.bustops_ingestion(self.date)
        self.save(busstops, "/data/trusted/busstops")

        lines = self.lines_ingestion(self.date)
        self.save(lines, "/data/trusted/lines")

    def vehicles_ingestion(self, period: str):
        return (self.etlspark.sqlContext.read.json(f"/data/raw/{period}/veiculos")
                .select(F.col("COD_LINHA").alias("line_code"),
                        F.date_format(F.unix_timestamp('dthr', 'dd/MM/yyyy HH:mm:ss').cast('timestamp'),
                                      "yyyy-MM-dd HH:mm:ss").alias('event_timestamp'),
                        F.regexp_replace("LAT", ",", ".").cast('double').alias("latitude"),
                        F.regexp_replace("LON", ",", ".").cast('double').alias("longitude"),
                        F.col("VEIC").alias("vehicle")
                        )
                .withColumn("year", F.year("event_timestamp"))
                .withColumn("month", F.month("event_timestamp"))
                .withColumn("day", F.dayofmonth("event_timestamp"))
                .dropDuplicates())

    def lines_ingestion(self, period: str) -> DataFrame:
        return (self.etlspark.extract(f"/data/raw/{period}/linhas")
                .withColumn("service_category", F.col("categoria_servico"))
                .withColumn("line_name", F.col("nome"))
                .withColumn("line_code", F.col("cod"))
                .withColumn("color", F.col("nome_cor"))
                .withColumn("card_only", F.col("somente_cartao"))
                .drop("categoria_servico", "cod", "nome", "nome_cor", "somente_cartao")
                .dropDuplicates())

    def bustops_ingestion(self, period: str) -> DataFrame:
        return (self.etlspark.extract(f"/data/raw/{period}/pontoslinha")
                .withColumn("line_code", F.col("cod"))
                .withColumn("latitude", F.regexp_replace("lat", ",", "."))
                .withColumn("longitude", F.regexp_replace("lon", ",", "."))
                .withColumn("name", F.col("nome"))
                .withColumn("number", F.col("num"))
                .withColumn("line_way", F.col("sentido"))
                .withColumn("seq", F.col("seq"))
                .withColumn("type", F.col("tipo"))
                .drop("GRUPO", "cod", "lat", "lon", "nome", "num", "sentido", "tipo")
                .dropDuplicates())

    def save(self, df: DataFrame, output: str):
        (df.coalesce(1).write.mode('overwrite')
         .partitionBy("year", "month", "day")
         .format("parquet").save(output))
