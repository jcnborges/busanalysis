import math
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window
from .sparketl import ETLSpark

@F.udf(returnType=T.DoubleType())
def interpolate_timestamp(seq: int, event_timestamp: T.TimestampType, next_seq: int, next_event_timestamp: T.TimestampType) -> float:
    try:
        """
        Interpolates a timestamp based on the seq number and time of surrounding rows.
        """

        # Linear interpolation:
        timestamp_diff = (next_event_timestamp - event_timestamp).total_seconds()
        interpolated_seconds = event_timestamp.timestamp() + timestamp_diff / (next_seq - seq)
        return interpolated_seconds
    except Exception as err:
        print(f"Exception has been occurred :{err}")
        print(f"seq: {seq} event_timestamp: {event_timestamp} next_seq: {next_seq} next_event_timestamp: {next_event_timestamp}")

@F.udf(returnType=T.DoubleType())
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    try:
        R: int = 6371000  # radius of Earth in meters
        phi_1 = math.radians(lat1)
        phi_2 = math.radians(lat2)

        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * \
            math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return round(R * c, 2)  # output distance in meters
    except Exception as err:
        print(f"Exception has been occurred :{err}")
        print(f"lon1: {lon1} lat1: {lat1} lon2: {lon2} lat2: {lat2}")

class BusItineraryRefinedProcess:
    
    def __init__(self, year, month, day):
        self.etlspark = ETLSpark()
        self.bus_stops = self.filter_data(year, month, day)

    def perform(self):
        # Convert columns to numeric
        self.bus_stops = self.bus_stops.withColumn("latitude", self.bus_stops["latitude"].cast("double"))
        self.bus_stops = self.bus_stops.withColumn("longitude", self.bus_stops["longitude"].cast("double"))
        self.bus_stops = self.bus_stops.withColumn("seq", self.bus_stops["seq"].cast("int"))

        # Sort values
        self.bus_stops = self.bus_stops.orderBy(["line_code", "itinerary_id", "seq"])

        # Add a row number column to ensure we can drop duplicates based on the first occurrence
        window_spec = Window.partitionBy("line_code", "itinerary_id", "latitude", "longitude", "name", "number", "line_way", "type", "year", "month", "day").orderBy("seq")
        self.bus_stops = self.bus_stops.withColumn("row_num", F.row_number().over(window_spec))

        # Drop duplicates based on the tuple and keep the first occurrence
        self.bus_stops = self.bus_stops.filter(F.col("row_num") == 1).drop("row_num")

        # Select specific columns
        self.bus_stops = self.bus_stops.select("line_code", "itinerary_id", "latitude", "longitude", "name", "number", "line_way", "type", "seq", "year", "month", "day").distinct()

        # Add 'id' column
        self.bus_stops = self.bus_stops.withColumn("id", self.bus_stops["number"].cast("int"))

        # Add next_stop_id, next_stop_latitude, next_stop_longitude using window function
        window_spec = Window.partitionBy("line_code", "itinerary_id").orderBy("seq")

        self.bus_stops = self.bus_stops.withColumn("next_stop_id", F.lag("id", -1).over(window_spec))
        self.bus_stops = self.bus_stops.withColumn("next_stop_latitude", F.lag("latitude", -1).over(window_spec))
        self.bus_stops = self.bus_stops.withColumn("next_stop_longitude", F.lag("longitude", -1).over(window_spec))
        self.bus_stops = self.bus_stops.withColumn("next_stop_delta_s", haversine(F.col("latitude"), F.col("longitude"), F.col("next_stop_latitude"), F.col("next_stop_longitude")))

        # Filter rows where id != next_stop_id or next_stop_id is null
        self.bus_stops = self.bus_stops.filter((F.col("id") != F.col("next_stop_id")) | F.col("next_stop_id").isNull())

        # Add 'seq' and 'max_seq' columns using window function
        self.bus_stops = self.bus_stops.withColumn("seq", F.row_number().over(window_spec) - 1)

        window_spec = Window.partitionBy("line_code", "itinerary_id")
        self.bus_stops = self.bus_stops.withColumn("max_seq", F.max("seq").over(window_spec))

        self.save(self.bus_stops, "/data/refined/bus_itineraries")

    def __call__(self, *args, **kwargs):
        self.perform()

    def filter_data(self, year: str, month: str, day: str) -> DataFrame:
        return (self.etlspark.sqlContext.read.parquet("/data/trusted/busstops")
                .filter(f"year =='{year}' and month=='{month}' and day=='{day}'"))

    @staticmethod
    def save(df: DataFrame, output: str):
        (df.write.mode('overwrite')
         .partitionBy("year", "month", "day")
         .format("parquet").save(output))
        
class BusLineRefinedProcess:
    
    def __init__(self, year, month, day):
        self.etlspark = ETLSpark()
        self.bus_lines = self.filter_data(year, month, day)      
    
    def perform(self):
        self.save(self.bus_lines, "/data/refined/bus_lines")

    def __call__(self, *args, **kwargs):
        self.perform()

    def filter_data(self, year: str, month: str, day: str) -> DataFrame:
        return (self.etlspark.sqlContext.read.parquet("/data/trusted/lines")
                .filter(f"year =='{year}' and month=='{month}' and day=='{day}'"))

    @staticmethod
    def save(df: DataFrame, output: str):
        (df.write.mode('overwrite')
         .partitionBy("year", "month", "day")
         .format("parquet").save(output))

class BusTrackingRefinedProcess:
    def __init__(self, year, month, day, line_code):
        self.etlspark = ETLSpark()
        self.vehicles = self.filter_data(year, month, day, line_code)
        self.bus_itineraries = self.filter_bus_itineraries(year, month, day, line_code)
        self.bus_lines = self.filter_bus_lines(year, month, day, line_code)      
    
    def perform(self):

        # Convert event_timestamp to TimestampType
        self.vehicles = self.vehicles.withColumn("event_timestamp", F.to_timestamp(F.col("event_timestamp")))

        # Create the dim_bus_stop DataFrame
        dim_bus_stop = self.bus_itineraries.select(
            "line_code", "latitude", "longitude", "id"
        ).distinct()

        # Rename columns to avoid ambiguity
        dim_bus_stop = dim_bus_stop.withColumnRenamed(
            "latitude", "bus_stop_latitude"
        ).withColumnRenamed("longitude", "bus_stop_longitude")

        # Join vehicles_df with dim_bus_stop_df on 'line_code'
        joined = self.vehicles.join(
            dim_bus_stop, on="line_code", how="left"
        )

        # Select the desired columns from the joined DataFrame
        map_matching = joined.select(
            "line_code",
            "event_timestamp",
            "latitude",  # From vehicles
            "longitude", # From vehicles
            "vehicle",
            "year",
            "month",
            "day",
            "id",  # From dim_bus_stop
            "bus_stop_latitude",  # From dim_bus_stop
            "bus_stop_longitude"  # From dim_bus_stop
        )        

        # Calculate the haversine distance
        map_matching = map_matching.withColumn("distance", haversine(F.col("latitude"), F.col("longitude"), F.col("bus_stop_latitude"), F.col("bus_stop_longitude")))

        # Filter for distances <= 50 meters
        map_matching = map_matching.filter(F.col("distance") <= 50)

        # Define a window partition by vehicle and event_timestamp
        window = Window.partitionBy("vehicle", "event_timestamp").orderBy("distance")

        # Find the row with minimum haversine distance within each group
        map_matching = map_matching.withColumn("row_num", F.row_number().over(window))

        # Filter for the row with row_num 1 (minimum distance)
        map_matching = map_matching.filter(F.col("row_num") == 1)

        # Calculate the mean event_timestamp for each group using the time window
        map_matching = map_matching.groupBy(
            "line_code",
            "vehicle",
            "year",
            "month",
            "day",
            "id",  # From dim_bus_stop
            F.window("event_timestamp", "30 minutes").alias("time_window")            
        ).agg(
            F.avg("event_timestamp").alias("mean_event_timestamp")
        )

        # Convert the mean_event_timestamp to TimestampType
        map_matching = map_matching.withColumn(
            "mean_event_timestamp",
            map_matching["mean_event_timestamp"].cast(T.TimestampType())
        ).withColumnRenamed("mean_event_timestamp", "event_timestamp")

        # Select only relevant columns for perform the map matching
        map_matching = map_matching.select(
            "line_code",
            "vehicle",
            "year",
            "month",
            "day",
            "id",
            "event_timestamp"
        )

        # Select the bus line itineraries
        bus_stop_itineraries = self.bus_itineraries.select(
            "line_code",
            "itinerary_id",
            "id",
            "seq",
            "max_seq"
        )

        # Join the DataFrames on 'line_code' and 'id'
        bus_itineraries_search = map_matching.join(
            bus_stop_itineraries,
            on=["line_code", "id"], 
            how="inner"  # Change to 'left', 'right', 'outer' if needed
        )

        # Define the window specification
        windowSpec = Window.partitionBy("line_code", "vehicle", "itinerary_id").orderBy("event_timestamp")

        # Order the DataFrame by event_timestamp within each partition
        ordered_df = bus_itineraries_search.withColumn(
            "row_num",
            F.row_number().over(windowSpec)
        ).orderBy("line_code", "vehicle", "itinerary_id", "row_num")        

        # Create the 'next_seq_1' and 'next_seq_2' columns using lead()
        ordered_df = ordered_df.withColumn(
            "next_seq_1",
            F.lead(F.col("seq"), 1, None).over(windowSpec) 
        ).withColumn("next_seq_2",
            F.lead(F.col("seq"), 2, None).over(windowSpec)             
        )

        # Filter rows where seq >= next_seq_1 or seq >= next_seq_2
        filtered_df = ordered_df.filter(
            ((F.col("seq") < F.col("next_seq_1")) & (F.col("seq") < F.col("next_seq_2"))) |
            ((F.col("seq") == F.col("max_seq")) | (F.col("seq") == (F.col("max_seq") - 1)))
        )

        # Add the "generated" column
        filtered_df = filtered_df.withColumn("generated", F.lit(False)) 

        filtered_df = filtered_df.select(
            "line_code",
            "itinerary_id",            
            "vehicle",
            "event_timestamp",
            "seq",
            "year",
            "month",
            "day",
            "generated"
        )        

        c = 0
        while c < 7:

            filtered_df = filtered_df.select(
                "line_code",
                "itinerary_id",            
                "vehicle",
                "event_timestamp",
                "seq",
                "year",
                "month",
                "day",
                "generated"
            )

            # Interpolate and expand the DataFrame
            interpolated_df = filtered_df.withColumn(
                "next_seq", F.lead(F.col("seq"), 1, None).over(windowSpec)
            ).withColumn(
                "next_event_timestamp", F.lead(F.col("event_timestamp"), 1, None).over(windowSpec)
            ).withColumn(
                "interpolated_next_seq", (F.col("seq") + 1)
            )

            # Filter for interpolated points
            interpolated_points = interpolated_df.filter(
                (F.col("next_seq") != 0) & (F.col("next_seq") > F.col("interpolated_next_seq")) 
            )

            print(f"c = {c} | count = {interpolated_points.count()}")

            if interpolated_points.count() == 0:
                break

            # Apply the interpolation UDF (passing columns directly)
            interpolated_points = interpolated_points.withColumn(
                "interpolated_timestamp",
                interpolate_timestamp(                
                    F.col("seq"),
                    F.col("event_timestamp"),
                    F.col("next_seq"),
                    F.col("next_event_timestamp")
                )
            )

            interpolated_points = interpolated_points.withColumn(
                "interpolated_timestamp", F.to_timestamp(F.col("interpolated_timestamp"))
            ).withColumn(
                "generated", F.lit(True)
            ).filter("interpolated_timestamp < next_event_timestamp")

            interpolated_points = interpolated_points.select(
                "line_code",
                "itinerary_id",            
                "vehicle",
                "interpolated_timestamp",  
                "interpolated_next_seq",
                "year",
                "month",
                "day",
                "generated"
            ).withColumnRenamed(
                "interpolated_timestamp", "event_timestamp"
            ).withColumnRenamed(
                "interpolated_next_seq", "seq"
            )

            filtered_df = filtered_df.union(interpolated_points).orderBy("event_timestamp")

            c = c + 1

        joined_df = filtered_df.join(
            bus_stop_itineraries,
            on=["line_code", "itinerary_id", "seq"],
            how="left"  # Change to 'inner', 'right', 'outer' if needed
        )

        joined_df = joined_df.select(
            "line_code",
            "itinerary_id",            
            "vehicle",
            "event_timestamp",
            "id",
            "seq",
            "year",
            "month",
            "day",
            "generated"
        )

        self.save(filtered_df, "/data/refined/bus_tracking")

    def __call__(self, *args, **kwargs):
        self.perform()

    def filter_data(self, year: str, month: str, day: str, line_code: str) -> DataFrame:
        return (self.etlspark.sqlContext.read.parquet("/data/trusted/vehicles")
                .filter(f"year =='{year}' and month=='{month}' and day=='{day}' and line_code == '{line_code}'"))

    def filter_bus_itineraries(self, year: str, month: str, day: str, line_code: str) -> DataFrame:
        return (self.etlspark.sqlContext.read.parquet("/data/refined/bus_itineraries")
                .filter(f"year =='{year}' and month=='{month}' and day=='{day}' and line_code == '{line_code}'"))

    def filter_bus_lines(self, year: str, month: str, day: str, line_code: str) -> DataFrame:
        return (self.etlspark.sqlContext.read.parquet("/data/refined/bus_lines")
                .filter(f"year =='{year}' and month=='{month}' and day=='{day}' and line_code == '{line_code}'"))

    @staticmethod
    def save(df: DataFrame, output: str):
        (df.write.mode('overwrite')
         .partitionBy("year", "month", "day", "line_code")
         .format("parquet").save(output))