import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, dayofweek, monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, FloatType, IntegerType, DateType, TimestampType



config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """
    Create a new or use the existing spark session.
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Load the songs data from the s3 bucket, processes it and loads the data back 
    into S3 bucket as a set of two dimensional tables e.g songs and the artists table

    :param spark: spark session
    :param input_data: s3 bucket path where the songs data json files exist
    :param output_data: s3 bucket path to save the dimensional tables in parquet format    
    """
    
    
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')
    
    song_schema = StructType([
        StructField("num_songs", IntegerType()),
        StructField("artist_id", StringType()),
        StructField("artist_latitude", DoubleType()),
        StructField("artist_longitude", DoubleType()),
        StructField("artist_location", StringType()),
        StructField("artist_name", StringType()),
        StructField("song_id", StringType()),
        StructField("title", StringType()),
        StructField("duration", FloatType()),
        StructField("year", IntegerType())
    ])
    
    # read song data file
    df = spark.read.json(song_data, schema=song_schema)
    
    
    df.createOrReplaceTempView("song_table_temp")

    # extract columns to create songs table
    songs_table = df.select("title", "artist_id", "year", "duration").dropDuplicates().withColumn("song_id", monotonically_increasing_id())
    
    # write songs table to parquet files partitioned by year and artist
    songs_table = songs_table.write.mode("overwrite").partitionBy("year", "artist_id").parquet("{}songs".format(output_data))

    # extract columns to create artists table
    artists_table =  df.select("artist_id", "artist_name", "artist_location", "artist_latitude", "artist_longitude").dropDuplicates()
    
    # write artists table to parquet files
    artists_table = artists_table.write.mode("overwrite").parquet("{}artists".format(output_data))


def process_log_data(spark, input_data, output_data):
    """
    Load the log data from the s3 bucket, processes it and creates a set of three 
    dimensional tables e.g user table, time table and songsplay table

    :param spark: spark session
    :param input_data: s3 bucket path where the log data json files exist
    :param output_data: s3 bucket path to save the dimensional tables in parquet format    
    """
    
    # get filepath to log data file
    log_data = os.path.join(input_data, 'log_data/*/*/*.json')

    # read log data file
    df = spark.read.json(log_data).dropDuplicates()
    
    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # extract columns for users table    
    users_table = df.select("userId", "firstName", "lastName", "gender", "level").dropDuplicates()
    
    # write users table to parquet files
    users_table = users_table.write.mode("overwrite").parquet("{}users".format(output_data))

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: int(x) / 1000, IntegerType())
    df = df.withColumn("time_stamp", get_timestamp("ts"))
    
    # create datetime column from newly created timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x), TimestampType())
    df = df.withColumn("start_time", get_datetime("time_stamp"))
    
    df = df.withColumn("hour", hour("start_time")) \
            .withColumn("day", dayofmonth("start_time")) \
            .withColumn("week", weekofyear("start_time")) \
            .withColumn("month", month("start_time")) \
            .withColumn("year", year("start_time")) \
            .withColumn("weekday", dayofweek("start_time")) \
    
    time_table = df.select("start_time", "hour", "day", "week", "month", "year", "weekday")

    # write time table to parquet files partitioned by year and month
    time_table =  time_table.write.mode("overwrite").partitionBy("year", "month").parquet("{}time".format(output_data))

    # read in song data to use for songplays table
    song_df = spark.sql("SELECT song_id, artist_id, artist_name, title FROM song_table_temp")

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = df.join(song_df, (df.artist == song_df.artist_name) & (df.song == song_df.title), "inner").distinct() \
                        .select("start_time", "userId", "level", "sessionId", "location", "userAgent", "song_id", "artist_id", "year", "month") \
                        .withColumn("songplay_id", monotonically_increasing_id())

    # write songplays table to parquet files partitioned by year and month
    songplays_table = songplays_table.write.mode("overwrite").partitionBy('year', 'month').parquet("{}songplays".format(output_data))



def main():
    """
    Load the songs and logs data from S3, processes them using Spark, and 
    loads the data back into S3 as a set of dimensional tables in parquet format
    """
    
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = ""
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
