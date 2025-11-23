package data

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col

// Case class for stock data (simplified schema)
case class Stock(
  stockCode: String,
  date: String,
  open: Double,
  high: Double,
  low: Double,
  close: Double,
  volume: Long
)

object DataReader {
  /** Read news from MongoDB */
  def readNews(spark: SparkSession): DataFrame = {
    import spark.implicits._
    spark.read
      .format("mongodb")
      .option("spark.mongodb.input.collection", "news_collection")
      .load()
      .select(
        col("_id").cast("string").alias("newsId"),
        col("publishTime").cast("timestamp").alias("newsTime"),
        col("source").alias("newsSource"),
        col("sentimentScore").cast("double").alias("sentiment")
      )
  }

  /** Read stock from HDFS CSV */
  def readStock(spark: SparkSession, hdfsPath: String = "hdfs:///project/market/ohlc/*.csv"): DataFrame = {
    import spark.implicits._
    spark.read
      .format("csv")
      .option("header", "true")
      .option("inferSchema", "false")
      .load(hdfsPath)
      .map(row => Stock(
        row.getString(0),
        row.getString(1),
        row.getString(2).toDouble,
        row.getString(3).toDouble,
        row.getString(4).toDouble,
        row.getString(5).toDouble,
        row.getString(6).toLong
      ))
      .toDF()
  }
}