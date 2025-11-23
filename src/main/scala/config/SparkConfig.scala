package config

import org.apache.spark.sql.SparkSession

object SparkConfig {
  /** Create basic SparkSession (simplified for quick use) */
  def getSparkSession(
    appName: String = "NewsStockDemo",
    master: String = "local[*]"  // Default to local mode for demos
  ): SparkSession = {
    SparkSession.builder()
      .appName(appName)
      .master(master)
      .config("spark.mongodb.input.uri", "mongodb://localhost:27017/news_db.news_collection")
      .config("spark.sql.adaptive.enabled", "true")
      .getOrCreate()
  }
}