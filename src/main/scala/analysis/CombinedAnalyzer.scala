package analysis

import org.apache.spark.sql.{DataFrame, functions}
import org.apache.spark.sql.functions.{avg, col, when}

object CombinedAnalyzer {
  /** Sentiment vs stock return (simplified) */
  def sentimentVsReturn(joinedDF: DataFrame): DataFrame = {
    joinedDF
      .withColumn("sentimentType",
        when(col("dailyAvgSentiment") >= 0.6, "Positive")
          .when(col("dailyAvgSentiment") <= 0.4, "Negative")
          .otherwise("Neutral")
      )
      .groupBy("sentimentType")
      .agg(
        avg("dailyAvgReturn").alias("avgReturn"),
        count("date").alias("dayCount")
      )
      .select(
        col("sentimentType"),
        (col("avgReturn") * 100).alias("avgReturn(%)"),
        col("dayCount")
      )
  }
}