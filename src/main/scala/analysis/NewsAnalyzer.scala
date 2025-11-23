package analysis

import org.apache.spark.sql.{DataFrame, functions}
import org.apache.spark.sql.functions.{col, count, desc, hour, when}

object NewsAnalyzer {
  /** Top news sources */
  def topSources(cleanedNewsDF: DataFrame, topN: Int = 10): DataFrame = {
    cleanedNewsDF
      .groupBy("newsSource")
      .agg(count("newsId").alias("articleCount"))
      .orderBy(desc("articleCount"))
      .limit(topN)
  }

  /** Sentiment distribution (Negative/Neutral/Positive) */
  def sentimentDist(cleanedNewsDF: DataFrame): DataFrame = {
    val total = cleanedNewsDF.count()
    cleanedNewsDF
      .withColumn("sentimentLevel",
        when(col("sentiment") <= 0.3, "Negative")
          .when(col("sentiment") <= 0.7, "Neutral")
          .otherwise("Positive")
      )
      .groupBy("sentimentLevel")
      .agg(
        count("newsId").alias("count"),
        (count("newsId") / total * 100).alias("percentage(%)")
      )
  }
}