package data

import org.apache.spark.sql.{DataFrame, functions}
import org.apache.spark.sql.functions.{avg, count, sum}

object DataJoiner {
  /** Join cleaned news + stock by date (simplified aggregation) */
  def joinByDate(cleanedNewsDF: DataFrame, cleanedStockDF: DataFrame): DataFrame = {
    val dailyNews = cleanedNewsDF
      .groupBy("newsDate")
      .agg(
        count("newsId").alias("dailyNewsCount"),
        avg("sentiment").alias("dailyAvgSentiment")
      )
      .withColumnRenamed("newsDate", "date")

    val dailyStock = cleanedStockDF
      .groupBy("date")
      .agg(
        avg("dailyReturn").alias("dailyAvgReturn"),
        sum("volume").alias("dailyTotalVolume")
      )

    dailyNews.join(dailyStock, Seq("date"), "inner").cache()
  }
}