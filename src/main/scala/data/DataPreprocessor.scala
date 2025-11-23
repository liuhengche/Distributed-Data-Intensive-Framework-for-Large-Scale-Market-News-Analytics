package data

import org.apache.spark.sql.{DataFrame, functions}
import org.apache.spark.sql.functions.{col, date_format}

object DataPreprocessor {
  /** Clean news data (filter + add date) */
  def cleanNews(rawNewsDF: DataFrame): DataFrame = {
    rawNewsDF
      .filter(col("newsId").isNotNull && col("newsTime").isNotNull && col("sentiment").isNotNull)
      .filter(col("sentiment").between(0.0, 1.0))
      .withColumn("newsDate", date_format(col("newsTime"), "yyyy-MM-dd"))
      .cache()
  }

  /** Clean stock data (filter + add daily return) */
  def cleanStock(rawStockDF: DataFrame): DataFrame = {
    rawStockDF
      .filter(
        col("open") > 0 && col("close") > 0 && col("volume") > 0 &&
        col("high") >= col("open") && col("high") >= col("close") &&
        col("low") <= col("open") && col("low") <= col("close")
      )
      .withColumn("dailyReturn", (col("close") - col("open")) / col("open"))
      .filter(col("dailyReturn").between(-0.1, 0.1))
      .cache()
  }
}