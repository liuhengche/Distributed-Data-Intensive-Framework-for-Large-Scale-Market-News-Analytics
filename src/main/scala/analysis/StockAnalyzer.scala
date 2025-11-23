package analysis

import org.apache.spark.sql.{DataFrame, functions}
import org.apache.spark.sql.functions.{avg, col, max, min, stddev}

object StockAnalyzer {
  /** Basic return stats (mean/std/max/min) */
  def returnStats(cleanedStockDF: DataFrame): DataFrame = {
    cleanedStockDF
      .agg(
        avg("dailyReturn").alias("avgReturn"),
        stddev("dailyReturn").alias("stdReturn"),
        max("dailyReturn").alias("maxReturn"),
        min("dailyReturn").alias("minReturn")
      )
      .select(
        (col("avgReturn") * 100).alias("avgReturn(%)"),
        (col("stdReturn") * 100).alias("stdReturn(%)"),
        (col("maxReturn") * 100).alias("maxReturn(%)"),
        (col("minReturn") * 100).alias("minReturn(%)")
      )
  }

  /** Single stock trend (e.g., APPL) */
  def stockTrend(cleanedStockDF: DataFrame, stockCode: String = "APPL"): DataFrame = {
    cleanedStockDF
      .filter(col("stockCode") === stockCode)
      .select("date", "open", "close", "volume")
      .orderBy("date")
  }
}