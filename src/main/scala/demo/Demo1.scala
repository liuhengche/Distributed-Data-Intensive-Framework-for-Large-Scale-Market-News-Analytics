package demo

import config.SparkConfig
import data.{DataJoiner, DataPreprocessor, DataReader}
import analysis.{CombinedAnalyzer, NewsAnalyzer, StockAnalyzer}
import util.Util
import org.apache.spark.sql.SparkSession

object Demo1_FullWorkflow {
  def main(args: Array[String]): Unit = {
    // 1. Initialize Spark
    val spark: SparkSession = SparkConfig.getSparkSession(appName = "FullWorkflowDemo")
    import spark.implicits._

    try {
      // 2. Read data
      val rawNews = DataReader.readNews(spark)
      val rawStock = DataReader.readStock(spark)
      Util.printDF(rawNews, "Raw News Sample")
      Util.printDF(rawStock, "Raw Stock Sample")

      // 3. Clean data
      val cleanNews = DataPreprocessor.cleanNews(rawNews)
      val cleanStock = DataPreprocessor.cleanStock(rawStock)
      println(s"\nCleaned News Count: ${cleanNews.count()}")
      println(s"Cleaned Stock Count: ${cleanStock.count()}")

      // 4. Join data
      val joinedDF = DataJoiner.joinByDate(cleanNews, cleanStock)
      Util.printDF(joinedDF, "Joined Data (Date-Aligned)")

      // 5. Analyze
      // News analysis
      val topSources = NewsAnalyzer.topSources(cleanNews)
      val sentimentDist = NewsAnalyzer.sentimentDist(cleanNews)
      Util.printDF(topSources, "Top 10 News Sources")
      Util.printDF(sentimentDist, "News Sentiment Distribution")
      Util.writeToHDFS(topSources, "hdfs:///project/analysis/top_news_sources")

      // Stock analysis
      val returnStats = StockAnalyzer.returnStats(cleanStock)
      val applTrend = StockAnalyzer.stockTrend(cleanStock)
      Util.printDF(returnStats, "Stock Return Statistics")
      Util.printDF(applTrend, "APPL Price/Volume Trend (First 5 Days)")

      // Combined analysis
      val sentimentReturn = CombinedAnalyzer.sentimentVsReturn(joinedDF)
      Util.printDF(sentimentReturn, "Sentiment vs Average Return")

    } finally {
      spark.stop()
      println("\nSpark session closed.")
    }
  }
}