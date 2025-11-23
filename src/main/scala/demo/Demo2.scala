package demo

import config.SparkConfig
import data.{DataPreprocessor, DataReader}
import analysis.NewsAnalyzer
import util.Util
import org.apache.spark.sql.SparkSession

object Demo2_SimpleAnalysis {
  def main(args: Array[String]): Unit = {
    // Simplified: Only analyze news sentiment (no stock data)
    val spark: SparkSession = SparkConfig.getSparkSession(appName = "SimpleNewsDemo")

    try {
      // Read → Clean → Analyze (news only)
      val rawNews = DataReader.readNews(spark)
      val cleanNews = DataPreprocessor.cleanNews(rawNews)
      val sentimentDist = NewsAnalyzer.sentimentDist(cleanNews)

      Util.printDF(sentimentDist, "Simple News Sentiment Analysis")
      Util.writeToHDFS(sentimentDist, "hdfs:///project/analysis/simple_sentiment_dist")

    } finally {
      spark.stop()
      println("Simple demo completed.")
    }
  }
}