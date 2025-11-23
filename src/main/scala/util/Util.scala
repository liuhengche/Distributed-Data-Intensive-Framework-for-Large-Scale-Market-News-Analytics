package util

import org.apache.spark.sql.DataFrame

object Util {
  /** Simplified function to write DataFrame to HDFS CSV */
  def writeToHDFS(df: DataFrame, path: String): Unit = {
    df.write
      .format("csv")
      .option("header", "true")
      .mode("overwrite")
      .save(path)
    println(s"Result saved to HDFS: $path")
  }

  /** Print DataFrame with custom message */
  def printDF(df: DataFrame, title: String, limit: Int = 5): Unit = {
    println(s"\n=== $title ===")
    df.show(limit, truncate = false)
  }
}