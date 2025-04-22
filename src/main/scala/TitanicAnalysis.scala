import org.apache.spark.sql.{SparkSession, functions => F}

import org.apache.spark.sql.functions._

object TitanicAnalysis {

  def main(args: Array[String]): Unit = {

    // Create a Spark session
    val spark = SparkSession.builder()
      .appName("Titanic Analysis")
      .config("spark.master", "local")
      .getOrCreate()

    // Load the Titanic dataset
    val df = spark.read.option("header", "true").csv("train.csv")

    // Show the schema to understand the structure of the dataset
    df.printSchema()

    // Question 1: Average ticket fare for each Ticket class
    val averageFare = df.groupBy("Pclass")
      .agg(F.avg("Fare").alias("Average_Fare"))
    averageFare.show()

    // Question 2: Survival percentage for each Ticket class
    val survivalRate = df.groupBy("Pclass")
      .agg(
        (F.count(when(col("Survived") === 1, 1))
          .divide(F.count("*"))
          .multiply(100))
          .alias("Survival_Percentage")
      )
    survivalRate.show()

    // Find the class with the highest survival rate
    val highestSurvivalClass = survivalRate.orderBy(col("Survival_Percentage").desc).limit(1)
    highestSurvivalClass.show()

    // Question 3: Find the number of passengers who could possibly be Rose DeWitt Bukater
    val roseCandidates = df.filter(
      col("Age") === 17 &&
        col("Sex") === "female" &&
        col("Pclass") === 1 &&
        col("SibSp") === 1 // One parent onboard (no siblings/spouse)
    )
    println(s"Possible Rose candidates: ${roseCandidates.count()}")

    // Question 4: Find the number of passengers who could possibly be Jack Dawson
    val jackCandidates = df.filter(
      col("Age") >= 19 &&
        col("Age") <= 20 &&
        col("Pclass") === 3 &&
        col("SibSp") === 0 && // No relatives onboard
        col("Parch") === 0 // No parents/children onboard
    )
    println(s"Possible Jack candidates: ${jackCandidates.count()}")

    // Question 5: Split the age for every 10 years and analyze the relation between age and ticket fare
    val dfAgeGroups = df.withColumn(
      "Age_Group",
      when(col("Age") <= 10, "1-10")
        .when(col("Age") > 10 && col("Age") <= 20, "11-20")
        .when(col("Age") > 20 && col("Age") <= 30, "21-30")
        .when(col("Age") > 30 && col("Age") <= 40, "31-40")
        .when(col("Age") > 40 && col("Age") <= 50, "41-50")
        .when(col("Age") > 50 && col("Age") <= 60, "51-60")
        .when(col("Age") > 60 && col("Age") <= 70, "61-70")
        .otherwise("71+")
    )

    // Calculate the average fare and survival percentage for each age group
    val ageFareRelation = dfAgeGroups.groupBy("Age_Group")
      .agg(
        F.avg("Fare").alias("Average_Fare"),
        (F.count(when(col("Survived") === 1, 1))
          .divide(F.count("*"))
          .multiply(100))
          .alias("Survival_Percentage")
      )
    ageFareRelation.show()

    // Find the age group most likely to survive
    val mostLikelySurvived = ageFareRelation.orderBy(col("Survival_Percentage").desc).limit(1)
    mostLikelySurvived.show()

    // Stop the Spark session
    spark.stop()
  }
}
