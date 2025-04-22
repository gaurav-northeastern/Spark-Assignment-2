import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}

object TitanicML {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("Titanic ML Prediction")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val trainPath = "resources/train.csv"
    val testPath = "resources/test.csv"

    val trainRaw = spark.read.option("header", "true").option("inferSchema", "true").csv(trainPath)
    val testRaw = spark.read.option("header", "true").option("inferSchema", "true").csv(testPath)

    // Handle missing values
    val ageMedian = trainRaw.stat.approxQuantile("Age", Array(0.5), 0.0)(0)
    val fareMedian = trainRaw.stat.approxQuantile("Fare", Array(0.5), 0.0)(0)

    val train = trainRaw.na.fill(Map("Age" -> ageMedian, "Fare" -> fareMedian, "Embarked" -> "S"))
    val test = testRaw.na.fill(Map("Age" -> ageMedian, "Fare" -> fareMedian, "Embarked" -> "S"))

    // Indexing categorical columns
    val sexIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndexed").fit(train)
    val embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndexed").fit(train)

    // Assemble features
    val assembler = new VectorAssembler()
      .setInputCols(Array("Pclass", "SexIndexed", "Age", "SibSp", "Parch", "Fare", "EmbarkedIndexed"))
      .setOutputCol("features")

    // Random Forest model
    val rf = new RandomForestClassifier()
      .setLabelCol("Survived")
      .setFeaturesCol("features")
      .setNumTrees(100)

    // Pipeline
    val pipeline = new Pipeline().setStages(Array(sexIndexer, embarkedIndexer, assembler, rf))

    // Fit model
    val model = pipeline.fit(train)

    // Predict on training data
    val predictions = model.transform(train)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Survived")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(f"\nTraining Accuracy: ${accuracy * 100}%.2f%%")

    // Predict on test data (for generating output file)
    val testPredictions = model.transform(test)
    val output = testPredictions.select($"PassengerId", $"prediction".cast("Int").alias("Survived"))

    output.write.option("header", "true").csv("resources/titanic_predictions")

    spark.stop()
  }
}
