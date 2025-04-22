name := "TitanicAnalysis"

version := "1.0"

scalaVersion := "2.12.10" // Spark 3.3.0 requires Scala 2.12.x

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.3.0",
  "org.apache.spark" %% "spark-sql"  % "3.3.0",
  "org.apache.spark" %% "spark-mllib" % "3.3.0" // Needed for ML pipeline
)

Compile / mainClass := Some("TitanicML")
