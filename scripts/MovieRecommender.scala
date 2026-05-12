

import spark.implicits._
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions._
import java.io._
import scala.collection.JavaConverters._

// ── TÂCHE 1 : Chargement et exploration du dataset ─────────────────────────
val ratings = spark.read
  .option("sep", "\t")
  .option("inferSchema", "true")
  .csv("C:/MLProject/data/ml-100k/u.data")
  .toDF("userId", "movieId", "rating", "timestamp")

println("=== SCHEMA ===")
ratings.printSchema()
println(s"Total ratings : ${ratings.count()}")
println(s"Total users   : ${ratings.select("userId").distinct().count()}")
println(s"Total movies  : ${ratings.select("movieId").distinct().count()}")
ratings.groupBy("rating").count().orderBy("rating").show()
ratings.describe("rating").show()

// ── TÂCHE 2 : Préparation des données ──────────────────────────────────────
// Sélection des colonnes utiles + suppression des valeurs nulles
val data = ratings
  .select("userId", "movieId", "rating")
  .na.drop()

println(s"Lignes avant nettoyage : ${ratings.count()}")
println(s"Lignes après nettoyage : ${data.count()}")

// ── TÂCHE 3 : Séparation train / test ──────────────────────────────────────
// 80% entraînement, 20% test, seed=42 pour reproductibilité
val Array(training, test) = data.randomSplit(Array(0.8, 0.2), seed = 42)
println(s"Ensemble entraînement : ${training.count()} notes")
println(s"Ensemble test         : ${test.count()} notes")

// ── TÂCHE 4 : Entraînement du modèle ALS ───────────────────────────────────
// ALS = Alternating Least Squares (filtrage collaboratif)
val als = new ALS()
  .setUserCol("userId")
  .setItemCol("movieId")
  .setRatingCol("rating")
  .setMaxIter(10)
  .setRegParam(0.1)
  .setRank(10)
  .setColdStartStrategy("drop") // ignore les utilisateurs inconnus

val model = als.fit(training)
println(s"Modèle entraîné ! Rank = ${model.rank}")

// ── TÂCHE 5 : Évaluation — RMSE, MAE, R² ───────────────────────────────────
val predictions = model.transform(test)

val rmseEval = new RegressionEvaluator()
  .setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")
val maeEval = new RegressionEvaluator()
  .setMetricName("mae").setLabelCol("rating").setPredictionCol("prediction")
val r2Eval = new RegressionEvaluator()
  .setMetricName("r2").setLabelCol("rating").setPredictionCol("prediction")

val rmse = rmseEval.evaluate(predictions)
val mae  = maeEval.evaluate(predictions)
val r2   = r2Eval.evaluate(predictions)

println(f"RMSE : $rmse%.4f")
println(f"MAE  : $mae%.4f")
println(f"R²   : $r2%.4f")

// ── TÂCHE 6 : Optimisation par cross-validation ─────────────────────────────
// Test de 18 configurations (3×rank × 2×maxIter × 3×regParam)
val als2 = new ALS()
  .setUserCol("userId").setItemCol("movieId")
  .setRatingCol("rating").setColdStartStrategy("drop")

val paramGrid = new ParamGridBuilder()
  .addGrid(als2.rank,     Array(5, 10, 15))
  .addGrid(als2.maxIter,  Array(5, 10))
  .addGrid(als2.regParam, Array(0.01, 0.1, 0.5))
  .build()

val cv = new CrossValidator()
  .setEstimator(als2)
  .setEvaluator(rmseEval)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)
  .setParallelism(2)

val cvModel = cv.fit(data)
val bestALS = cvModel.bestModel
  .asInstanceOf[org.apache.spark.ml.recommendation.ALSModel]

val bestRmse = rmseEval.evaluate(bestALS.transform(test))
println(f"Meilleur RMSE (CV) : $bestRmse%.4f")
println(s"Meilleur rank      : ${bestALS.rank}")

// ── TÂCHE 7 : Recommandations pour utilisateurs spécifiques ─────────────────
val recs = bestALS.recommendForAllUsers(10)

val flatRecs = recs
  .withColumn("rec", explode($"recommendations"))
  .select($"userId", $"rec.movieId".alias("movieId"), $"rec.rating".alias("score"))

println(s"Total recommandations : ${flatRecs.count()}")

// Afficher top 10 pour utilisateurs spécifiques
flatRecs.filter($"userId" === 1).orderBy(desc("score")).show(10)
flatRecs.filter($"userId" === 42).orderBy(desc("score")).show(10)

// Joindre avec les titres de films
val movies = spark.read
  .option("sep", "|").option("inferSchema", "true")
  .csv("C:/MLProject/data/ml-100k/u.item")
val moviesClean = movies.select($"_c0".alias("movieId"), $"_c1".alias("title"))

val finalRecs = flatRecs.join(moviesClean, "movieId")
finalRecs.filter($"userId" === 1).orderBy(desc("score")).show(10, false)

// ── TÂCHE 8 : Analyse de la matrice de facteurs latents ─────────────────────
val userFactors = bestALS.userFactors
val itemFactors = bestALS.itemFactors
println(s"Matrice utilisateurs : ${userFactors.count()} x ${bestALS.rank}")
println(s"Matrice films        : ${itemFactors.count()} x ${bestALS.rank}")
userFactors.show(5, truncate = false)

// Top films par facteur latent 0
itemFactors
  .withColumn("factor0", $"features"(0))
  .join(moviesClean, $"id" === $"movieId")
  .select("title", "factor0")
  .orderBy(desc("factor0"))
  .show(10, false)

// ── EXPORT CSV ────────────────────────────────────────────────────────────────
new File("C:/MLProject/web/data").mkdirs()

val pw = new PrintWriter("C:/MLProject/web/data/recommendations.csv")
pw.println("userId,title,score")
finalRecs.orderBy($"userId", desc("score"))
  .toLocalIterator.asScala.foreach { r =>
    val userId = r.getInt(1)
    val score  = r.getFloat(2)
    val title  = r.getString(3).replace(",", " ")
    pw.println(s"$userId,$title,$score")
  }
pw.close()
println("Export terminé !")