
import spark.implicits._
import org.apache.spark.sql.functions._

// Prédire les notes pour des paires utilisateur-film spécifiques
val specificPairs = Seq(
  (1, 1),    // Toy Story
  (1, 50),   // Star Wars
  (1, 100),  // Fargo
  (1, 172),  // Empire Strikes Back
  (1, 258),  // Contact
  (1, 302),  // L.A. Confidential
  (1, 313),  // Titanic
  (1, 318)   // Schindler's List
).toDF("userId", "movieId")

// Appliquer le modèle entraîné
val predictions = bestALS.transform(specificPairs)

// Joindre avec les titres
predictions
  .join(moviesClean, "movieId")
  .select("userId", "title", "prediction")
  .orderBy(desc("prediction"))
  .show(10, false)