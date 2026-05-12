# 🎬 Projet 3 — Système de Recommandation avec Spark MLlib

## Description
Système de recommandation de films basé sur l'algorithme ALS (Alternating Least Squares)
implémenté dans Apache Spark MLlib, utilisant le dataset MovieLens 100K.

## Résultats
| Métrique | Valeur |
|----------|--------|
| RMSE (défaut) | 0.9209 |
| RMSE (après CV) | 0.8212 |
| MAE | 0.7291 |
| R² | 0.3323 |
| Amélioration | 10.83% |
| Meilleur rank | 5 |

## Technologies
- Apache Spark 3.5.8
- Scala 2.12.18
- Spark MLlib (ALS)
- Spark SQL
- MovieLens 100K

## Run Website
- in cmd navigate to project folder MLProject\web 
- the run in with npx : npx serve .
- or run it with python : python -m http.server 8000
