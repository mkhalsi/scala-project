import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{SparkSession, types}

// Définir la session Spark
val spark = SparkSession.builder()
  .appName("Fraud Detection with Visualization")
  .config("spark.master", "local")
  .getOrCreate()

// Définir le schéma des données
val schema = types.StructType(Array(
  types.StructField("step", types.IntegerType, nullable = true),
  types.StructField("type", types.StringType, nullable = true),
  types.StructField("amount", types.DoubleType, nullable = true),
  types.StructField("nameOrig", types.StringType, nullable = true),
  types.StructField("oldbalanceOrg", types.DoubleType, nullable = true),
  types.StructField("newbalanceOrig", types.DoubleType, nullable = true),
  types.StructField("nameDest", types.StringType, nullable = true),
  types.StructField("oldbalanceDest", types.DoubleType, nullable = true),
  types.StructField("newbalanceDest", types.DoubleType, nullable = true),
  types.StructField("isFraud", types.IntegerType, nullable = true),
  types.StructField("isFlaggedFraud", types.IntegerType, nullable = true)
))

// Charger les données à partir du fichier CSV avec le schéma spécifié
val data = spark.read.option("header", "true").schema(schema).csv("/FileStore/tables/frauddetection_csv.csv")

// Créer une vue temporaire pour les données
data.createOrReplaceTempView("table_fraude")
display(spark.sql("select * from table_fraude LIMIT 5"))
// Sélectionner les colonnes pertinentes pour l'analyse (montant, ancien solde, nouveau solde)
val assembler = new VectorAssembler()
  .setInputCols(Array("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"))
  .setOutputCol("features")

val assembledData = assembler.transform(data)

// Appliquer K-means
val kmeans = new KMeans().setK(2).setSeed(1L)
val model = kmeans.fit(assembledData)

// Prédire les clusters pour chaque transaction
val predictions = model.transform(assembledData)

// Définition de la fonction pour calculer le pourcentage de transactions frauduleuses
def calculateFraudPercentage(predictions: DataFrame): DataFrame = {
  // Calculer le nombre total de transactions
  val totalTransactions = predictions.count()

  // Calculer le nombre de transactions frauduleuses
  val fraudTransactions = predictions.filter(col("prediction") === 1).count()
   // Calculer le nombre de transactions non frauduleuses
  val nonFraudTransactions = totalTransactions - fraudTransactions
   // Créer un DataFrame avec les pourcentages
  val fraudPercentage = Seq(
    ("Frauduleux", fraudTransactions.toDouble / totalTransactions * 100),
    ("Non-Frauduleux", nonFraudTransactions.toDouble / totalTransactions * 100)
  ).toDF("Transaction Type", "Pourcentage")

  fraudPercentage
}

// Calculer le pourcentage de transactions frauduleuses par rapport aux transactions non frauduleuses
val fraudPercentage = calculateFraudPercentage(predictions)
// Créer une vue temporaire pour les pourcentages de fraudes
fraudPercentage.createOrReplaceTempView("fraud_porcentage")
display(spark.sql("select * from fraud_porcentage"))
 
 //Afficher les transactions prédites comme frauduleuses
predictions.createOrReplaceTempView("predicted_transactions")
display(spark.sql("select * from predicted_transactions"))

 

