from pyspark.sql import SparkSession
from pathlib import Path
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
#%%

# Criar/obter SparkSession
spark = SparkSession.builder.appName("SpotifyCluster").getOrCreate()

# ==== CARREGAR DADOS ====
file_path = str(Path.cwd().parent / 'data' / 'charts.csv')
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(file_path)
print(df.columns)
df.printSchema()
#%%

# ==== LIMPEZA E FORMATAÇÃO DOS DADOS ====
df = df.withColumn('rank', F.expr("try_cast(rank as double)")) \
       .withColumn('streams', F.expr("try_cast(streams as double)")) \
       .withColumn('region', F.trim(F.col('region'))) \
       .dropna(subset=['rank', 'streams', 'artist', 'region'])
df.printSchema()
#%%

# ==== ESCOLHA ENTRE SAMPLING OU DATASET COMPLETO ====
user_input = input("Proceder para análise com todos os dados ou amostra? (todos/amostra): ").strip().lower()
if user_input == 'amostra':
    df = df.orderBy(F.rand()).limit(5000)
    print("Usando amostra de 5000 linhas para análise.")
else:
    print("Usando todos os dados para análise.")
#%%

# ==== AGREGAÇÃO ====
features = df.groupBy('artist').agg(
    F.mean('streams').alias('artist_streams_mean'),
    F.min('streams').alias('artist_streams_min'),
    F.max('streams').alias('artist_streams_max'),
    F.stddev('streams').alias('artist_streams_std'),
    #F.mean('rank').alias('rank_mean'), -- acho que não faz muito sentido visto que pode ter mais de uma música
    F.min('rank').alias('artist_rank_best'),
    F.countDistinct('region').alias('n_regions'),
    F.count('*').alias('n_appearances')
)
features.show(10)
#%%

# Separando as principais features que irão ser usadas e preenchendo NA com 0 
features = features.fillna(0)

feature_cols = ['artist_streams_mean', 'artist_streams_min', 'artist_streams_max',
                'artist_rank_best','n_appearances', 'n_regions']


# Transformando as features em vetores
# - Fazemos isso porque o PySpark trabalha com vetores e não colunas convencionais

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol='features_raw'
)

# Utilizando o StandardScalar para normalizar os dados

scaler = StandardScaler(
    inputCol='features_raw',
    outputCol='features',
    withStd=True,
    withMean=True
)


# Criando um loop para encontrar o número ideal de Cluster -- FAVOR REVISAR, ESSE [FOR] PODE SER UM TANTO QUANTO CARO
# Esse loop se basea na annálise do gráfico de cotovelo (avalia o custo total da operação)
costs = []
K_range = range(2, 11)

for k in K_range:
    
    kmeans = KMeans(k=k, seed=42, featuresCol='features', predictionCol='prediction')
    pipeline = Pipeline(stages=[assembler, scaler, kmeans])
    model = pipeline.fit(features)
    predictions = model.transform(features)
    kmeans_model = model.stages[-1]
    cost = kmeans_model.summary.trainingCost
    costs.append(cost)
    
    print(f"K={k}, Custo={cost:.2f}")

# Plotar gráfico cotovelo
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(K_range, costs, 'bo-')
plt.xlabel('Número de Clusters')
plt.ylabel('Custo Computacional')
plt.title('Método do Cotovelo')
plt.grid(True)
plt.savefig('../figures/elbow_plot.png')
plt.show()
#%%
# Escolhendo o melhor K com base no resultado do gráfico cotovelo 
# Podemos testar outros K's se necessário
k_optimal = 5

kmeans = KMeans(k=k_optimal, seed=42, featuresCol='features')
pipeline = Pipeline(stages=[assembler, scaler, kmeans])
model = pipeline.fit(features)

# Fazer predições
features_clustered = model.transform(features)
# Iniciando a análise dos clusters com o número de Clusters sendo k_optimal

cluster_stats = features_clustered.groupBy('prediction').agg(
    F.count('*').alias('count'),
    F.mean('artist_streams_mean').alias('avg_streams_mean'),
    F.mean('artist_rank_best').alias('avg_rank_best'),
    F.mean('n_regions').alias('avg_n_regions'),
    F.mean('n_appearances').alias('avg_n_appearances')
).orderBy('prediction')

cluster_stats.show()

# Mostrar top artistas por cluster
for i in range(k_optimal):
    print(f"\nCLUSTER {i}")
    top_artists = features_clustered.filter(F.col('prediction') == i) \
        .groupBy('artist') \
        .count() \
        .orderBy(F.desc('count')) \
        .limit(5)
    
    print("  Top artistas:")
    top_artists.show(truncate=False)

# A conversão para pandas é apenas para facilitar a visualização
# Apenas os dados necessários para não sobrecarregar memória

features_pd = features_clustered.select(
    'artist', 'prediction', *feature_cols
).toPandas()

# Garantir que não há valores nulos que atrapalhem o PCA
features_pd = features_pd.dropna(subset=feature_cols)

# Fazendo PCA

X = features_pd[feature_cols].values
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=features_pd['prediction'], 
                     cmap='viridis', alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('Clusters Visualizados (PCA)')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig('../figures/pca_cluster.png')
plt.show()

'''
Acidicionando os Clusters no dataframe original
Vizualização em 2D dos Cluesters
PC1 (horizontal, 98.7%): Captura quase TODA a diferença entre músicas
 - Esquerda = Músicas com baixo desempenho
 - Direita = Músicas com alto desempenho (muito streams)
PC2 (vertical, 1.1%): Captura pequenas variações adicionais
'''


## Creio que Análise poderia terminar aqui, visto que já temos duas técnicas de ML não supervisionadas aplicadas, Kmeans e PCA.

'''
df_with_clusters = df.join(
    features_clustered.select('title', 'artist', 'prediction'),
    on=['title', 'artist'],
    how='left'
).withColumnRenamed('prediction', 'cluster')

# Análise por região
print("\n=== DISTRIBUIÇÃO POR REGIÃO ===")
region_cluster = df_with_clusters.groupBy('region', 'cluster') \
    .count() \
    .orderBy('region', 'cluster')

region_cluster.show(50)

# Salvando os resultado
df_with_clusters.write.mode('overwrite').parquet('../data/output/spotify_com_clusters.parquet')
print("Arquivo salvo: spotify_com_clusters.parquet")

# salvar apenas as features com clusters
features_clustered.write.mode('overwrite').parquet('../data/output/spotify_features_clusters.parquet')
print("Features salvas: spotify_features_clusters.parquet")
'''
