import pandas as pd
import warnings
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt

df = pd.read_csv("Churn_Modelling.csv")

#either drop all columns that are not numerical or change them to numerical
#also drop irrelivant columns
df = df.set_index('CustomerId')
df = df.drop('Surname', axis=1)
df = df.drop('RowNumber', axis=1)
df = df.drop("Exited", axis=1)
df['Gender'] = df['Gender'].replace({"Male": 0, "Female": 1})
df['Geography'] = df['Geography'].replace({"France": 0, "Spain": 1, "Germany": 2})

print(df.head(10))

#using the elbow method to determine the number of k klusters to use
warnings.filterwarnings('ignore')
interia = []
for test_k in range(2, 50, 2):
    print(test_k)
    
    tmp_model = KMeans(n_clusters=test_k)
    tmp_model.fit(df)
    
    score = tmp_model.inertia_
    interia.append((test_k, score))

intertia_df = pd.DataFrame(interia, columns=["k", "score"])
print(intertia_df)

fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(1,1,1)

intertia_df.plot("k", "score", ax=ax)
ax.set_ylabel("Intertia")
plt.show()

#the graph tells us we should use k=10
k = 10

cluster_model = KMeans(n_clusters=k, random_state=12)
cluster_model.fit(df)
clusters = cluster_model.predict(df)
clusters_column = pd.DataFrame(clusters, index=df.index, columns=["Cluster"])
clusters_df = clusters_column.join(df)
print(clusters_df.head(20))

num_per_cluster = pd.DataFrame(clusters_df['Cluster'].value_counts())
print(num_per_cluster)