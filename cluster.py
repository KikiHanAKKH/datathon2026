import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# load cleaned data
X = pd.read_csv("outputs/credit_drop_A2/processed_features.csv")

# test k = 2-6, score each with silhouette (higher = better)
print("testing k values:")
scores = {}
for k in range(2, 7):
    labels = KMeans(n_clusters=k, random_state=67, n_init=10).fit_predict(X)
    scores[k] = silhouette_score(X, labels)
    print(f"  k={k}: silhouette={scores[k]:.3f}")

# pick the k with the best score
best_k = max(scores, key=scores.get)
print(f"best k = {best_k}")

# cluster for real with the best k
model = KMeans(n_clusters=best_k, random_state=67, n_init=10)
clusters = model.fit_predict(X)

# save the cluster assignments
pd.DataFrame({"cluster": clusters}).to_csv(
    "outputs/credit_drop_A2/cluster_labels.csv", index=False)
print(f"done — clustered into {best_k} groups")

# save the model
joblib.dump(model, "outputs/credit_drop_A2/kmeans.pkl")
print("done — model saved")