import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("tartu.csv")

data['start_time'] = pd.to_datetime(data['start_time'])
data['finish_time'] = pd.to_datetime(data['finish_time'])

data['weekend'] = data['start_time'].dt.dayofweek >= 5
data['hour'] = data['start_time'].dt.hour

coords = data[['vehicle_start_lat', 'vehicle_start_lng']]
kmeans = KMeans(n_clusters=10, random_state=42)
data['cluster'] = kmeans.fit_predict(coords)

features = ['weekend', 'hour', 'cluster']
target = 'ride_value'
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae}")
print(f"R2: {r2}")

new_data = []
for hour in range(24):
    for cluster in range(10):
        for weekend in [0, 1]:
            new_data.append([weekend, hour, cluster])
new_data = pd.DataFrame(new_data, columns=features)

new_data['predicted_ride_value'] = model.predict(new_data)

centroids = kmeans.cluster_centers_
centroids_df = pd.DataFrame(centroids, columns=['centroid_lat', 'centroid_lng'])
new_data[['centroid_lat', 'centroid_lng']] = new_data['cluster'].apply(lambda x: centroids_df.iloc[x]).reset_index(drop=True)

optimal_points = new_data.sort_values(by='predicted_ride_value', ascending=False).drop_duplicates(subset=['centroid_lat', 'centroid_lng'])

optimal_points.to_csv("Optimal_Predicted_Points.csv", index=False)

plt.figure(figsize=(12, 8))
scatterplot = sns.scatterplot(x='centroid_lng', y='centroid_lat', hue='predicted_ride_value', size='predicted_ride_value', sizes=(20, 200), data=optimal_points, palette='viridis')
plt.title('Predicted Optimal Points for Transport Placement')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

for index, row in optimal_points.iterrows():
    plt.annotate(f"{row['centroid_lat']:.4f}, {row['centroid_lng']:.4f}\n€{row['predicted_ride_value']:.2f}",
                 (row['centroid_lng'], row['centroid_lat']),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha='center')

plt.legend()
plt.tight_layout()

plt.savefig('Optimal_Predicted_Points.png', dpi=300)

plt.show()

print(optimal_points)
