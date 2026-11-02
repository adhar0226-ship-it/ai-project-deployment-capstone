import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def train_and_save():
    # Synthetic data matching retail profiling
    np_data = [
        [15, 39], [15, 81], [16, 6], [16, 77], [17, 40],
        [60, 50], [62, 53], [63, 51], [65, 55], [70, 58],
        [85, 90], [87, 95], [88, 91], [90, 92], [95, 98]
    ]
    df = pd.DataFrame(np_data, columns=['Annual_Income_k', 'Spending_Score'])
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    model.fit(scaled)
    
    joblib.dump(model, 'models/kmeans_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Model and Scaler serialized successfully in models/")

if __name__ == '__main__':
    train_and_save()
