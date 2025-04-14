# train_crop_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Step 1: Load Dataset
data = pd.read_csv('Data/Crop_recommendation.csv')  # Adjust the path if needed

# Step 2: Features and Labels
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Save Model using current version of scikit-learn
with open('models/RandomForest.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved using current scikit-learn version.")
