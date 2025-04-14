import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load your dataset (Update this path if needed)
data = pd.read_csv("Data\crop_recommendation.csv")

# Split data into input (X) and output (y)
X = data.drop(['label'], axis=1)
y = data['label']

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model using current scikit-learn version
with open("crop_recommendation_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully.")
