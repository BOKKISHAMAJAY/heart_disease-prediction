import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("heart_cleveland_upload.csv")

# Rename target
df = df.rename(columns={'condition': 'target'})

# Split data
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train_scaled, y_train)

# Save BOTH files
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("Files created: model.pkl and scaler.pkl")