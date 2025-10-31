import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Step 1: Load dataset
df = pd.read_csv("crop_revenue.csv")

# Step 2: Revenue column create kar le (Production * Yield se)
df['Revenue'] = df['Production'] * df['Yield']

# Step 3: Encode categorical columns
le_crop = LabelEncoder()
le_state = LabelEncoder()
le_season = LabelEncoder()

df['Crop'] = le_crop.fit_transform(df['Crop'])
df['State'] = le_state.fit_transform(df['State'])
df['Season'] = le_season.fit_transform(df['Season'])

# Step 4: Features aur target split
X = df[['Crop', 'State', 'Season', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
y = df['Revenue']

# Step 5: Split training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Save model aur encoders
pickle.dump(model, open("crop_revenue_model.pkl", "wb"))
pickle.dump(le_crop, open("le_crop.pkl", "wb"))
pickle.dump(le_state, open("le_state.pkl", "wb"))
pickle.dump(le_season, open("le_season.pkl", "wb"))

print("✅ Model training completed and saved successfully!")
