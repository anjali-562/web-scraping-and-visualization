<<<<<<< Updated upstream
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set theme for better visuals
sns.set_theme(style="whitegrid")

# Step 1: Load the dataset
df = pd.read_csv("Samples/Companies.csv")
print("✅ Data Loaded Successfully")
print(df.head())

# Step 2: Clean numeric columns
df["Revenue (Billions USD)"] = (
    df["Revenue (Billions USD)"].astype(str).str.replace(",", "").astype(float)
)
df["Employees"] = (
    df["Employees"].astype(str).str.replace(",", "").replace("N/A", "0").astype(int)
)
df["Market Cap (B)"] = (
    df["Market Cap (B)"].astype(str).str.replace(",", "").replace("N/A", "0").astype(float)
)

# =============================
# 📊 Visualization Section
# =============================

# 1. Top 10 Companies by Revenue
plt.figure(figsize=(12,6))
top10 = df.nlargest(10, "Revenue (Billions USD)")
sns.barplot(x="Company Name", y="Revenue (Billions USD)", data=top10, palette="viridis")
plt.xticks(rotation=45)
plt.title("Top 10 Companies by Revenue", fontsize=16, weight="bold")
plt.tight_layout()
plt.savefig("Samples/top10_revenue.png")
plt.show()

# 2. Revenue vs Market Cap Scatterplot
plt.figure(figsize=(10,6))
sns.scatterplot(
    x="Revenue (Billions USD)", 
    y="Market Cap (B)", 
    hue="Industry", 
    size="Employees",
    sizes=(50,500),
    data=df, alpha=0.7, palette="tab10"
)
plt.title("Revenue vs Market Cap (Colored by Industry)", fontsize=16, weight="bold")
plt.tight_layout()
plt.savefig("Samples/revenue_vs_marketcap.png")
plt.show()

# 3. Distribution of Employees
plt.figure(figsize=(10,6))
sns.histplot(df["Employees"], bins=30, kde=True, color="blue")
plt.title("Distribution of Employees Across Companies", fontsize=16, weight="bold")
plt.xlabel("Number of Employees")
plt.tight_layout()
plt.savefig("Samples/employee_distribution.png")
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(8,6))
corr = df[["Revenue (Billions USD)", "Employees", "Market Cap (B)"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Financial Metrics", fontsize=16, weight="bold")
plt.tight_layout()
plt.savefig("Samples/correlation_heatmap.png")
plt.show()

# 5. Country-wise Total Revenue
plt.figure(figsize=(12,6))
country_revenue = df.groupby("Country")["Revenue (Billions USD)"].sum().reset_index()
top_countries = country_revenue.sort_values(by="Revenue (Billions USD)", ascending=False).head(10)
sns.barplot(
    x="Revenue (Billions USD)", 
    y="Country", 
    data=top_countries, 
    palette="magma"
)
plt.title("Top 10 Countries by Total Revenue", fontsize=16, weight="bold")
plt.tight_layout()
plt.savefig("Samples/country_revenue.png")
plt.show()

=======
# visualize.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Ensure plots look clean
sns.set(style="whitegrid")

# Step 1: Load the scraped data
df = pd.read_csv("Samples/Companies.csv")
print("✅ Data Loaded")
print(df.head())

# Convert numeric columns properly (remove commas, convert to float/int)
df["Revenue (Billions USD)"] = df["Revenue (Billions USD)"].astype(float)
df["Employees"] = df["Employees"].str.replace(",", "").astype(int)
df["Market Cap (B)"] = df["Market Cap (B)"].astype(float)

# Step 2: Visualize raw data -> Top 10 companies by Revenue
plt.figure(figsize=(10,6))
sns.barplot(x="Company Name", y="Revenue (Billions USD)", data=df.head(10))
plt.xticks(rotation=45)
plt.title("Top 10 Companies by Revenue")

# Save bar chart
plt.savefig("Samples/top10_revenue.png", dpi=300, bbox_inches='tight')
plt.show()

# Step 3: Prepare data for ML (treat revenue like a sequence)
data = df["Revenue (Billions USD)"].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Function to create sequences for LSTM
def create_sequences(data, seq_length=3):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, seq_length=3)

# Step 4: Build LSTM model
model = Sequential([
    LSTM(50, activation="relu", input_shape=(X.shape[1], X.shape[2])),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=50, verbose=1)

# Step 5: Make predictions
predictions = model.predict(X)
predicted_revenue = scaler.inverse_transform(predictions)

# Step 6: Visualize predicted vs actual revenue
plt.figure(figsize=(10,6))
sns.lineplot(x=range(len(df)), y=df["Revenue (Billions USD)"], label="Actual Revenue")
sns.lineplot(x=range(3, len(predicted_revenue)+3), y=predicted_revenue.flatten(), label="Predicted Revenue")
plt.title("Actual vs Predicted Revenue (LSTM)")
plt.xlabel("Company Index")
plt.ylabel("Revenue (Billions USD)")
plt.legend()

# Save LSTM chart
plt.savefig("Samples/revenue_predictions.png", dpi=300, bbox_inches='tight')
plt.show()

# Step 7: Forecast future revenue for upcoming years
def forecast_future(model, data, seq_length=3, future_steps=5):
    """
    model: trained LSTM model
    data: scaled revenue data
    seq_length: how many past values used to predict next
    future_steps: how many future years to predict
    """
    predictions = []
    current_seq = data[-seq_length:]  # last known sequence
    
    for _ in range(future_steps):
        pred = model.predict(current_seq.reshape(1, seq_length, 1))
        predictions.append(pred[0, 0])
        # update sequence by appending new prediction and removing first element
        current_seq = np.append(current_seq[1:], pred).reshape(seq_length, 1)
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Predict next 5 years of revenue
future_years = 5
future_revenue = forecast_future(model, scaled_data, seq_length=3, future_steps=future_years)

# Step 8: Visualize future predictions
plt.figure(figsize=(10,6))
# Plot actual revenue
sns.lineplot(x=range(len(df)), y=df["Revenue (Billions USD)"], label="Actual Revenue")
# Plot model predictions on training data
sns.lineplot(x=range(3, len(predicted_revenue)+3), y=predicted_revenue.flatten(), label="Predicted (Training)")
# Plot future forecast
sns.lineplot(x=range(len(df), len(df)+future_years), y=future_revenue.flatten(), label="Future Forecast", linestyle="--", marker="o")

plt.title("Revenue Forecast for Upcoming Years (LSTM)")
plt.xlabel("Company Index / Year progression")
plt.ylabel("Revenue (Billions USD)")
plt.legend()

# Save forecast chart
plt.savefig("Samples/future_revenue_forecast.png", dpi=300, bbox_inches='tight')
plt.show()

print("📈 Future revenue forecast saved at: Samples/future_revenue_forecast.png")
>>>>>>> Stashed changes


