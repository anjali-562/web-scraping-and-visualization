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



