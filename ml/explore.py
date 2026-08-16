# %%
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# %%
BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# %%
data = fetch_california_housing()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["Price"] = data.target

print("Dataset loaded successfully")

# %%
print("Dataset shape:", df.shape)


# %%
df.head()


# %%
df.info()


# %%
print("Missing values:")
print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())


# %%
print("Duplicate rows:", df.duplicated().sum())


# %%
df.describe()  

# %%
# Target variable distribution
#Price distribution → whether the target is skewed and how prices are distributed.

plt.figure(figsize=(10, 6))

sns.histplot(
    df["Price"],
    bins=50,
    kde=True
)

plt.title("House Price Distribution")
plt.xlabel("Price ($100,000)")
plt.ylabel("Number of Houses")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "price_distribution.png", dpi=300)
plt.show()
plt.close()


# %%
# Correlation heatmap
#Correlation heatmap → which features have stronger relationships with price.

plt.figure(figsize=(10, 8))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300)
plt.show()
plt.close()


# %%
# Median Income vs Price
#MedInc vs Price → this is particularly useful because we can visually see whether higher-income areas tend to have higher house prices.

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="MedInc",
    y="Price",
    alpha=0.3
)

plt.title("Median Income vs House Price")
plt.xlabel("Median Income")
plt.ylabel("House Price ($100,000)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "medinc_vs_price.png", dpi=300)
plt.show()
plt.close()


# %%
# House Age vs Price

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="HouseAge",
    y="Price",
    alpha=0.3
)

plt.title("House Age vs House Price")
plt.xlabel("House Age")
plt.ylabel("House Price ($100,000)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "age_vs_price.png", dpi=300)
plt.show()
plt.close()


# %%
# Geographic distribution
#Geographic plot → this makes use of the latitude/longitude features and gives the project a more interesting real-world angle.

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Longitude",
    y="Latitude",
    hue="Price",
    palette="viridis",
    alpha=0.4,
    legend=False
)

plt.title("Geographic Distribution of House Prices")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "geo_dist_price.png", dpi=300)
plt.show()
plt.close()


# %%
# Boxplots for numerical features

features = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude"
]

for feature in features:
    plt.figure(figsize=(8, 4))

    sns.boxplot(x=df[feature])

    plt.title(f"{feature} - Boxplot")
    plt.xlabel(feature)

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / f"{feature.lower()}_boxplot.png",
        dpi=300
    )

    plt.show()
    plt.close()


# %%
# Boxplot for target variable

plt.figure(figsize=(8, 4))

sns.boxplot(x=df["Price"])

plt.title("House Price - Boxplot")
plt.xlabel("Price ($100,000)")

plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "price_boxplot.png",
    dpi=300
)

plt.show()
plt.close()


# %%
# Boxplot for target variable

plt.figure(figsize=(8, 4))

sns.boxplot(x=df["Price"])

plt.title("House Price - Boxplot")
plt.xlabel("Price ($100,000)")
plt.show()

