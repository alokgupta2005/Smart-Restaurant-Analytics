import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("data/zomato_cleaned.csv")

print("Dataset Shape:", df.shape)

os.makedirs("reports/figures", exist_ok=True)

plt.figure(figsize=(10, 6))

sns.histplot(
    df["Aggregate rating"],
    bins=20,
    kde=True
)

plt.title("Restaurant Rating Distribution")
plt.xlabel("Aggregate Rating")
plt.ylabel("Number of Restaurants")

plt.tight_layout()

plt.savefig(
    "reports/figures/rating_distribution.png",
    dpi=300
)

plt.close()

print("Rating distribution graph saved successfully!")

# Top 10 Cuisines
top_cuisines = (
    df["Cuisines"]
    .value_counts()
    .head(10)
)

print("\nTop 10 Cuisines:")
print(top_cuisines)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_cuisines.values,
    y=top_cuisines.index
)

plt.title("Top 10 Most Popular Cuisines")
plt.xlabel("Number of Restaurants")
plt.ylabel("Cuisine")

plt.tight_layout()

plt.savefig(
    "reports/figures/top_10_cuisines.png",
    dpi=300
)

plt.close()

print("Top 10 cuisines graph saved successfully!")

# Top 10 Cities
top_cities = (
    df["City"]
    .value_counts()
    .head(10)
)

print("\nTop 10 Cities:")
print(top_cities)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_cities.values,
    y=top_cities.index
)

plt.title("Top 10 Cities by Number of Restaurants")
plt.xlabel("Number of Restaurants")
plt.ylabel("City")

plt.tight_layout()

plt.savefig(
    "reports/figures/top_10_cities.png",
    dpi=300
)

plt.close()

print("Top 10 cities graph saved successfully!")

# Price Range Distribution
price_range = df["Price range"].value_counts().sort_index()

print("\nPrice Range Distribution:")
print(price_range)

plt.figure(figsize=(8, 5))

sns.barplot(
    x=price_range.index,
    y=price_range.values
)

plt.title("Restaurant Price Range Distribution")
plt.xlabel("Price Range")
plt.ylabel("Number of Restaurants")

plt.tight_layout()

plt.savefig(
    "reports/figures/price_range_distribution.png",
    dpi=300
)

plt.close()

print("Price range graph saved successfully!")

# Votes vs Aggregate Rating
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Votes",
    y="Aggregate rating",
    alpha=0.5
)

plt.title("Votes vs Aggregate Rating")
plt.xlabel("Number of Votes")
plt.ylabel("Aggregate Rating")

plt.tight_layout()

plt.savefig(
    "reports/figures/votes_vs_rating.png",
    dpi=300
)

plt.close()

print("Votes vs Rating graph saved successfully!")
