import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/zomato_cleaned.csv")

print("Dataset Shape:", df.shape)

print("\nRestaurant Data:")
print(
    df[
        [
            "Restaurant Name",
            "City",
            "Cuisines",
            "Price range",
            "Aggregate rating"
        ]
    ].head(10)
)

# User preferences
user_city = "New Delhi"
user_cuisine = "North Indian"
user_price_range = 2
user_min_rating = 4.0

print("\nUser Preferences:")
print("City:", user_city)
print("Cuisine:", user_cuisine)
print("Price Range:", user_price_range)
print("Minimum Rating:", user_min_rating)

# Filter restaurants based on user preferences

filtered_restaurants = df[
    (df["City"].str.lower() == user_city.lower()) &
    (df["Cuisines"].str.lower().str.contains(user_cuisine.lower(), na=False)) &
    (df["Price range"] == user_price_range) &
    (df["Aggregate rating"] >= user_min_rating)
]

print("\nFiltered Restaurants:")
print(
    filtered_restaurants[
        [
            "Restaurant Name",
            "City",
            "Cuisines",
            "Price range",
            "Aggregate rating"
        ]
    ].head(10)
)
print("\nNumber of matching restaurants:", len(filtered_restaurants))

# Rank filtered restaurants by rating

recommended_restaurants = filtered_restaurants.sort_values(
    by="Aggregate rating",
    ascending=False
)

print("\nTop Recommended Restaurants:")

print(
    recommended_restaurants[
        [
            "Restaurant Name",
            "City",
            "Cuisines",
            "Price range",
            "Aggregate rating"
        ]
    ].head(10)
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Create restaurant profile
df["restaurant_profile"] = (
    df["City"].fillna("") + " " +
    df["Cuisines"].fillna("") + " " +
    df["Price range"].astype(str)
)

print("\nRestaurant Profile:")
print(df[["Restaurant Name", "restaurant_profile"]].head())

# Convert restaurant profiles into numerical vectors

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(
    df["restaurant_profile"]
)

print("\nTF-IDF Matrix Shape:")
print(tfidf_matrix.shape)

similarity_matrix = cosine_similarity(tfidf_matrix)

print("\nSimilarity Matrix Shape:")
print(similarity_matrix.shape)
user_profile = (
    user_city + " " +
    user_cuisine + " " +
    str(user_price_range)
)

print("\nUser Profile:")
print(user_profile)

user_vector = vectorizer.transform([user_profile])

user_similarity = cosine_similarity(
    user_vector,
    tfidf_matrix
).flatten()

df["similarity_score"] = user_similarity

recommendations = df[
    (df["City"].str.lower() == user_city.lower()) &
    (df["Cuisines"].str.lower().str.contains(user_cuisine.lower(), na=False)) &
    (df["Price range"] == user_price_range) &
    (df["Aggregate rating"] >= user_min_rating)
]

recommendations = recommendations.sort_values(
    by="similarity_score",
    ascending=False
)

print("\nTop 5 Recommended Restaurants:")

print(
    recommendations[
        [
            "Restaurant Name",
            "City",
            "Cuisines",
            "Price range",
            "Aggregate rating",
            "similarity_score"
        ]
    ].head(5)
)

def recommend_restaurants(city, cuisine, price_range, min_rating=4.0, top_n=5):

    # Filter restaurants
    filtered = df[
        (df["City"].str.lower() == city.lower()) &
        (df["Cuisines"].str.lower().str.contains(cuisine.lower(), na=False)) &
        (df["Price range"] == price_range) &
        (df["Aggregate rating"] >= min_rating)
    ].copy()

    # Create user profile
    user_profile = (
        city + " " +
        cuisine + " " +
        str(price_range)
    )

    user_vector = vectorizer.transform([user_profile])
    similarity = cosine_similarity(
        user_vector,
        tfidf_matrix
    ).flatten()

    filtered["similarity_score"] = similarity[filtered.index]
    filtered = filtered.sort_values(
        by="similarity_score",
        ascending=False
    )
    return filtered[
        [
            "Restaurant Name",
            "City",
            "Cuisines",
            "Price range",
            "Aggregate rating",
            "similarity_score"
        ]
    ].head(top_n)

# Test recommendation function

result = recommend_restaurants(
    city="New Delhi",
    cuisine="North Indian",
    price_range=2,
    min_rating=4.0,
    top_n=5
)

print("\nFinal Restaurant Recommendations:")
print(result)

result1 = recommend_restaurants(
    city="New Delhi",
    cuisine="North Indian",
    price_range=2,
    min_rating=4.0,
    top_n=5
)

print("\n========== TEST 1 ==========")
print(result1)
result2 = recommend_restaurants(
    city="New Delhi",
    cuisine="Chinese",
    price_range=2,
    min_rating=3.5,
    top_n=5
)

print("\n========== TEST 2 ==========")
print(result2)
result3 = recommend_restaurants(
    city="Noida",
    cuisine="Fast Food",
    price_range=1,
    min_rating=3.5,
    top_n=5
)

print("\n========== TEST 3 ==========")
print(result3)

# Calculate similarity between restaurants

similarity_matrix = cosine_similarity(tfidf_matrix)

print("\nSimilarity Matrix Shape:")
print(similarity_matrix.shape)
import joblib
import os

# Create models folder
os.makedirs("models", exist_ok=True)

# Save TF-IDF vectorizer
joblib.dump(
    vectorizer,
    "models/recommendation_vectorizer.pkl"
)

print("\nRecommendation vectorizer saved successfully!")
print("File: models/recommendation_vectorizer.pkl")
# Save restaurant data for recommendation system

recommendation_data = df[
    [
        "Restaurant Name",
        "City",
        "Cuisines",
        "Price range",
        "Aggregate rating"
    ]
].copy()

recommendation_data.to_csv(
    "data/recommendation_data.csv",
    index=False
)

print("Recommendation data saved successfully!")
print("File: data/recommendation_data.csv")
