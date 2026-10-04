import pandas as pd

# Load raw dataset
df = pd.read_csv("data/zomato.csv", encoding="latin1")

print("Original Dataset Shape:", df.shape)


df["Cuisines"] = df["Cuisines"].fillna("Unknown")

df = df.drop_duplicates()


df.columns = df.columns.str.strip()


binary_columns = [
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now",
    "Switch to order menu"
]

for col in binary_columns:
    df[col] = df[col].map({
        "Yes": 1,
        "No": 0
    })

df.to_csv("data/zomato_cleaned.csv", index=False)

print("Cleaned Dataset Shape:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nCleaned dataset saved successfully!")
