import streamlit as st
import pandas as pd


# Page Configuration
st.set_page_config(
    page_title="Smart Restaurant Analytics",
    page_icon="🍽️",
    layout="wide"
)


# Load Dataset
df = pd.read_csv("data/zomato_cleaned.csv")


# Title
st.title("🍽️ Smart Restaurant Analytics")
st.write("Smart Restaurant Analytics & Recommendation System")


# Sidebar Navigation
st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Choose a module:",
    [
        "Dashboard",
        "Explore Restaurants"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.header("📊 Business Insights Dashboard")

    st.write(
        "Explore restaurant trends, ratings, pricing, cuisines and delivery insights."
    )

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🍽️ Total Restaurants",
        len(df)
    )

    col2.metric(
        "🏙️ Total Cities",
        df["City"].nunique()
    )

    col3.metric(
        "⭐ Average Rating",
        round(df["Aggregate rating"].mean(), 2)
    )

    col4.metric(
        "👍 Total Votes",
        f"{df['Votes'].sum():,}"
    )

    st.divider()

    # Second Row Metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🥘 Total Cuisines",
        df["Cuisines"].nunique()
    )

    col2.metric(
        "💰 Avg Cost for Two",
        f"₹{df['Average Cost for two'].mean():.0f}"
    )

    online_delivery_count = (
        df["Has Online delivery"]
        .astype(str)
        .str.lower()
        .isin(["yes", "1"])
        .sum()
    )

    col3.metric(
        "🚚 Online Delivery",
        online_delivery_count
    )

    table_booking_count = (
        df["Has Table booking"]
        .astype(str)
        .str.lower()
        .isin(["yes", "1"])
        .sum()
    )

    col4.metric(
        "🪑 Table Booking",
        table_booking_count
    )

    st.divider()

    # Top Cities and Rating Distribution
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🏙️ Top 10 Cities")

        top_cities = (
            df["City"]
            .value_counts()
            .head(10)
        )

        st.bar_chart(top_cities)

    with col2:

        st.subheader("⭐ Rating Distribution")

        rating_data = (
            df["Aggregate rating"]
            .value_counts()
            .sort_index()
        )

        st.line_chart(rating_data)

    st.divider()

    # Top Cuisines
    st.subheader("🥘 Top 10 Popular Cuisines")

    cuisine_data = (
        df["Cuisines"]
        .value_counts()
        .head(10)
    )

    st.bar_chart(cuisine_data)

    st.divider()

    # Price Range
    st.subheader("💰 Restaurants by Price Range")

    price_data = (
        df["Price range"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(price_data)

    st.divider()

    # Delivery and Table Booking
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🚚 Online Delivery")

        delivery_data = (
            df["Has Online delivery"]
            .value_counts()
        )

        st.bar_chart(delivery_data)

    with col2:

        st.subheader("🪑 Table Booking")

        booking_data = (
            df["Has Table booking"]
            .value_counts()
        )

        st.bar_chart(booking_data)

    st.divider()

    # Business Insights
    st.subheader("💡 Business Insights")

    highest_rated = df.loc[
        df["Aggregate rating"].idxmax(),
        "Restaurant Name"
    ]

    most_voted = df.loc[
        df["Votes"].idxmax(),
        "Restaurant Name"
    ]

    top_city = (
        df["City"]
        .value_counts()
        .idxmax()
    )

    top_cuisine = (
        df["Cuisines"]
        .value_counts()
        .idxmax()
    )

    highest_rating = df["Aggregate rating"].max()

    average_cost = df["Average Cost for two"].mean()

    st.info(
        f"""
📍 **Most restaurants are located in:** {top_city}

🥘 **Most common cuisine:** {top_cuisine}

⭐ **Highest rating in dataset:** {highest_rating}

🏆 **Highest-rated restaurant:** {highest_rated}

👍 **Most voted restaurant:** {most_voted}

💰 **Average cost for two:** ₹{average_cost:.0f}
"""
    )


# =========================================================
# EXPLORE RESTAURANTS
# =========================================================

elif page == "Explore Restaurants":

    st.header("🔍 Explore Restaurants")

    st.write(
        "Explore restaurants by city, rating and price range."
    )

    # City Filter
    cities = sorted(
        df["City"]
        .dropna()
        .unique()
    )

    city = st.selectbox(
        "🏙️ Select City",
        ["All Cities"] + cities
    )

    if city == "All Cities":

        filtered_data = df.copy()

    else:

        filtered_data = df[
            df["City"]
            .astype(str)
            .str.lower()
            == city.lower()
        ].copy()

    # Rating Filter
    min_rating = st.slider(
        "⭐ Minimum Rating",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.5
    )

    filtered_data = filtered_data[
        filtered_data["Aggregate rating"] >= min_rating
    ]

    # Price Filter
    price_options = sorted(
        filtered_data["Price range"]
        .dropna()
        .unique()
    )

    price = st.selectbox(
        "💰 Price Range",
        ["All"] + list(price_options)
    )

    if price != "All":

        filtered_data = filtered_data[
            filtered_data["Price range"] == price
        ]

    st.divider()

    # Restaurant Count
    st.subheader("📊 Restaurant Count")

    col1, col2 = st.columns(2)

    col1.metric(
        "🍽️ Available Restaurants",
        len(filtered_data)
    )

    col2.metric(
        "⭐ Average Rating",
        round(
            filtered_data["Aggregate rating"].mean(),
            2
        ) if len(filtered_data) > 0 else 0
    )

    st.divider()

    # Number of Restaurants to Display
    st.subheader("🔢 Select Number of Restaurants")

    number_of_restaurants = st.slider(
        "🍽️ Number of Restaurants to Show",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    # Restaurant Details
    st.subheader("🍽️ Restaurant Details")

    display_columns = [
        "Restaurant Name",
        "City",
        "Locality",
        "Address",
        "Cuisines",
        "Price range",
        "Aggregate rating",
        "Votes"
    ]

    restaurants_to_show = (
        filtered_data
        .sort_values(
            by="Aggregate rating",
            ascending=False
        )
        .head(number_of_restaurants)
    )

    st.info(
        f"🍽️ Showing top {len(restaurants_to_show)} restaurants."
    )

    st.dataframe(
        restaurants_to_show[display_columns],
        use_container_width=True
    )

    st.divider()

    # Restaurant Location Map
    st.subheader("🗺️ Restaurant Locations")

    map_data = restaurants_to_show[
        ["Latitude", "Longitude"]
    ].copy()

    map_data["latitude"] = pd.to_numeric(
        map_data["Latitude"],
        errors="coerce"
    )

    map_data["longitude"] = pd.to_numeric(
        map_data["Longitude"],
        errors="coerce"
    )

    map_data = map_data[
        ["latitude", "longitude"]
    ].dropna()

    if len(map_data) > 0:

        st.map(map_data)

        st.caption(
            "📍 Map shows the locations of the selected restaurants from the Zomato dataset."
        )

    else:

        st.warning(
            "⚠️ Location coordinates are not available for the selected restaurants."
        )
