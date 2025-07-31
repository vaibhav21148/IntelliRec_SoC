
#Frontend using streamlit

import streamlit as st
import requests

st.title("IntelliRec")
st.subheader("Intelligent Recommendation Engine")
st.write("Get top recommendations for a user Id using Deep Learning Model")

user_id = st.number_input("Enter User ID:", min_value=1, step=1)

if st.button("Get Recommendations"):
    with st.spinner("Fetching recommendations..."):
        response = requests.get("http://localhost:5000/recommend", params={'user_id': user_id})

        if response.status_code == 200:
            data = response.json()
            st.success(f"Top 5 Recommendations for User {data['user_id']}:")
            for idx, title in enumerate(data['recommendations'], 1):
                st.write(f"{idx}. {title}")
        else:
            st.error("User ID not found or API error")


st.subheader("Rate The Recommendation Model")
rating = st.slider("Give a star rating (1–5):", 1, 5, step=1)

if st.button("Submit Rating"):
    try:
        res = requests.post(
            "http://localhost:5000/rating",
            json={"rating": rating, "user_id": int(user_id)}
        )
        if res.status_code == 200:
            st.success("Thanks! Your rating has been recorded.")
        else:
            st.error("Failed to save rating.")
            st.write("Status code:", res.status_code)
            st.write("Response:", res.text)
    except Exception as e:
        st.error(f"API error: {e}")
