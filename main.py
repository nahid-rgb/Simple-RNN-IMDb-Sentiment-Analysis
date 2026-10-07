# Import required libraries
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# Load the IMDb dataset word index
word_index = imdb.get_word_index()

# Load the trained sentiment analysis model
model = load_model('simple_rnn_imdb.keras')


# Convert a user's text into padded integer IDs that can be given to the model
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],padding='pre',maxlen=500)
    return padded_review


# Predict whether a given review is Positive or Negative
def predict_sentiment(review):
    padded_review = preprocess_text(review)
    prediction_score = model.predict(padded_review, verbose=0)
    sentiment = 'Positive' if prediction_score[0][0] > 0.5 else 'Negative'
    return sentiment, prediction_score[0][0]


# Streamlit app
st.title("🎬 IMDb Movie Review Sentiment Analysis")
st.write("Enter a movie review below and the model will predict whether it is positive or negative.")

# Get movie review from the user
review_text = st.text_area(
    "Enter your movie review:",
    placeholder="Example: The movie was amazing and I really enjoyed it!",
    height=150
)

# Make prediction when the user clicks the button
if st.button("🔍 Predict Sentiment"):

    if review_text.strip():

        sentiment, prediction_score = predict_sentiment(review_text)

        st.subheader("Prediction")

        if sentiment == "Positive":
            st.success(f"😊 Sentiment: {sentiment}")
        else:
            st.error(f"😞 Sentiment: {sentiment}")

        st.write(f"Prediction Score: {prediction_score:.10f}")

    else:
        st.warning("⚠️ Please enter a movie review first.")