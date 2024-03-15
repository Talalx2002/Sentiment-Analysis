import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("imdb_sentiment_model.h5")

# Function to predict sentiment based on user input
def predict_sentiment(review):
    # Tokenize the review
    review = [review]
    # Make the prediction
    prediction = model.predict(review)
    # Determine sentiment based on prediction
    if prediction[0] >= 0.5:
        return "Positive"
    else:
        return "Negative"

# Prompt the user for input
review = input("Enter your review: ")

# Predict sentiment
sentiment = predict_sentiment(review)
print("Predicted sentiment:", sentiment)
