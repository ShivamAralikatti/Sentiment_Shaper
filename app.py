from flask import Flask, render_template, request
from tensorflow import keras
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import load_model

app = Flask(__name__)

vocab_size = 10000
max_length = 200

# Load the IMDb dataset and preprocess
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = sequence.pad_sequences(x_train, maxlen=max_length)

# Load the pre-trained model
model = load_model("sentiment_model.h5")  # Replace with the actual path to your trained model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    if request.method == 'POST':
        user_input = request.form['review']
        user_sequence = [imdb.get_word_index().get(word.lower(), 2) for word in user_input.split()]
        user_sequence = sequence.pad_sequences([user_sequence], maxlen=max_length)
        prediction = model.predict(user_sequence)[0]
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        return render_template('index.html', prediction=sentiment, review=user_input)

    # Handle GET request (e.g., initial page load)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
