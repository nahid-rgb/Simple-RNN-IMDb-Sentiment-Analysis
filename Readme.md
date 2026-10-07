# 🎬 IMDb Movie Review Sentiment Analysis

A deep learning project that uses a **Simple RNN** to classify IMDb movie reviews as **Positive** or **Negative**.

## 🚀 Project Overview

The model is trained on the **IMDb Movie Review Dataset**, provided through the TensorFlow/Keras dataset API. It contains **50,000 movie reviews** labeled as positive or negative.

The project uses an **Embedding layer** followed by a **Simple RNN** for sentiment classification. A **Streamlit web application** is included where users can enter a movie review and get a sentiment prediction.

## 📊 Dataset

The IMDb dataset is loaded directly through TensorFlow/Keras:

```python
from tensorflow.keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=10000
)
```

* 25,000 training reviews
* 25,000 testing reviews
* Binary sentiment labels: `0 = Negative`, `1 = Positive`
* Vocabulary limited to the 10,000 most frequent words

**Dataset Source:** TensorFlow/Keras IMDb Dataset

## 🧠 Model Architecture

```text
Input Review
     ↓
Integer Encoding
     ↓
Padding (500)
     ↓
Embedding (128)
     ↓
Simple RNN (128)
     ↓
Dense (1, Sigmoid)
     ↓
Positive / Negative
```

## 📊 Model Performance

The model achieved **82.7% accuracy** on the IMDb test dataset.

## 🌐 Streamlit App

Try the deployed Streamlit application:

**[IMDb Movie Review Sentiment Analysis](https://simple-rnn-imdb-sentiment-analysis-gkhbkelibf6gjvwwppxzae.streamlit.app/)**

The Streamlit application allows users to:

* Enter a movie review
* Get a Positive or Negative sentiment prediction
* View the prediction score

## 📁 Project Structure

```text
├── main.py
├── simple_rnn_imdb.keras
├── SimpleRNN.ipynb
├── prediction.ipynb
├── embeddings.ipynb
├── requirements.txt
├── runtime.txt
└── .gitignore
```

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Simple RNN
* Embedding Layer
* Streamlit
* Jupyter Notebook

## ▶️ Run Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```cmd
streamlit run main.py
```

The application will open in your browser.

## ⚠️ Note

The model performs better on longer movie reviews. Very short custom reviews may sometimes produce unreliable predictions.
