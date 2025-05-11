# Fake News Detection Using LSTM and Word Embeddings
This project aims to detect fake news articles using a deep learning approach based on LSTM (Long Short-Term Memory) networks. The model uses a custom embedding layer trained on a labeled news dataset to classify articles as real or fake.

# Features
✅ Preprocessing and cleaning of news data

✅ Tokenization and padding of text sequences

✅ Trainable Embedding layer to learn word representations

✅ LSTM-based neural network for sequence classification

✅ Visualizations: word clouds, token distribution, class counts

✅ Model evaluation with accuracy and loss metrics

✅ Easily extendable to use pretrained embeddings (GloVe, FastText)

# Technologies Used
Python

TensorFlow / Keras

NLTK

Pandas, NumPy

Matplotlib, Seaborn, Plotly

Scikit-learn

# Dataset
A labeled dataset of news articles where each article is classified as:

1 = Real

0 = Fake

The dataset is cleaned and tokenized before being fed into the model.

# How to Run
bash
Copy
Edit
# Clone the repo
git clone https://github.com/your-username/fake-news-lstm.git
cd fake-news-lstm

# Install requirements
pip install -r requirements.txt

# Run the training script
python train_model.py
📈 Results
Achieves competitive accuracy and stable training performance on binary fake news classification. Embedding + LSTM architecture shows strong capability in capturing text patterns.

📌 Future Work
Integrate pretrained embeddings (GloVe, FastText)

Add GRU/BiLSTM variants

Deploy as a REST API using FastAPI or Flask
