from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
import string
from nltk.corpus import stopwords

# Initialize the Flask application
app = Flask(__name__)

# Load and preprocess the data
data = pd.read_csv("netflixData.csv")
data = data[["Title", "Description", "Content Type", "Genres"]]
data = data.dropna().reset_index(drop=True)

nltk.download('stopwords')
stopword = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in stopword]
    text = " ".join(text)
    return text

data["Title"] = data["Title"].apply(clean)
data["Description"] = data["Description"].apply(clean)
data["Genres"] = data["Genres"].apply(clean)
data['combined'] = data['Title'] + ' ' + data['Description'] + ' ' + data['Genres']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['combined'])
similarity = cosine_similarity(tfidf_matrix)
indices = pd.Series(data.index, index=data['Title']).drop_duplicates()

def netflix_recommendation(title, similarity=similarity, indices=indices, data=data):
    if title not in indices:
        return f"{title} is not in the Database"
    
    index = indices[title]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:11]
    movie_indices = [i[0] for i in similarity_scores]
    return data['Title'].iloc[movie_indices].tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.form['title']
    recommendations = netflix_recommendation(title)
    if isinstance(recommendations, str):
        return render_template('index.html', recommendations=[recommendations])
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
