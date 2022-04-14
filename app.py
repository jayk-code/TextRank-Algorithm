from flask import Flask,render_template,url_for,request
import pandas as pd
import joblib
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import re 
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
import ast

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}



stop_words = stopwords.words('english')
data = pd.read_csv(r'C:\Users\jayka\Downloads\top_reviews.csv')
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new




word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        ProductId = request.form['ProductId']
        reviews = list(data[data['ProductId'] == ProductId]['review_list'])
        reviews = reviews[0]
        reviews = ast.literal_eval(reviews) 
        
        clean_reviews = pd.Series(reviews).str.replace("[^a-zA-Z]", " ")
        clean_reviews = [s.lower() for s in reviews]
        clean_reviews = [remove_stopwords(r.split()) for r in clean_reviews]
        
        review_vectors = []
        for i in clean_reviews:
            if len(i) != 0:
                v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
            else:
                v = np.zeros((100,))
            review_vectors.append(v)
        sim_mat = np.zeros([len(reviews), len(reviews)])
        for i in range(len(reviews)):
            for j in range(len(reviews)):
                if i != j:
                    sim_mat[i][j] = cosine_similarity(review_vectors[i].reshape(1,100), review_vectors[j].reshape(1,100))[0,0]

        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        ranked_reviews = sorted(((scores[i],s) for i,s in enumerate(reviews)), reverse=True)
        sn = int(request.form['Number of top reviews'])
        list_of_most_imp_reviews = []
        for i in range(sn):
            list_of_most_imp_reviews.append(str(i+1) + ')' + ' ' + ranked_reviews[i][1])
        
        my_prediction = '\n'.join(list_of_most_imp_reviews)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
