import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import re
import warnings
warnings.filterwarnings("ignore")


#summerizer lib
# NLP Pkgs
import spacy 
nlp = spacy.load('en_core_web_sm')
#Normalizing Text
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
# Finding the Top N Sentences
from heapq import nlargest

app = Flask(__name__)
# Load the model from the file 
lr_from_joblib = joblib.load('newsclassify.pkl') 


def process_text(text):
    text = text.lower().replace('\n',' ').replace('\r','').strip()
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[^\w\s]','',text)
    
    
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    
    text = " ".join(filtered_sentence)
    return text

#summerizer here
def summerizer(raw_docx):
    raw_text = raw_docx
    docx = nlp(raw_text)
    stopwords = list(STOP_WORDS)


    # Build Word Frequency # word.text is tokenization in spacy
    word_frequencies = {}  
    for word in docx:  
        if word.text not in stopwords:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1
                

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)


    # Sentence Tokens
    sentence_list = [ sentence for sentence in docx.sents ]

    #Calculate Sentence Scores
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if len(sent.text.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]



    # Find N Largest and Join Sentences
    summarized_sentences = nlargest(7, sentence_scores, key=sentence_scores.get)
    final_sentences = [ w.text for w in summarized_sentences ]
    summary = ' '.join(final_sentences)
    print("Original Document\n")
    print(raw_docx)
    print("Total Length:",len(raw_docx))



    print('\n\nSummarized Document\n')
    print(summary)
    print("Total Length:",len(summary))

    return summary

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    message =request.form['message']
    pre_text= process_text(message)

    summery =summerizer(message)


    data = pd.read_csv("BBC News.csv")
    data['Text_parsed'] = data['Text'].apply(process_text)

    #encode data
    from sklearn import preprocessing 
    label_encoder = preprocessing.LabelEncoder() 
    data['Category_target']= label_encoder.fit_transform(data['Category']) 

    #split data
    X_train, X_test, y_train, y_test = train_test_split(data['Text_parsed'], 
                                                    data['Category_target'], 
                                                    test_size=0.2, 
                                                    random_state=8)
    #vectorizer
    ngram_range = (1,2)
    min_df = 10
    max_df = 1.
    max_features = 300
    tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
                        
    features_train = tfidf.fit_transform(X_train).toarray()
    labels_train = y_train
    print(features_train)

    features_test = tfidf.transform(X_test).toarray()
    labels_test = y_test
    print(features_test.shape)

                      
    features_test_ = tfidf.transform([pre_text]).toarray()
    # Use the loaded model to make predictions 
    predict=lr_from_joblib.predict(features_test_) 

    predict = predict[0]
    if predict == 0:
        send ='Business'
    elif predict ==1:
        send ='Entertainment'
    elif predict ==2:
        send ='politics'
    elif predict== 4:
        send ='technology'
    elif predict ==3:
        send ='sports'


    return render_template('index.html', prediction_text=send , summery = summery)
'''
@app.route('/predict_api',methods=['POST'])
def predict_api():
    

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)'''

if __name__ == "__main__":
    app.run(debug=True)
