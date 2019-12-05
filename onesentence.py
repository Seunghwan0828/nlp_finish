from flask import Flask, request, jsonify
from flask_cors import CORS
from krwordrank.sentence import summarize_with_sentences
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from konlpy.tag import Okt
okt = Okt()

app = Flask(__name__)
CORS(app)

from keras.models import model_from_json
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

global model
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")
global graph
graph = tf.get_default_graph()
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])

tokenizer = okt

data = pd.read_csv('datahap_2.csv',encoding='cp949')
data.label.value_counts()
labels = to_categorical(data['label'], num_classes=5)
stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다',',','대숲',',,','하이','대학','안녕','익명','글쓴이','쓰니','오늘','나','너','저','누나','오빠']
X_train=[]

max_len=30
max_words = 5000

for sentence in data['text']:
    temp_X = []
    temp_X=okt.morphs(sentence, stem=True) # 토큰화
    temp_X=[word for word in temp_X if not word in stopwords] # 불용어 제거
    X_train.append(temp_X)

tokenizer = Tokenizer(num_words=max_words,lower=False)
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)
X = pad_sequences(sequences, maxlen=max_len)
X_train, X_test, y_train, y_test = train_test_split(X , labels, test_size=0.20)

accr = model.evaluate(X_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}%'.format(accr[0],accr[1]*100))


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/summary', methods=['POST'])
def summary():
    text = request.get_json()  # json 데이터를 받아옴
    json_data = text
    data = json_data["text"]
    emotion = json_data["emotion"]
    data_list = []
    for sentence in data:
        list_sentence1 = sentence.split('\n')
        for list_sentence2 in list_sentence1:
            list_sentence = list_sentence2.replace('. ', '.   ...').replace('? ', '?   ...').replace('! ','!   ...').split('  ...')
            for lines in list_sentence:
                line = lines.strip()
                data_list.append(line)
    data_list1 = list(data_list)
    for i in range(len(data_list)):
        x = data_list1.count('')
        for j in range(x):
            data_list1.remove('')
    texts = data_list1
    penalty = lambda x: 0 if (20 <= len(x) <= 120) else 1
    stopwords = {'오늘', '오늘은'}
    keywords, sents = summarize_with_sentences(
        texts,
        penalty=penalty,
        stopwords=stopwords,
        diversity=0.5,
        num_keywords=7,
        num_keysents=3,
        scaling=lambda x: 1,
        verbose=False,
        min_count=1)
    before_sentiment = []
    sentiment = []
    keyword = []
    for sent in sents:
        before_sentiment.append(sent)
    keywords = list(keywords.keys())
    for l in keywords:
        k = okt.nouns(l)
        if len(k) > 0:
            for n in k:
                if len(n) > 1:
                    keyword.append(n)
    print(before_sentiment)
    print(keywords)
    print(keyword)

    def text_input(a):
        global graph
        with graph.as_default():
            txt = []
            txt.append(a)
            text = []
            for sentence in txt:
                temp_X = []
                temp_X = okt.morphs(sentence, stem=True)  # 토큰화
                temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제거
                text.append(temp_X)
            seq = tokenizer.texts_to_sequences(text)
            padded = pad_sequences(seq, maxlen=max_len)
            pred = model.predict(padded)
            labels = [0,1,2,3,4]
        return labels[np.argmax(pred)]

    for i in range(3):
        sentiment.append(text_input(a=before_sentiment[i]))
    print(sentiment)

    def find_nearest(array, value):
        n = [abs(i - value) for i in array]
        idx = n.index(min(n))
        return idx

    a = find_nearest(sentiment, emotion)
    sentiment_sent = before_sentiment[a]

    return jsonify({"onesentence": sentiment_sent,
                    "keyword": keyword})  # 받아온 데이터를 다시 전송


if __name__ == "__main__":
    app.run(host="0.0.0.0",port="8080")
