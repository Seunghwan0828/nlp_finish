import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from konlpy.tag import Okt
okt = Okt()

from keras.models import model_from_json
json_file = open("model_1.json", "r")
loaded_model_json = json_file.read()
json_file.close()

global model
model = model_from_json(loaded_model_json)
model.load_weights("model_1.h5")
print("Loaded model from disk")
global graph
graph = tf.get_default_graph()
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])

tokenizer = okt

data = pd.read_csv('datahap_1.csv',encoding='cp949')
data.label.value_counts()
labels = to_categorical(data['label'], num_classes=5)
stopwords=['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다',
           ',', '대숲', ',,', '하이', '대학', '안녕', '익명', '글쓴이', '쓰니', '오늘', '나', '너', '저', '누나', '오빠',
           '기분', '정말', '매우', '몹시', '너무', '엄청', '아주', '훨씬', '가장', '최고', '더', '덜', '맣이', '조금']
X_train=[]

max_len=30
max_words = 8000

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

def text_input(a):
    for i in range(a):
        input_txt = input("텍스트를 입력해 주세요: ")
        txt = []
        txt.append(input_txt)
        text = []
        for sentence in txt:
            temp_X = []
            temp_X=okt.morphs(sentence, stem=True) # 토큰화
            temp_X=[word for word in temp_X if not word in stopwords] # 불용어 제거
            text.append(temp_X)
            seq = tokenizer.texts_to_sequences(text)
            padded = pad_sequences(seq, maxlen=max_len)
            pred = model.predict(padded)
            labels = ['매우나쁨','나쁨','중간','좋음','매우좋음']
            print(labels[np.argmax(pred)])
a = int(input("몇번 입력 하실건가요?: "))
text_input(a)