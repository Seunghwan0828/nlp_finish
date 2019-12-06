import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import model_from_yaml
from konlpy.tag import Okt
okt = Okt()
data = pd.read_csv('datahap_1.csv', encoding='cp949')
labels = to_categorical(data['label'], num_classes=5)
print(data.label.value_counts())
stopwords=['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다',
           ',', '대숲', ',,', '하이', '대학', '안녕', '익명', '글쓴이', '쓰니', '오늘', '나', '너', '저', '누나', '오빠',
           '기분', '정말', '매우', '몹시', '너무', '엄청', '아주', '훨씬', '가장', '최고', '더', '덜', '맣이', '조금']
X_train=[]
for sentence in data['text']:
    temp_X = []
    temp_X=okt.morphs(sentence, stem=True) # 토큰화
    temp_X=[word for word in temp_X if not word in stopwords] # 불용어 제거
    X_train.append(temp_X)
max_len = 30
max_words = 8000
tokenizer = Tokenizer(num_words=max_words, lower=False)
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)
X = pad_sequences(sequences, maxlen=max_len)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.20)
epochs = 100
emb_dim = 256
batch_size = 256
model = Sequential()
model.add(Embedding(max_words, emb_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.5))
model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(5, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',patience=7, min_delta=0.0001)])

model_json = model.to_json()
with open("model_1.json", "w") as json_file :
    json_file.write(model_json)
model.save_weights("model_1.h5")
print("Saved model to disk")
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()