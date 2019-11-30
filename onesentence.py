from flask import Flask, request, jsonify
# from pykospacing import spacing
from krwordrank.sentence import summarize_with_sentences
import warnings
from flask_cors import CORS
from konlpy.tag import Okt
import torch
import torch.nn as nn
from torchtext.data import Field, TabularDataset, Iterator
tokenizer = Okt()

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)


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
        if data_list1[i:i + 1] == ['']:
            data_list1.remove('')
            texts = data_list1
        elif data_list1[i:i + 1] == ['"']:
            data_list1.remove('"')
            texts = data_list1
        else:
            texts = data_list1
    if texts[len(texts) - 1:len(texts)] == ['']:
        texts.pop()
    elif texts[len(texts) - 1:len(texts)] == ['"']:
        texts.pop()
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
        k = tokenizer.nouns(l)
        if len(k) > 0:
            for n in k:
                if len(n) > 1:
                    keyword.append(n)
    print(before_sentiment)
    print(keywords)
    print(keyword)
    # defining the Field
    TEXT = Field(sequential=True,
                 use_vocab=True,
                 tokenize=tokenizer.morphs,
                 lower=True,
                 batch_first=True)
    LABEL = Field(sequential=False,
                  use_vocab=False,
                  preprocessing=lambda x: int(x),
                  batch_first=True,
                  is_target=True)
    ID = Field(sequential=False,
               use_vocab=False,
               is_target=False)

    # divide train_data and test_data by using TabularDataset.splits function
    train_data, test_data = TabularDataset.splits(
        path='./data', format='tsv',
        train="ratings_train.txt",
        test="ratings_test.txt",
        fields=[('id', ID), ('text', TEXT), ('label', LABEL)],
        skip_header=True)

    # making Vocabulary
    TEXT.build_vocab(train_data, min_freq=2)

    # makeing model by using LSTM
    class SentimentCls(nn.Module):
        def __init__(self, vocab_size, embed_size, hidden_size, output_size,
                     num_layers=3, batch_first=True, bidirec=True, dropout=0.5):
            super(SentimentCls, self).__init__()
            self.hidden_size = hidden_size
            self.n_layers = num_layers
            self.n_direct = 2 if bidirec else 1
            self.embedding_layer = nn.Embedding(vocab_size, embed_size)
            self.rnn_layer = nn.LSTM(input_size=embed_size,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     batch_first=batch_first,
                                     bidirectional=bidirec,
                                     dropout=0.5)
            self.dropout = nn.Dropout(dropout)
            self.linear = nn.Linear(self.n_direct * hidden_size, output_size)

        def forward(self, x):
            embeded = self.dropout(self.embedding_layer(x))
            hidden, cell = self.init_hiddens(x.size(0), self.hidden_size, device=x.device)
            output, (hidden, cell) = self.rnn_layer(embeded, (hidden, cell))
            last_hidden = torch.cat([h for h in hidden[-self.n_direct:]], dim=1)
            scores = self.linear(last_hidden)
            return scores.view(-1)

        def init_hiddens(self, batch_size, hidden_size, device):
            hidden = torch.zeros(self.n_direct * self.n_layers, batch_size, hidden_size)
            cell = torch.zeros(self.n_direct * self.n_layers, batch_size, hidden_size)
            return hidden.to(device), cell.to(device)

    # setting arguments for model
    vocab_size = len(TEXT.vocab)  # the size of vocabulary
    embed_size = 128  # the size of embedding
    hidden_size = 256  # the size of hidden layer
    output_size = 1  # the size of output layer
    num_layers = 3  # the number of RNN layer
    batch_first = True  # if RNN's frist dim of input is the size of minibatch
    bidirec = True  # BERT
    dropdout = 0.5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # device

    # model
    model = SentimentCls(vocab_size, embed_size, hidden_size, output_size,
                         num_layers, batch_first, bidirec, dropdout).to(DEVICE)

    # loading model
    model.load_state_dict(torch.load("./movie-review-model.pt", map_location=torch.device('cpu')))
    print("Load Complete!")

    # test
    def test_input(model, field, tokenizer, device, sentence):
        sentence = sentence
        x = field.process([tokenizer.morphs(sentence)]).to(device)
        output = model(x)
        pred = torch.sigmoid(output).item()*4
        if pred > 3.2:
            x = pred
        elif pred > 2.4:
            x = pred
        elif pred > 1.6:
            x = pred
        elif pred > 0.8:
            x = pred
        elif pred > 0:
            x = pred
        return x
    for i in range(3):
        sentiment.append(test_input(model, field=TEXT, tokenizer=tokenizer, device=DEVICE, sentence=before_sentiment[i]))
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
    app.run(port="8000")
