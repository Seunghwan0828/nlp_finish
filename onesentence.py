from flask import Flask, request, jsonify
# from pykospacing import spacing
from krwordrank.sentence import summarize_with_sentences
import warnings
from flask_cors import CORS
from konlpy.tag import Okt
okt = Okt()
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
    user = []
    keyword = []
    for sent in sents:
        user.append(sent)
    keywords = list(keywords.keys())
    for l in keywords:
        k = okt.nouns(l)
        if len(k) > 0:
            keyword = keyword + list(k[0:1])
    return jsonify({"onesentence": user,
                    "keyword": keyword})  # 받아온 데이터를 다시 전송


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080")
