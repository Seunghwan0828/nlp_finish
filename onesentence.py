from flask import Flask, request, jsonify
from pykospacing import spacing
from krwordrank.sentence import summarize_with_sentences
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)


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
        list_sentence1 = sentence.replace('\n', '').replace('""', '').split('\n')
        for list_sentence2 in list_sentence1:
            list_sentence3 = list_sentence2.replace('.', '.  ..').split('  ..')
            for list_sentence4 in list_sentence3:
                list_sentence = list_sentence4.replace('?', '?  ??').split('  ??')
                for lines in list_sentence:
                    line = spacing(lines).strip()
                    data_list.append(line)
    texts = list(set(data_list))
    if '' in texts:
        texts.remove('')
    penalty = lambda x: 0 if (7 <= len(x) <= 85) else 1
    stopwords = {'오늘'}
    keywords, sents = summarize_with_sentences(
        texts,
        penalty=penalty,
        stopwords=stopwords,
        diversity=0.5,
        num_keywords=7,
        num_keysents=1,
        scaling=lambda x: 1,
        verbose=False,
        min_count=1)
    keyword = []
    for sent in sents:
        user = sent
    return jsonify({"onesentence": user})  # 받아온 데이터를 다시 전송


if __name__ == "__main__":
    app.run(host='0.0.0.0', port="8080")
