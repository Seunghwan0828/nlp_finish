from flask import Flask, request, jsonify
from onesentence import gettext, summarytext

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/summary', methods=['POST'])
def summary():
    text = request.get_json()  # json 데이터를 받아옴
    texts = gettext.get_texts(text)
    user = summarytext.summary_text(texts)
    return jsonify({"onesentence": user})  # 받아온 데이터를 다시 전송


if __name__ == "__main__":
    app.run()
