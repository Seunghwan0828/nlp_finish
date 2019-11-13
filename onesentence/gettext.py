from pykospacing import spacing
# import json
import warnings

warnings.filterwarnings("ignore")


def get_texts(data_path):
    # with open(data_path, 'rt', encoding='UTF8') as json_file:
    json_data = data_path
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

    return texts
