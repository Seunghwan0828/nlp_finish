from krwordrank.sentence import summarize_with_sentences


def summary_text(texts):
    texts = texts
    penalty = lambda x:0 if (7 <= len(x) <= 85) else 1
    stopwords = {'오늘'}
    keywords, sents = summarize_with_sentences(
        texts,
        penalty=penalty,
        stopwords = stopwords,
        diversity=0.5,
        num_keywords=7,
        num_keysents=1,
        scaling=lambda x:1,
        verbose=False,
        min_count = 1
    )
    keyword = []
    
    for sent in sents:
        print(sent)

    return sent


