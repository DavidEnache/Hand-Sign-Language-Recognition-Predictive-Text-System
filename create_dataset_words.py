import nltk
import pickle
import numpy as np
from collections import Counter, defaultdict
from nltk.corpus import brown
import re

# Installazione dei pacchetti necessari per il funzionamento degli algoritmi
nltk.download('words')
nltk.download('brown')

# Crea il dataset delle parole e lo salva in un file pickle
def create_dataset_words():
    words = np.array([word for word in nltk.corpus.words.words() if re.fullmatch(r"[a-zA-Z]+", word)])
    word_freq = Counter(words)
    with open("dataset-words-en.pkl", "wb") as f:
        pickle.dump(word_freq, f)

# Crea un modello di probabilità basato sui bigrammi del corpus Brown
def create_bigram_model_from_corpus():
    words = np.array([word.lower() for word in brown.words() if re.fullmatch(r"[a-zA-Z]+", word)])
    bigram_model = defaultdict(Counter)

    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        bigram_model[word1][word2] += 1

    for word1 in bigram_model:
        total_count = sum(bigram_model[word1].values())
        for word2 in bigram_model[word1]:
            bigram_model[word1][word2] /= total_count

    with open("bigram-model.pkl", "wb") as f:
        pickle.dump(bigram_model, f)
    

if __name__ == "__main__":
    create_dataset_words()
    create_bigram_model_from_corpus()