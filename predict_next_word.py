import pickle

# Caricamento del dataset di parole e bigramma
with open('./dataset-words-en.pkl', 'rb') as f:
    word_freq = pickle.load(f)

with open('./bigram-model.pkl', 'rb') as f:
    bigram_model = pickle.load(f)

def suggest_next_word(context, current_input, bigram_model, word_freq, top_n=3):
    current_input = current_input.lower()
    context = context.lower()

    # Se non c'è contesto, suggerisci parole che iniziano con l'input attuale
    if not context.strip():
        matching_words = [word for word in word_freq if word.startswith(current_input)]
        sorted_words = sorted(matching_words, key=lambda w: -word_freq[w])
        return sorted_words[:top_n]

    last_word = context.split()[-1].lower()
    next_word_candidates = bigram_model.get(last_word, {})

    # Inizializza la lista finale
    final_words = []  

    # Se ci sono suggerimenti dai bigrammi
    if next_word_candidates:
        # Ordina i bigrammi per probabilità e filtra quelli che iniziano con l'input parziale
        sorted_suggestions = sorted(next_word_candidates.items(), key=lambda x: -x[1])
        filtered_suggestions = [word for word, _ in sorted_suggestions if word.startswith(current_input)]

        # Aggiungi i suggerimenti dei bigrammi se disponibili
        final_words.extend(filtered_suggestions[:top_n])

    # Se non abbiamo ancora abbastanza suggerimenti, aggiungi parole dal dizionario
    if len(final_words) < top_n:
        matching_words = [word for word in word_freq if word.startswith(current_input)]
        dict_words = sorted(matching_words, key=lambda w: -word_freq[w])

        # Aggiungi parole dal dizionario fino a riempire top_n
        final_words.extend(dict_words[:top_n - len(final_words)])

    return final_words


# Funzione che ritorna un gruppo di 3 parole
def get_word_suggestions(current_input, current_phrase):
    # Restituisce i 3 suggerimenti basati sull'input corrente e sulla frase
    return suggest_next_word(current_phrase, current_input, bigram_model, word_freq)
