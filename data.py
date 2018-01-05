import csv
import nltk
import numpy as np

#from model import EmbeddingNeuralNetwork

def load_B_task_dataset(file_path):
    assert isinstance(file_path, str)  # Type check
    messages, polarities = [], []
    with open(file_path) as dataset_file:
        for instance in csv.reader(dataset_file, delimiter='\t'):
            context, message, polarity = instance[-3], instance[-1], instance[-2]
            messages.append(message)
            if polarity == 'negative':
                polarities.append(0)
            elif polarity == 'positive':
                polarities.append(2)
            else:
                polarities.append(1)
    return messages, polarities


def process_messages(raw_messages):  # For raw messages, each message is str
    from nltk.corpus import stopwords
    try:
        stop_words = set(stopwords.words())
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words())
    punc=[".",";","!",":","(",")","[","]","?",",","&"]
    assert isinstance(raw_messages, list) and all(isinstance(msg, str) for msg in raw_messages)  # Type check
    tokenizer = nltk.tokenize.TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokened_messages = [tokenizer.tokenize(msg) for msg in raw_messages]
    clean_tokens = [[token for token in tokens if token not in stop_words and not (token.startswith("http") or token.startswith("www")) and token not in punc] for tokens in tokened_messages]
    return clean_tokens


def generate_vocabulary(processed_messages):  # For processed messages, each message is list of tokens
    assert isinstance(messages, list) and all(isinstance(msg, list) for msg in processed_messages)  # Type check
    vocabulary = set()
    for message in processed_messages:
        vocabulary.update(message)
    return vocabulary


def get_indices(vocabulary):
    assert isinstance(vocabulary, set)
    words_indices = {}
    index = 0
    for word in vocabulary:
        words_indices[word] = index
        index += 1
    return words_indices


def transform_to_indices(processed_messages, words_indices):
    assert isinstance(processed_messages, list) and all(isinstance(msg, list) for msg in processed_messages)
    assert isinstance(words_indices, dict) and all(isinstance(v, int) for v in words_indices.values())  # Type check
    return [[words_indices[w] for w in msg] for msg in processed_messages]


if __name__ == '__main__':  # Just a testing code
    import sys
    messages, polarities = load_B_task_dataset(sys.argv[1])
    p_messages = process_messages(messages[:10])
    vocab = generate_vocabulary(p_messages)
    vocab_indices = get_indices(vocab)
    for msg, msg_idx in zip(p_messages, transform_to_indices(p_messages, vocab_indices)):
        print(msg)
        #print(msg_idx)

    #print("taille vocabulaire = "+str(len(vocab)))