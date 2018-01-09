# -*- coding: utf-8 -*-
import csv
import nltk
import numpy as np
import re
import gensim


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


def clean_digit(token):
    if re.search(r'.*\d.*', token): #whatever_digit_whatever
        return False
    return True


def clean_url(token):
    if re.search(r'\w+\.\w+', token): #word.word
        return False
    if not (token.startswith("http") or token.startswith("www")):
        return True
    return False


def clean_punc(token):
    punc = [".", ";", "!", ":", "(", ")", "[", "]", "?", ",", "&", "-", "$", "~", "#", "{", "}", "/", "\\", '\'', '\"', '<', '>', '`', "+", "|", "^","@","%","=","*"]
    if token in punc:
        return False
    if re.search(r'-+>', token): #----->
        return False
    if re.search(r'<-+', token): # <-----
        return False
    if re.search(r'\.+', token): #........
        return False
    if re.search(r'(\.\s)+\.', token): #. . . .
        return False
    if re.search(r':+', token): #:::::
        return False
    if re.search(r'\</?\w>', token): #Balises
        return False
    if re.match(r'_+', token): #____
        return False
    return True


def clean_tag(token):
    if re.search(r'@\w+', token): #@person , Emails
        return False
    return True


def clean_hashtag(token):
    if token.startswith("#"):
        hashtag = token[1:]
        #print("hashtag = "+ hashtag)
        hashtag_parts = []
        part = ""
        for c in hashtag:
            if(c.islower()):
                part += c
            else: #upper
                if(part != ""):
                    hashtag_parts.append(part)
                    part = ""+c
                else:
                    part = ""+c
        if (part != ""):
            hashtag_parts.append(part)
        #print("hashtag parts = "+ str(hashtag_parts))

    return token


def process_messages(raw_messages):  # For raw messages, each message is str
    from nltk.corpus import stopwords
    try:
        stop_words = set(stopwords.words())
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words())
    assert isinstance(raw_messages, list) and all(isinstance(msg, str) for msg in raw_messages)  # Type check
    tokenizer = nltk.tokenize.TweetTokenizer(preserve_case=True, strip_handles=True, reduce_len=True)
    tokened_messages = [tokenizer.tokenize(msg) for msg in raw_messages]
    clean_tokens = [[token for token in tokens if token not in stop_words and clean_url(token) and clean_digit(token) and clean_punc(token) and clean_tag(token)] for tokens in tokened_messages]
    no_hashtag_tokens = [[clean_hashtag(token) for token in tokens] for tokens in clean_tokens]
    return no_hashtag_tokens


def generate_vocabulary(processed_messages):  # For processed messages, each message is list of tokens
    assert isinstance(processed_messages, list) and all(isinstance(msg, list) for msg in processed_messages)  # Type check
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


def create_vocabulary_embeddings(vocabulary, p_messages):
    p_messages.append(["<<<<<<OOV>>>>>>>"])
    embedding_size = 100
    model = gensim.models.Word2Vec(p_messages, min_count = 1, size = embedding_size)
    i=0
    vocabulary_embeddings = []
    embeddings_indices = {}
    for word in vocabulary:
        try:
            embedding = model[word]
        except:
            embedding =  ["<<<<<<OOV>>>>>>>"]
            print("yes")
        vocabulary_embeddings.append(embedding)
        embeddings_indices[word] = i
        i += 1
    vocabulary_embeddings = np.array(vocabulary_embeddings)
    return vocabulary_embeddings, embeddings_indices


if __name__ == '__main__':  # Just a testing code
    import sys
    messages, polarities = load_B_task_dataset(sys.argv[1])
    p_messages = process_messages(messages)

