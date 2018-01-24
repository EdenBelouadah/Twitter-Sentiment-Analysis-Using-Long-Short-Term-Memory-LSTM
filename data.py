# -*- coding: utf-8 -*-
import csv
import nltk
import numpy as np
import re
import gensim

#loading data
def load_B_task_dataset(file_path):
    assert isinstance(file_path, str)  # Type check
    messages, polarities = [], []
    with open(file_path) as dataset_file:
        for instance in csv.reader(dataset_file, delimiter='\t'): #read the dataset as CSV file
            context, message, polarity = instance[-3], instance[-1], instance[-2]
            messages.append(message)
            if polarity == 'negative': #negative polarities are replaced with 0
                polarities.append(0)
            elif polarity == 'positive': #positive polarities are replaced with 2
                polarities.append(2)
            else: #neutral polarities are replaced with 1
                polarities.append(1) 
    return messages, polarities


def clean_digit(token): #check whether there is no digit
    if re.search(r'.*\d.*', token): #whatever_digit_whatever
        return False
    return True


def clean_url(token): #check whether there is not an URL
    if re.search(r'\w+\.\w+', token): #word.word
        return False
    if not (token.startswith("http") or token.startswith("www")):
        return True
    return False


def clean_punc(token): #check whether there is no punctuation
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


def clean_tag(token): #check whether there is no tag
    if re.search(r'@\w+', token): #@person , Emails
        return False
    return True

def get_hashtag_parts(token): #tokenize every hashtag whenever it is possible
    if token.startswith("#"):
        hashtag = token[1:]
        if hashtag.isupper():
            return [hashtag]
        hashtag_parts = []
        part = ""
        for c in hashtag:
            if(c.islower()):
                part += c
            else: #upper
                if(part != ""):
                    hashtag_parts.append(part)
                part = ""+c
        if part != "":
            hashtag_parts.append(part)
        return hashtag_parts
    return [token]


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
    clean_hashtag_messages = []
    for msg in tokened_messages:
        clean_msg = []
        for token in msg:
            hashtag_parts = get_hashtag_parts(token)
            for part in hashtag_parts:
                clean_msg.append(part.lower())
        clean_hashtag_messages.append(clean_msg)

    assert(len(tokened_messages) == len(clean_hashtag_messages))
    clean_tokens = [[token for token in tokens if token not in stop_words and clean_url(token) and clean_digit(token) and clean_punc(token) and clean_tag(token)] for tokens in tokened_messages]
    return clean_tokens


def generate_vocabulary(processed_messages):  # For processed messages, each message is list of tokens
    assert isinstance(processed_messages, list) and all(isinstance(msg, list) for msg in processed_messages)  # Type check
    vocabulary = set()
    for message in processed_messages:
        vocabulary.update(message)
    return vocabulary

#replace every word by its index
def get_indices(vocabulary):
    assert isinstance(vocabulary, set)
    words_indices = {}
    index = 0
    for word in vocabulary:
        words_indices[word] = index
        index += 1
    return words_indices


def transform_to_indices(processed_messages, words_indices): #transform messages of tokens to messages of indices.
    assert isinstance(processed_messages, list) and all(isinstance(msg, list) for msg in processed_messages)
    assert isinstance(words_indices, dict) and all(isinstance(v, int) for v in words_indices.values())  # Type check
    return [[words_indices[w] for w in msg] for msg in processed_messages]


def create_vocabulary_embeddings(vocabulary, p_messages, embedding_size = 100): #generate the embeddings values
    model = gensim.models.Word2Vec(p_messages, min_count = 1, size = embedding_size) #use word2vec
    i=0
    vocabulary_embeddings = []
    embeddings_indices = {}
    for word in vocabulary:
        embedding = model[word]
        vocabulary_embeddings.append(embedding)
        embeddings_indices[word] = i
        i += 1
    vocabulary_embeddings = np.array(vocabulary_embeddings)
    return vocabulary_embeddings, embeddings_indices


if __name__ == '__main__':  # Just a testing code
    import sys
    messages, polarities = load_B_task_dataset(sys.argv[1])
    p_messages = process_messages(messages)
