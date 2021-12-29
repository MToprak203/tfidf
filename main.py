import os
import math
from string import punctuation, digits
from collections import Counter

PATH = f"{os.getcwd()}"
os.chdir(PATH)

'''
Keeps the text files' names as key and stemmed words in a list as values.
'''
txt_dict = {}

''' 
    Keeps the text files' names as key,
    values are kept in a dict, which has stemmed words as keys and number of the words as values, in a tuple.
    The dict is returned object of Counter class.
'''
word_counts_dict = {}

words_tfidf = {}


def prepare_dicts():
    def prepare_words(lines):
        def remove_punctuation(sentence):
            for ch in sentence:
                if ch in punctuation or ch in digits:
                    sentence = sentence.replace(ch, " ")
            return sentence

        result_words = []
        for line in lines:
            line = remove_punctuation(line)
            word_list = line.split()
            word_list[0] = word_list[0].replace("\ufeff", "")
            for word in word_list:
                word = word.lower()
                result_words.append(word)
        return result_words

    def read_text_file(_file_path):
        with open(_file_path, 'r', encoding="utf-8") as _file:
            lines = _file.readlines()
        return prepare_words(lines)

    def clean_stop_words():
        for key in txt_dict.keys():
            for word in txt_dict[key].copy():
                if word in stop_words:
                    txt_dict[key].remove(word)

    def stem():
        for key in txt_dict.keys():
            for i in range(len(txt_dict[key])):
                if len(txt_dict[key][i]) < 5:
                    continue
                txt_dict[key][i] = txt_dict[key][i][:5]

    def word_counter():
        for key in txt_dict.keys():
            word_counts_dict[key] = Counter(txt_dict[key])

    for file in os.listdir("dataset"):
        if file.endswith(".txt"):
            file_path = f"{PATH}\\dataset\\{file}"
            txt_dict[file.replace(".txt", "")] = read_text_file(file_path)

    stop_words = read_text_file(f"{PATH}\\stopwords.txt")
    clean_stop_words()
    stem()
    word_counter()


def tfidf(term, doc):
    def tf():
        return word_counts_dict[doc][term]

    def df():
        _result = 0
        for _key in word_counts_dict.keys():
            if term in word_counts_dict[_key].keys():
                _result += 1
        return _result

    return tf() * math.log(len(word_counts_dict.keys()) / df())


def find_tfidf_values():
    for key in word_counts_dict.keys():
        words_tfidf[key] = {}
        for term in word_counts_dict[key].keys():
            words_tfidf[key][term] = tfidf(term, key)

        words_tfidf[key] = {k: v for k, v in sorted(words_tfidf[key].items(), key=lambda item: item[1], reverse=True)}


prepare_dicts()
find_tfidf_values()


for key in words_tfidf.keys():
    result = {k: "{:.2f}".format(v) for (k, v) in list(words_tfidf[key].items())[:10]}
    print(f"{key}'s top 10 tfidf terms are : {result}\n")
