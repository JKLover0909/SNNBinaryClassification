


# update eng stop words
# import string
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# stop_words = stopwords.words('english')
# with open("e-stopwords.txt", 'w') as file:
#     for i in stop_words:
#         file.write(i + '\n')


def get_vie_stopwords():
    vie_stopwords_file = 'vietnamese-stopwords.txt'
    stop_words = []
    with open(vie_stopwords_file, encoding = 'utf-8') as file:
        for line in file:
            stop_words.append(line.strip())
    
    return stop_words

def get_eng_stopwords():
    eng_stopwords_file = 'e-stopwords.txt'
    stop_words = []
    with open(eng_stopwords_file) as file:
        for line in file:
            stop_words.append(line.strip())
    return stop_words

def get_stopwords():
    stop_words = get_eng_stopwords()
    stop_words += get_vie_stopwords()
    return stop_words
