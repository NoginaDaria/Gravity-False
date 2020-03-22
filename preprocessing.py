import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from string import punctuation
from nltk.tokenize import RegexpTokenizer
import enchant

'''
exp: preprocess_text()

SnowballStemmer - стеммер, к которому можно подключить русский язык. Может быть, есть стеммер для русского получше
nltk.corpus.stopwords - русские стоп-слова, может, есть более полный список
nltk.tokenize.RegexpTokenizer - по названию понятно, думаю, занимается тем, что оставляет слова по регуляркам
enchant - хорошая библиотека, но я пока до конца не разобралась, как с ней работать. нужна для поиска опечаток. 
плюс в том, что есть русский язык. enchant.Dict("ru") создаёт словарь, enchant_d.check(token) выдаёт bool, есть ли токен в словаре
'''

enchant_d = enchant.Dict("ru")
mystem = SnowballStemmer("russian")
russian_stopwords = stopwords.words("russian")
tokenizer = RegexpTokenizer(r'\w+')

def preprocess_text(text):
    text = ' '.join(tokenizer.tokenize(title))
    tokens = mystem.stem(text.lower()).split()
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation \
              #and enchant_d.check(token) \
              and not token.isdigit()]
    text = " ".join(tokens)
    
    return text
