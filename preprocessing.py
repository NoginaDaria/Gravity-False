import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from string import punctuation
from nltk.tokenize import RegexpTokenizer
import enchant

'''
exp: 
preprocess_text('Присутствие CSD является одной из характерных черт Y-бокс-связывающих белков и позволяет причислить их к более обширной группе белков c доменом холодового шока.')
получается:
'присутствие cсд является одной характерных черт ы бокс связывающих белков позволяет причислить обширной группе белков c доменом холодового шок'

сразу видно проблемы: 
анлийские буквы переводятся в странные русские, 
некоторые слова корявятся, 
плохо работет стемминг

что надо сделать:
найти нормальный стеммер / лемматайзер для русского языка
научиться исправлять ошибки (а именно разобраться с JamSpeller)


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

enchant_d = enchant.Dict("ru")
mystem = SnowballStemmer("russian")
russian_stopwords = stopwords.words("russian")
tokenizer = RegexpTokenizer(r'\w+')

def preprocess_text(text):
    text = ' '.join(tokenizer.tokenize(text))
    tokens = mystem.stem(text.lower()).split()
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation \
              #and enchant_d.check(token) \
              and not token.isdigit()]
    text = " ".join([mystem.stem(token) for token in set(tokens)])
    
    return text
