import preprocessing
from bs4 import BeautifulSoup
import codecs

from glob import glob
files = glob('content/*')

page_dict = {}
for filename in files:
    doc_id = int(filename.split('/')[-1].split('.')[0])
    with codecs.open(filename, 'r', 'utf-8') as f:
        url = f.readline().strip()
        page = BeautifulSoup(f, 'lxml')
        for script in page(["script", "style"]):
            script.decompose()
        page_dict[doc_id] = preprocessing.preprocess_text(page.get_text())
        
import pickle
with open('page_dict.p', 'wb') as fp:
    pickle.dump(page_dict, fp)
