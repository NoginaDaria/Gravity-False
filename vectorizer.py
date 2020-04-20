import pickle
with open('page_dict.p', 'rb') as fp:
    page_dict = pickle.load(fp)
    
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=0.02, max_df=0.1, stop_words='english')
vectors = vectorizer.fit_transform(page_dict.values()).toarray()
vector_map = dict(zip(page_dict.keys(), vectors.tolist()))
import pickle
with open('vector_map.p', 'wb') as fp:
    pickle.dump(vector_map, fp)
    
import pandas as pd
train_data = pd.read_csv('./anomaly-detection-competition-ml1-ts-spring-2020/train_groups.csv')
traingroups_titledata = {}
for i in range(len(train_data)):
    new_doc = train_data.iloc[i]
    doc_group = new_doc['group_id']
    doc_id = new_doc['doc_id']
    target = new_doc['target']
    title = vector_map[doc_id]
    if doc_group not in traingroups_titledata:
        traingroups_titledata[doc_group] = []
    traingroups_titledata[doc_group].append((doc_id, title, target))
    
y_train = []
X_train = []
groups_train = []
for new_group in traingroups_titledata:
    docs = traingroups_titledata[new_group]
    for k, (doc_id, title, target_id) in enumerate(docs):
        y_train.append(target_id)
        groups_train.append(new_group)
        all_dist = []
        for j in range(0, len(docs)):
            if k == j:
                continue
            doc_id_j, title_j, target_j = docs[j]
            all_dist.append(cosine_similarity(title, title_j))
        X_train.append(sorted(all_dist, reverse=True)[0:25]    )
X_train = np.array(X_train)
y_train = np.array(y_train)
groups_train = np.array(groups_train)
print (X_train.shape, y_train.shape, groups_train.shape)

with open('X_train.p', 'wb') as fp:
    pickle.dump(X_train, fp)
    
with open('y_train.p', 'wb') as fp:
    pickle.dump(y_train, fp)
    
with open('groups_train.p', 'wb') as fp:
    pickle.dump(groups_train, fp)
