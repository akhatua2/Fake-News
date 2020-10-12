import pandas as pd
from nltk.corpus import stopwords
import re
import time
import pickle
print("Modules imported")

df = pd.read_csv('train.csv')
df.dropna(inplace = True)
df.reset_index(inplace =True)
df = df.sample(frac = 1).reset_index(drop = True)

x = df[['title' , 'author' , 'text']]
y = df['label']


s = time.time()
corpus = []
for i in range(len(x)):
    if i + 1 % 100 == 0:
        print(i)

    text = re.sub('[^a-zA-Z]', " ", x['text'][i])
    text = text.lower()
    text = text.split()

    word = [words for words in text if words not in stopwords.words('english')]
    word = " ".join(word)
    corpus.append(word)
print('done')

with open('outfile', 'wb') as fp:
    pickle.dump(corpus, fp)

print((time.time() - s) * 1000)









