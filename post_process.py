import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

print("Modules imported")

with open('outfile', 'rb') as fp:
    corpus = pickle.load(fp)

df = pd.read_csv('train.csv')
df.dropna(inplace=True)
df.reset_index(inplace=True)
df = df.sample(frac=1).reset_index(drop=True)

x = df[['title', 'author', 'text']]
y = df['label']

voc_size = 6000
one_hot_sentence = [one_hot(words, voc_size) for words in corpus]
max_length_of_sent = 50
embedding_sent = pad_sequences(one_hot_sentence, padding='pre', maxlen=max_length_of_sent)
embedding_feature_size = 256

print("Starting model")
# model
model = Sequential()
model.add(Embedding(voc_size, embedding_feature_size, input_length=max_length_of_sent))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(128))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()

X = np.array(embedding_sent)
Y = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1000)

print("Starting training")
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128)

plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.savefig('plt_one.png')

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig('plt_two.png')

y_pred = model.predict_classes(x_test)

acc = accuracy_score(y_pred, y_test)
print(acc)
