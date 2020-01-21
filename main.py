

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


xinp=[]
with open("filteredtext.txt","r") as file:
	for value in file:
		temp = value.split('\n')
		xinp.append(temp[0])

p=len(xinp)
xinp=np.array(xinp)

# Converting texts to sequence
tokenizer = Tokenizer()
tokenizer.fit_on_texts(xinp)
X=tokenizer.texts_to_sequences(xinp)
X = pad_sequences(X)
#print(X[0])

vocab_size = len(tokenizer.word_counts)
print(vocab_size)

# Output Ratings pre-processing
y = []
with open("Ratings.txt","r") as rat:
	for value in rat:
		temp=value.split('.0')
		y.append(temp[0])
y=np.array(y)
encode = OneHotEncoder(sparse=False)
Y=encode.fit_transform(np.reshape(y,(y.shape[0], 1)))

# Model Architectures
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential

def LSTM_model1():
  model1 = Sequential()
  model1.add(Embedding(input_dim=vocab_size, output_dim=512))
  model1.add(LSTM(256, activation='tanh', kernel_initializer='he_uniform'))
  #model1.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
  model1.add(Dense(20, activation='relu', kernel_initializer='he_uniform'))
  model1.add(Dense(5, activation='softmax'))
  model1.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
  return model1
def bilstm_model():
  model2 = Sequential()
  model2.add(Embedding(input_dim=vocab_size, output_dim=384))
  layer = LSTM(128)
  model2.add(Bidirectional(layer))
  model2.add(Dense(20, activation='relu', kernel_initializer='he_uniform'))
  model2.add(Dense(5, activation='softmax'))
  model2.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
  return model2

# Training
X_train = X[:40000, :]
X_val = X[40000:60000, :]
X_test=X[60000:70000, :]
Y_train= Y[:40000, :]
Y_val= Y[40000:60000, :]
Y_test= Y[60000:70000, :]

model=LSTM_model1()
model=bilstm_model()
history=model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=40, epochs=2, verbose=2)
loss, accuracy=model.evaluate(X_test, Y_test, batch_size=50)

