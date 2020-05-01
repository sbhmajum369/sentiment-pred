
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from Models import LSTM_model1, bilstm_model, CNN_model, FFN, new_GRU		# Importing the models

def summarize_diagnostics(history):
	# plot loss
	# plt.subplot(1,2,1)
	plt.title('Cross Entropy Loss')
	plt.plot(history.history['loss'], color='blue', label='train')
	plt.plot(history.history['val_loss'], color='orange', label='test')
	plt.show()
	# plot accuracy
	# plt.subplot(1,2,2)
	plt.title('Classification Accuracy')
	plt.plot(history.history['acc'], color='blue', label='train')
	plt.plot(history.history['val_acc'], color='orange', label='test')
	plt.show()

xinp=[]
with open("filteredtext.txt","r") as file:
	for value in file:
		temp = value.split('\n')
		xinp.append(temp[0])

p=len(xinp)
print("Number of reviews:",p)

# Converting texts to sequences
tokenizer = Tokenizer()
xinp=np.array(xinp)
tokenizer.fit_on_texts(xinp)
X=tokenizer.texts_to_sequences(xinp)
X = pad_sequences(X)
#print(X[0])

n2=X.shape[1]			# Input length
vocab_size = len(tokenizer.word_counts)
print(vocab_size)				# Total number of words

# Output Ratings pre-processing
y = []
with open("Ratings.txt","r") as rat:
	for value in rat:
		temp=value.split('.0')
		y.append(temp[0])
y=np.array(y)
encode = OneHotEncoder(sparse=False)
Y=encode.fit_transform(np.reshape(y,(y.shape[0], 1)))

# Training
X_train = X[:60000, :]		# Training set length
X_val = X[60000:80000, :]	# Validation set length
X_test=X[80000:100000, :]	# Test set length
Y_train= Y[:60000, :]
Y_val= Y[60000:80000, :]
Y_test= Y[80000:100000, :]

# Selecting the model to run
# model=CNN_model(vocab_size, n2)
# model=new_GRU((n2,),vocab_size)
# model=bilstm_model((n2,),vocab_size)
# model=LSTM_model1((n2,),vocab_size)
model= FFN(vocab_size)

history=model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=50, epochs=3, verbose=1)
loss, accuracy=model.evaluate(X_test, Y_test, batch_size=50)

# Result Visualization
summarize_diagnostics(history)
print("Test Accuracy:",accuracy*100,"%")
print("Test Loss:",loss)

