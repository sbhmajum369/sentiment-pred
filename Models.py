from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Input, Dropout, GlobalAveragePooling1D, Conv1D, MaxPool1D, AveragePooling1D, GRU, SimpleRNN
from tensorflow.keras.models import Sequential

def LSTM_model1(input_shape, vocab_len):
  model1 = Sequential()
  embedding_layer=Embedding(input_dim=vocab_len+1, output_dim=256)
  inp_layer = Input(shape=input_shape)
  model1.add(inp_layer)
  model1.add(embedding_layer)
  model1.add(LSTM(256, return_sequences=True))
  model1.add(LSTM(128))
  model1.add(Dense(256, activation='relu'))
  model1.add(Dropout(0.2))
  model1.add(Dense(128, activation='relu'))
  model1.add(Dropout(0.2))
  model1.add(Dense(64, activation='relu'))
  model1.add(Dense(5, activation='sigmoid'))
  model1.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
  return model1

def bilstm_model(input_shape, vocab_len):
  model2 = Sequential()
  embedding_layer=Embedding(input_dim=vocab_len+1, output_dim=256)
  inp_layer = Input(shape=input_shape)
  model2.add(inp_layer)
  model2.add(embedding_layer)
  model2.add(Bidirectional(LSTM(256,  return_sequences=True)))
  model2.add(Bidirectional(LSTM(128)))
  model2.add(Dense(256, activation='relu'))
  model2.add(Dropout(0.2))
  model2.add(Dense(128, activation='relu'))
  model2.add(Dropout(0.2))
  model2.add(Dense(64, activation='relu'))
  model2.add(Dense(5, activation='softmax'))
  model2.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
  return model2

def CNN_model(vocab_len, vec_len):
  model3 = Sequential()
  embedding_layer=Embedding(input_dim=vocab_len+1, output_dim=256)
  inp_layer = Input(shape=(vec_len,))
  model3.add(inp_layer)
  model3.add(embedding_layer)
  model3.add(Conv1D(64, 4, activation='tanh', kernel_initializer='he_uniform', input_shape=(None, 256)))
  model3.add(AveragePooling1D(4))
  model3.add(Conv1D(128, 4, activation='tanh', kernel_initializer='he_uniform'))
  model3.add(AveragePooling1D(4))
  model3.add(GlobalAveragePooling1D())
  model3.add(Dense(256, activation='relu'))
  model3.add(Dropout(0.2))
  model3.add(Dense(128, activation='relu'))
  model3.add(Dropout(0.2))
  model3.add(Dense(64, activation='relu'))
  model3.add(Dense(5, activation='softmax'))
  model3.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
  return model3

def FFN(vocab_len):
  model3 = Sequential()
  model3.add(Embedding(input_dim=vocab_len+1, output_dim=256))
  model3.add(GlobalAveragePooling1D())
  model3.add(Dense(256, activation='relu'))
  model3.add(Dropout(0.2))
  model3.add(Dense(128, activation='relu'))
  model3.add(Dropout(0.2))
  model3.add(Dense(64, activation='relu'))
  model3.add(Dense(5, activation='sigmoid'))
  model3.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
  return model3

def new_GRU(input_shape, vocab_len):
  inp_layer = Input(shape=input_shape)
  embedding=Embedding(input_dim=vocab_len+1, output_dim=256)(inp_layer)
  Layer1=GRU(256, return_sequences=True)(embedding)
  Layer2=SimpleRNN(128)(Layer1)
  Layer3 = Dense(128, activation='relu')(Layer2)
  Layer4 = Dense(64, activation='relu')(Layer3)
  Layer5 = Dense(32, activation='relu')(Layer4)
  Fin_Layer=Dense(units=5, activation='softmax')(Layer5)
  New_Model = tf.keras.Model(inputs=inp_layer, outputs=Fin_Layer)
  New_Model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
  return New_Model
