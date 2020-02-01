# Reviews to Rating prediction

First download the json file from: https://www.yelp.com/dataset/download. Then from 'review.json' extract the 'text' and 'stars' in 2 separate .txt files: "Reviews.txt" and "Ratings.txt".

Install all the dependencies.

Afterwards run the files in the following order:

1) Text-Preprocess.py

2) main.py

Here, different neural architectures are designed and tested on text data. Models tested include: LSTM, biLSTM, 1-D CNN, GRU and Feed-forward network.

GRU and LSTM provided best results on this dataset.

