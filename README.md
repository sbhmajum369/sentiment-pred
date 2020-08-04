# Reviews to Rating prediction

## Steps
A) First download the json file from: https://www.yelp.com/dataset/download.

Then from 'review.json' extract the 'text' and 'stars' in 2 separate .txt files: "Reviews.txt" and "Ratings.txt".

B) Install all the dependencies. 

If you have Python 3, then do: pip3 install 'library name'

else, pip install 'library name'

For this project you will need: (Additional)
1) Tensorflow
2) NLTK
3) Regex
4) scikit-learn
5) Matplotlib.

C) Afterwards run the files in the following order:

1) Text-Preprocess.py

2) main.py

Here, different neural architectures are designed and tested on text data. Models tested include: LSTM, biLSTM, 1-D CNN, GRU and Feed-forward network.

GRU and LSTM provided best result of 86%, during testing, on this dataset.

