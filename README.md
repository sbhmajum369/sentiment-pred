# Reviews to Rating prediction
## Introduction

This project predicts the rating, from the review on a public website. This Deep learning based approach utilizes Natural Language Processing (NLP) technique for getting a comprehensive idea of a business's public image based on the public reviews left on its comment section of the webpage.

Currently, we have utilized the 'Reviews' dataset from Yelp, which provides real-world samples, for a Supervised Learning approach. In order to use any other dataset, the files have to be processed accordingly to generate 2 files: One containing the reviews and another, containing the corresponding ratings.


## Steps for Training and Testing 

Before we begin, first dowload the repo using: git clone https://github.com/smajum-AI/sentiment-pred.git

A) Download the json file from: https://www.yelp.com/dataset/download.

From 'review.json' extract the 'text' and 'stars' in 2 separate .txt files: "Reviews.txt" and "Ratings.txt".

B) Install all the dependencies. 

If you have Python 3, then do:

pip3 install 'library name'

else,

pip install 'library name'

For this project you will need: (Additional)
1) Tensorflow
2) NLTK
3) Regex
4) scikit-learn
5) Matplotlib.

C) Afterwards run the files in the following order:

1) Text-Preprocess.py: For filtering and processing the text, before feeding it to the network.
2) main.py: For training and testing. Hyper-parameters can be changed accordingly, from inside the file.

Here, different neural architectures are designed and tested on text data. Models tested include: LSTM, biLSTM, 1-D CNN, (GRU+RNN) and Feed-forward network.

(GRU+RNN) architecture provided best result of 86%, during testing, on this dataset.

