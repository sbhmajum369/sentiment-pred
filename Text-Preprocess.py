

import nltk
#nltk.download('all')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import io

x = []
stop = set(stopwords.words('english'))

with open('Reviews.txt','r') as file:
	for value in file:
		#print(value)
		temp = value.split('\n')
		x.append(temp[0])

# print(x[1])

q=len(x)
print(q)
result=[]

# Removing stopwords and character symbols
for i in range(0,q):
	inp=re.sub(r'[\?\$\!\(\|\)\[\]\;\%\@\#\*\,\.]','',x[i])
	inp=inp.lower()
	#print(inp)
	tokens=word_tokenize(inp)
	#print(tokens)
	w=len(tokens)
	for r in range(0,w):
		if not tokens[r] in stop:
			result.append(tokens[r])
			result.append(" ")
	result.append("\n")


# print(result[0],result[1],result[2])

# Saving the processed reviews as a file 
appendFile = open('filteredtext.txt','a')
for w in result:
  appendFile.write(w)
appendFile.close()
