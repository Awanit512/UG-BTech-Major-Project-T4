# -*- coding: utf-8 -*-
"""SpamDetection_1.1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CsVl7cfXrpeKPsXeC96b5w6espiBjHsq

##Spam detection using ML models
####By: Pintu (181co139), Awanit (181co161),Vivek (181co159), 
####

mount Googledrive
"""

from google.colab import drive
drive.mount('/content/drive')

"""Read the dataset"""

#Pintu 29/01/22
from sklearn import preprocessing
import pandas as pd

dataset=pd.read_csv('/content/drive/MyDrive/Colab datasets/Spam_dataset.csv', header=0, delimiter=",")

#remove the missing data.
dataset=dataset.dropna()
print("Head of the dataset =>\n",dataset.head()) #dataset.tail()

#Category -> Integer
#Ham = 0
#Spam = 1
# le = preprocessing.LabelEncoder()
# le.fit(dataset.Category)
# dataset['Category'] = le.transform(dataset.Category)

spam_cnt=0
for i in range(0, len(dataset['Category'])):
  if (dataset['Category'][i]=='ham'):
    dataset['Category'][i]="0"
  else:
    dataset['Category'][i]="1"
    spam_cnt+=1


print("\nTail of dataset after labelling =>\n",dataset.tail())
print("\nTotal spam texts are =>", spam_cnt," out of ",len(dataset))

"""PreProcessing

"""

#Pintu 29/01/22
#Preprocessing of the dataset step

#PUNCTUATION 
#library that contains punctuation
import string
string.punctuation
#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

#TOLOWER
#converting text into lower format.
def to_lower(text):
    lowered_text=text.lower()
    return lowered_text

#TOKENIZATION
#defining function for tokenization
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
def tokenization(text):
    tokens = word_tokenize(text)
    return tokens

#REMOVE STOPWORDS
#importing nlp library
import nltk
nltk.download('stopwords')
#Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')
print("\nFew english stopwords are: ",stopwords[0:10])

#defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

#STEMMING
#importing the Stemming function from nltk library
from nltk.stem.porter import PorterStemmer
#defining the object for stemming
porter_stemmer = PorterStemmer()
#defining a function for stemming
def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text

#LEMMATIZATION
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
#defining the function for lemmatization
def lemmatizer(text):
    lemmed_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemmed_text


##PRE-PROCESSING##
def text_preprocessing(text):
    punctuation_free=remove_punctuation(text)
    lower_text=to_lower(punctuation_free)
    tokenized_text=tokenization(lower_text)
    stopwords_free=remove_stopwords(tokenized_text)
    # stemmed_text=stemming(stopwords_free)

    ## TODO : In order to improve word root removing stemming
    stemmed_text = stopwords_free
    lemmatized_text=lemmatizer(stemmed_text)
    return [punctuation_free,lower_text,tokenized_text,stopwords_free,stemmed_text,lemmatized_text]

#Apply pre-processing function.
for i in range(0,len(dataset['Message'])):
  dataset['Message'][i]=text_preprocessing(dataset['Message'][i])[5]

#Many other steps include: URL removal, HTML tags removal, Rare words removal, Frequent words removal, Spelling checking, and many more.
#But we are not going for further pre-processing cause we need to detect spams and spam contains URL.

"""Verify pre-processing"""

#Pintu 29/01/22
#verify.

text="Let's See; CONNECTION Misunderstanding swimming; ASSure the Authentication: of Working of tExt Pre-processing"
[punctuation_free,lower_text,tokenized_text,stopwords_free,stemmed_text,lemmatized_text]=text_preprocessing(text)
print("Remove punctuation=> ",punctuation_free)
print("Lowered text      => ",lower_text)
print("Tokens separated  => ",tokenized_text)
print("Remove stopwords  => ",stopwords_free)
print("Stemming          => ",stemmed_text)
print("Lemmatizer        => ",lemmatized_text)

print("\nVerify the proper preprocessing:\n",dataset['Message'][0:7])

#success msg.
print("\nAll texts have been preprocessed successfully!!")

"""Create Vocabulary"""

#pintu 30/01/22

from tensorflow.keras.preprocessing.text import Tokenizer                    
from tensorflow.keras.preprocessing.sequence import pad_sequences

x=dataset['Message']
y=dataset['Category']

#create the dict.
tokenizer = Tokenizer(num_words=8000)
tokenizer.fit_on_texts(x)

#number of unique words in dict.
print("Number of unique words in dictionary=",len(tokenizer.word_index))

#replace words with their index in vocab.
x_train = tokenizer.texts_to_sequences(x)


# Adding 1 because of  reserved 0 index
vocab_size = len(tokenizer.word_index) + 1  

#size of random text in training set.
print("Length of random text=")
for i in range(10):
  print("Length of ",i+1," text is => ",len(x_train[i]))

#Maximum length of each text
maxlen = 30

#pad the short text and truncate longer texts.
x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)

"""Confirm that text is converted into vector"""

#pintu 30/01/22
print("First tweet=> ",x[0])
print("Text to seq=> ",x_train[1])
print("\nVerification successful!")

"""SMOTE: Synthetic Minority Oversampling TEchnique"""

#Pintu 30/01/22

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
#print(imblearn.__version__)
from collections import Counter
import numpy as np

msg,y= x_train, dataset['Category'].to_numpy()
print("Type of message column and class label => ",type(msg),type(y))
count = Counter(y)
print("\nInitial Count of each class is =>",count)

# over = SMOTE(sampling_strategy=0.2) #sampling_strategy argument tells that minority = 10% of majority.
# X, y = over.fit_resample(X, y)
# under = RandomUnderSampler(sampling_strategy=0.5) #sampling_strategy argument tells that majority = 1/0.5 of minority.
# X, y = under.fit_resample(X, y)

#Using Pipeline instead of doing separately.
over = SMOTE(sampling_strategy=0.2)
under = RandomUnderSampler(sampling_strategy=0.5)
steps_in_order = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps_in_order)

# transform the dataset
X, y = pipeline.fit_resample(msg, y)
# summarize the new class distribution
counter = Counter(y)
print("\nNew count after SMOTE of both class =>",counter)
print("\nSize of X and y => ",len(X),len(y))

"""Download zip glove file"""

!wget http://nlp.stanford.edu/data/glove.6B.zip

"""Unzip glove file"""

!unzip glove*.zip

"""Create embeddings of vocab."""

#Pintu 30/01/22
import numpy as np

#vocab: 'the': 1, mapping of words with integers in seq. 1,2,3..
#embedding: 1->dense vector

def embedding_for_vocab(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  
    # Adding again 1 because of reserved 0 index
    embedding_matrix_vocab = np.zeros((vocab_size, embedding_dim))
  
    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix_vocab[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix_vocab

#matrix for vocab: word_index
embedding_dim = 50
embedding_matrix_vocab = embedding_for_vocab('/content/glove.6B.50d.txt',tokenizer.word_index,embedding_dim)

"""Verify the vocabulary and embedding matrix."""

#Pintu 30/01/22

print("Type of vocabulary => ",type(tokenizer.word_index))
print("\nVocabulary glance => ",)
stop=0

#see the dict.
for word,index in tokenizer.word_index.items():
  stop+=1
  if(stop==10):
    break
  print(index," => ",word)
  
#dense vector.
print("\nDense vector of first word in dict => \n",embedding_matrix_vocab[1])

"""Create embedding for all tweet texts."""

#Pintu 30/01/22

import numpy as np

#tweet_text: [23, 0, 34, ..., 35] tweet converted into seq.
#goal: [[dense vector for 23], ..., [dense vecor for 35]]
def embedded_tweet_text(embedding_matrix_vocab, tweet_text):
    matrix_size = len(tweet_text)   
    
    tweet_text_matrix = np.zeros((matrix_size, embedding_dim))
    
    #traverse pad_seq of each tweet text
    for i in range(0,matrix_size):
      index=tweet_text[i]
      if(index==0):
        continue
      else:
        tweet_text_matrix[i]=embedding_matrix_vocab[index]
    return tweet_text_matrix

#define zero matrix.
n=len(X)
x1=np.zeros((n, 30, 50))

#populate matrix.
for i in range(0,n):
  x1[i]=embedded_tweet_text(embedding_matrix_vocab, X[i])

#ML model don't accept input in 3D shape
#so reshape 3D into 2D: 1D array per tweet.
x2=x1.reshape(len(X),-1)
print("Shape of x2 => ",x2.shape)

"""Verifying whole embedding."""

#x <- preprocessed
#x_train <- padded seq
#X <- smoted
#x1 <- final embedding

print("First tweet as text =>\n",dataset['Message'][1])
print("First tweet: text->seq =>\n",x_train[1])
print("Dense vector for 1st word of above tweet =>\n",embedding_matrix_vocab[22])
print("First tweet -> 2D matrix of dense vectors =>\n",x1[1])

"""Build Models: k-fold cross validation using accuracy as metric."""

#Pintu 30/01/22

#from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# X and y are being SMOTE

folds = range(3,10)

# define the model to be evaluate

# get a list of models to evaluate
def get_models():
	models = list()
	models.append(LogisticRegression())
	models.append(SGDClassifier())
	models.append(KNeighborsClassifier())
	models.append(DecisionTreeClassifier())
	models.append(SVC())
	models.append(GaussianNB())
	models.append(RandomForestClassifier())
	models.append(GradientBoostingClassifier())
	return models

models = get_models()
# evaluate each k value for a model
def evaluate_model(model):
  for k in folds:
    cross_validation = KFold(n_splits=k, shuffle=True, random_state=1)
    scores = cross_val_score(model, x2, y, scoring='accuracy', cv=cross_validation, n_jobs=-1)
    print("Average of the accuracy for ",k,"folds => ",mean(scores))

# evaluate each model
for m in models:
  model=m
  print("\nSummary of model:",model)
  evaluate_model(model)

print("Program ended!")