
#Importing all the libraries 

print("WORKING ON ML Models in object oriented way....")

import time
start_time = time.time()
# from numpy import asarray
# from numpy import loadtxt
# from numpy import savetxt
import pickle
from sklearn.model_selection import train_test_split
import sys, os, re, csv, codecs, numpy as np, pandas as pd, matplotlib.pyplot as plt 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.layers import Layer
from keras.layers import InputSpec
import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam, RMSprop
# from keras.optimizers import  RMSprop, adam_v2
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import GRU, BatchNormalization, Conv1D, MaxPooling1D
import warnings
warnings.filterwarnings("ignore")
import_time = round((time.time()-start_time)*1000,3)
np.random.seed(32)
# os.environ["OMP_NUM_THREADS"] = "4"


print(f"'\n\nALL LIBRARIES IMPORTED SUCCESSFULLY!!' in TIME : {import_time} msec")

class RocAucEvaluation(Callback):
    '''
     Customized Class to handle ROC AUC Evaluation for each epoch and printing epoch number and score for each epoch which is multiple of interval.  
    '''
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))

#################################################################################################################################################

class Modelbase :
    def __init__(self, embed_size = 300, max_features = 130000, max_len = 220, model_Type = "Spam-Detection", *args, **kwargs,):
        self.model_Type = model_Type
        self.embed_size = embed_size
        self.max_features = max_features #Vocabulary Size 
        self.max_len = max_len  # maximum length of tweet. 
        self.X_train = None
        self.Y_train = None
        self.X_valid = None
        self.Y_valid = None
        self.train = None
        self.y = None
        self.raw_text_train = None 
        self.raw_text_valid= None
        if self.model_Type == "Spam-Detection":
            self.list_classes = ["ham", "spam"]
            self.tweet_column = "Message"
            self.train_df1 = None
            self.train_df2 = None            
        else:
            self.list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
            self.tweet_column = "comment_text"
            self.test = None
            self.raw_test_valid= None        
            self.X_test =  None
            
        
    def read_dataset(self,filepath):
        '''
        Reading the training and testing data i.e converting that into dataframes
        input : file path 
        ----------------
        output : dataframe 
        '''
        df = pd.read_csv(filepath)
        return df 
    
    def build_train_test_dev_set(self, 
                                 trainDF1 = "../input/spam-or-not-spam-dataset/spam_or_not_spam.csv" , 
                                 trainDF2 = "../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv" , 
                                 trainDF  = "../input/jigsaw-toxic-comment-classification-challenge/train.csv" , 
                                 testDF   = "../input/jigsaw-toxic-comment-classification-challenge/test.csv"  ):
        
        if self.model_Type == "Spam-Detection"  :
            self.train_df1 = self.read_dataset(trainDF1)
            self.train_df2 = self.read_dataset(trainDF2)
            self.spam_Dataset_Aggregator()
            self.train = self.train.dropna()
            return 
        else:    
            self.train  = self.read_dataset(trainDF)
            self.test  = self.read_dataset(testDF)
#             self.train = self.train.dropna()
            return
        
    def spam_Dataset_Aggregator(self):
        self.train_df1.rename( columns={self.train_df1.columns[0] : "Message", self.train_df1.columns[1] : "Category" },inplace=True)
        self.train_df2[self.train_df2.columns[0]].replace({"ham": 0, "spam": 1}, inplace=True)
        self.train_df2 = self.train_df2[ [ self.train_df2.columns[1], self.train_df2.columns[0] ] ]
        self.train = pd.concat([self.train_df1, self.train_df2], ignore_index=True, sort=False)
        return  
        
    def __repr__(self):
        if self.model_Type == "Spam-Detection":
            return f"Model for Spam Detection (Ham/Spam detection) \n Model Type :  {self.model_Type}"
        return f"Model for Inappropraue Content Detection (Ham/Spam detection) \n Model Type :  {self.model_Type}" #Inappropriate-Content-Detection

    
    
#######################################################################################################################################################



class Preprocessor(Modelbase):
    
    def __init__(self,embed_size = 300, 
                 max_features = 130000, 
                 max_len = 220, 
                 model_Type = "Spam-Detection", 
                 *args, **kwargs):
#         super(Model, self).__init__()
        super().__init__(embed_size , max_features, max_len, model_Type)
        # num_words = the maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept.
        self.tk = Tokenizer(num_words = max_features)
        self.embedding_matrix_fastText = None
        self.embedding_matrix_glove = None
        self.embedding_index_fastText = None
        self.embedding_index_glove = None
        self.embedding_matrix_fastText_shape = None
        self.embedding_matrix_glove_shape = None
#         self.embedding_index_fastText_shape = None
#         self.embedding_index_glove_shape = None
        
    def train_test_splitter(self,validation_size=0.1):
        '''
        Input : training dataset Dataframe -> train
        ------------
        Output : Training and dev dataframe with X and Y values along with raw train and dev set with all comments in lower.
        '''
        if self.model_Type != "Spam-Detection" :
            # y = np.asarray(y).astype(np.float32)
            self.y = self.train[self.list_classes].values
            self.train[self.tweet_column].fillna("no comment")
            self.test[self.tweet_column].fillna("no comment") 
#             self.train = self.train.dropna()  #**********************************************************
        else:
            self.y = self.train["Category"].values
            # y = np.asarray(y).astype(np.float32)
            self.train[self.tweet_column].fillna("no comment")
        # train_df2["Message"].fillna("no comment")
        validation_size = validation_size   #by default it is 0.1
        # splitting train:dev -->  9:1 so train consist of 90 percent and validation set consist of 10 percent of entire training data 
        #Note : This validation_size is configurable and can be changed later.
        self.X_train, self.X_valid, self.Y_train, self.Y_valid = train_test_split(self.train, self.y, test_size = validation_size, random_state=512)
        # Lowering all the comments of training, validation and test data let callled it as raw.
        self.raw_text_train = self.X_train[self.tweet_column].str.lower()
        self.raw_text_valid = self.X_valid[self.tweet_column].str.lower()
        if self.model_Type != "Spam-Detection" :
            self.raw_test_valid = self.test[self.tweet_column].str.lower()
        return

    def tokenizer(self):
            '''
            Input : raw train and dev set along with in-rawed train and dev set.
            --------------------------------------------------------------------------------
            Output: tokenizer object along with padded and sequenced training and validation dataset
            '''
            # Article for better undrstanding of Tokenizer : https://machinelearningknowledge.ai/keras-tokenizer-tutorial-with-examples-for-fit_on_texts-texts_to_sequences-texts_to_matrix-sequences_to_matrix/
            #Tokeinzing the raw training set
            self.tk.fit_on_texts(self.raw_text_train)
            print(self.raw_text_train.shape )
            self.X_train[self.tweet_column] = self.tk.texts_to_sequences(self.raw_text_train)
            self.X_valid[self.tweet_column] = self.tk.texts_to_sequences(self.raw_text_valid)
            self.X_train = pad_sequences(self.X_train[self.tweet_column], maxlen = self.max_len)
            self.X_valid = pad_sequences(self.X_valid[self.tweet_column], maxlen = self.max_len)
            if self.model_Type != "Spam-Detection" :
                self.test[self.tweet_column] = self.tk.texts_to_sequences(self.raw_test_valid )
                self.test = pad_sequences(self.test[self.tweet_column], maxlen = self.max_len) #test.comment_seq
            return
    
    def get_coefs(self,word,*arr): 
        '''
        Creating Embeeding Index which can help further to create embedding matrix for the words in our training dataset vocabulary.
        This Ebedding index is created from the fastText or Glove.
        '''
        return word, np.asarray(arr, dtype='float32')
    
    
    def embeeding_Index_Builder(self, embedding_path):
        '''
        Embedding Index correpsonding to embeddding Path
        Input : embeddig path 
        ----------------------
        Output : embedding Index
        '''
        embedding_index = dict(self.get_coefs(*o.strip().split(" ")) for o in open(embedding_path))
        return embedding_index

    def embeeding_Matrix_Builder(self,embedding_index ):
        '''
        Input : tokenizer object , with maximum features and embedding size along with fastText and Glove Embedding Index for building Embeeding Matrix
        ----------------------------------------------------------------------------------------------------------------------------------------------
        Output: embedding matix corresponding to FastText and Glove
        '''
        # Preparing Our Embeeding matrix from Embedding Index from glove or fastText or any other index. 
        damword_index = self.tk.word_index
        nb_words = min(self.max_features, len(damword_index))
        embedding_matrix = np.zeros((nb_words+1, self.embed_size))
        for word, i in damword_index.items():
            if i >= self.max_features: continue
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        return embedding_matrix
    
    def save_embeeding_index_or_matrix_as_csv(self, embedding_index_or_matrix, arrayType="index", embeddingType="FastText"):
        # save numpy array embeeding_index as csv file
        # define data
        if arrayType!="index":
            data = copy.deepcopy(embedding_index_or_matrix)
            # save to csv file
            data = data.ravel()
            np.savetxt(f'embedding_{arrayType}_{embeddingType}.csv', data, delimiter=',')
        else:
            a_file = open(f'embedding_{arrayType}_{embeddingType}.pkl', "wb")
            pickle.dump(embedding_index_or_matrix, a_file)
            a_file.close() 
        
    def load_embedding_index_or_matrix_from_csv(self, arrayType="index", embeddingType="FastText"):
        # load numpy array  embeeding_index from csv file
        # load array
        if arrayType!="index":
            data = np.loadtxt(f'embedding_{arrayType}_{embeddingType}.csv', delimiter=',')
            reshaped_data = None

            if embeddingType=="FastText" :
#                 if arrayType=="index":
                reshaped_data = np.reshape(data,self.embedding_index_fastText_shape )
#                 else:
#                     reshaped_data = np.reshape(data,self.embedding_matrix_fastText_shape )
            elif embeddingType=="Glove":
#                 if arrayType=="index":
                reshaped_data = np.reshape(data,self.embedding_index_glove_shape )
#                 else:
#                     reshaped_data = np.reshape(data,self.embedding_matrix_glove_shape )
            # print the array
            return reshaped_data
        else:
            a_file = open(f'embedding_{arrayType}_{embeddingType}.pkl', "rb")
            output = pickle.load(a_file)
            return output

            
    
    def save_embeeding_index_or_matrix_as_binary(self, embedding_index_or_matrix , arrayType="index", embeddingType="FastText"):
        # save numpy array embeeding_index as binary file npy
        # define data
        if arrayType!="index":
            np.save(f'embedding_{arrayType}_{embeddingType}.npy', embedding_index_or_matrix )
            return
        else:
            a_file = open(f'embedding_{arrayType}_{embeddingType}.pkl', "wb")
            pickle.dump(embedding_index_or_matrix, a_file)
            a_file.close() 
        
    def load_embedding_index_or_matrix_from_binary(self, arrayType="index", embeddingType="FastText"):
        # load numpy array  embeeding_index from npy file
        # load array
        if arrayType!="index":
            data = np.load(f'embedding_{arrayType}_{embeddingType}.csv')
            return data
        else:
            a_file = open(f'embedding_{arrayType}_{embeddingType}.pkl', "rb")
            output = pickle.load(a_file)
            return output


    def preprocessing(self, 
                      validation_size=0.1,
                      embedding_path_fastText = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec",
                      embedding_path_glove ="../input/glove840b300dtxt/glove.840B.300d.txt",
                      isEmbeddingIndexFileSaved     =False,
                      isEmbeddingMatrixFileSaved    = False,
                      wantToSaveEmbeedingIndexFile  = False,
                      wantToSaveEmbeedingMatrixFile = False ):
        self.build_train_test_dev_set()
        print('Building Trainning and Testing is Done.')
        self.train_test_splitter(validation_size)
        print('Spliting Trainning and Testing is Done.')
        self.tokenizer()
        print('Tokenizing is Done.')
        if not isEmbeddingIndexFileSaved :
            self.embedding_index_fastText = self.embeeding_Index_Builder(embedding_path_fastText)
            print('Embeeding Index for Fasttext is Done.')
            self.embedding_index_glove = self.embeeding_Index_Builder(embedding_path_glove)
            print('Embeeding Index for Glove is Done.')
            if wantToSaveEmbeedingIndexFile:
                self.save_embeeding_index_or_matrix_as_binary(self.embedding_index_fastText  , arrayType="index", embeddingType="FastText")
                self.save_embeeding_index_or_matrix_as_binary(self.embedding_index_glove  , arrayType="index", embeddingType="Glove")
        else:
            self.embedding_index_fastText  = self.load_embedding_index_or_matrix_from_binary(arrayType="index", embeddingType="FastText")
            self.embedding_index_glove  = self.load_embedding_index_or_matrix_from_binary(arrayType="index", embeddingType="Glove")
#         self.embedding_index_fastText_shape = self.embedding_index_fastText.shape
#         self.embedding_index_glove_shape = self.embedding_index_glove.shape
        
        if not isEmbeddingMatrixFileSaved :
            self.embedding_matrix_fastText = self.embeeding_Matrix_Builder( self.embedding_index_fastText )
            print(f'Embedding Matrix for FastText is Done., with it\'s shape as {self.embedding_matrix_fastText_shape}')
            self.embedding_matrix_glove = self.embeeding_Matrix_Builder( self.embedding_index_glove )
            print(f'Embedding Matrix for Glove is Done with shape as { self.embedding_matrix_glove_shape}')
            if  wantToSaveEmbeedingMatrixFile:
                self.save_embeeding_index_or_matrix_as_binary(self.embedding_matrix_fastText , arrayType="matrix", embeddingType="FastText")
                self.save_embeeding_index_or_matrix_as_binary(self.embedding_matrix_glove   , arrayType="matrix", embeddingType="Glove")
        else:
            self.embedding_matrix_fastText  = self.load_embedding_index_or_matrix_from_binary(arrayType="matrix", embeddingType="FastText")
            self.embedding_matrix_glove     = self.load_embedding_index_or_matrix_from_binary(arrayType="matrix", embeddingType="Glove")
#         self.embedding_matrix_fastText_shape = self.embedding_matrix_fastText.shape
#         self.embedding_matrix_glove_shape =  self.embedding_matrix_glove.shape 
        return
    
#######################################################################################################################################################



if __name__ == "__main__":
    print("Hello Preprocessor File is Running.")

