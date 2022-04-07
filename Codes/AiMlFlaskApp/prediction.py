#Importing required libraries 
import time
start_time = time.time()
# from numpy import asarray
# from numpy import loadtxt
# from numpy import savetxt
import pickle
from sklearn.model_selection import train_test_split
import sys, os, re, csv, codecs, numpy as np, pandas as pd, matplotlib.pyplot as plt 
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
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
from preprocessor import RocAucEvaluation, Modelbase , Preprocessor
from train import ML_Model
end = time.time()
import_time = round((end-start_time)*1000,3)
np.random.seed(32)
# os.environ["OMP_NUM_THREADS"] = "4"


print(f"'\n\nALL LIBRARIES IMPORTED SUCCESSFULLY!!' in TIME : {import_time} msec")


class PredictorObject:
    def __init__(self, max_len=220, 
                     max_features = 130000,
                     model_Type = "Spam-Detection",
                     file_path = "1",
                    n_multiClassificationClasses = 1,# if Inappropriate then 6 as they are : "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
                    tweet_column = None,
                 actual_model_path= None,
                 training_dataset_path = "../input/jigsaw-toxic-comment-classification-challenge/train.csv",
                     *args, **kwargs):
        self.start_time = time.time()        
        self.max_features = max_features
        self.max_len = max_len
        self.n_multiClassificationClasses  = n_multiClassificationClasses         
        self.model_instanc_type = model_Type 
        self.framework_idea = file_path 
        
        self.file_path = actual_model_path if actual_model_path else "best_model_" + "FrameWork" + self.framework_idea + "_" + self.model_instanc_type  + ".hdf5"
        self.result_filename = "RESULTS.csv"
        
        #check whether model is already prsent else load the model
        self.model = load_model(self.file_path)
        
        self.tweet = None
        self.tweet_array = None
        self.tweet_df = None
        self.original_tweet_df = None
        self.raw_tweet_df = None
        self.tweet_column,  self.list_classes = ("Email-Text", ["ham-or-spam"] ) if self.model_instanc_type == "Spam-Detection" else ("Tweet-Comment" , ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"] )  
        if tweet_column :
            self.tweet_column = tweet_column
        self.result = pd.DataFrame(columns = [self.tweet_column ] + self.list_classes) 
        
        self.tk = Tokenizer(num_words = self.max_features, lower = True)
        train = pd.read_csv(training_dataset_path)
        if model_Type != "Spam-Detection":
            train["comment_text"].fillna("no comment")
            X_train, X_valid, Y_train, Y_valid = train_test_split( train, train[self.list_classes].values, test_size = 0.3, random_state=512)
            ##Keeping hyperparaeter test_size as 0.3 
            raw_text_train = X_train["comment_text"].str.lower()
            self.tk.fit_on_texts(raw_text_train)
        else:
            pass


    def initializeAll(self, actual_model_path=None,  tweet_column=None):
        self.tweet = None
        self.tweet_array = None
        self.tweet_df = None
        self.raw_tweet_df = None
        self.original_tweet_df = None
        self.tweet_column,  self.list_classes = ("Email-Text", ["ham-or-spam"] ) if self.model_instanc_type == "Spam-Detection" else ("Tweet-Comment" , ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"] )            
        if tweet_column :
            self.tweet_column = tweet_column
        self.result = pd.DataFrame(columns = [self.tweet_column ] + self.list_classes) 
        self.result_filename = "RESULTS.csv"
        self.file_path = actual_model_path if actual_model_path else "best_model_" + "FrameWork" + self.framework_idea + "_" + self.model_instanc_type  + ".hdf5"
        if actual_model_path:
            self.model = load_model(self.file_path)
            
        
        
    def reLoadModel(self):
        #check whether model is already prsent else load the model
        self.model = load_model(self.file_path)
        return
        
    def prediction(self, tweet_data=None, isText=True, isArray=False, isDataFrame=False, path_to_Tweet_DataFrame=None, result_filename=None, actual_model_path=None,  tweet_column=None ):
        def exor(a, b): return ( a&(~b) or b&(~a) )        
        if ( not exor( exor(isText, isArray), isDataFrame ) ) or ( isText == isArray and isText==isDataFrame ) :
            print('404: all three or any two can not be true only one paramenters among isTweet, isDataFrame, isArray will be only true' )
            return
        self.initializeAll(actual_model_path,  tweet_column)
        if isText:
            self.tweet = tweet_data
            self.tweet_array = [ tweet_data]
            y_pred = self.preprocessDataFrame()
        elif isArray:
            self.tweet = None
            self.tweet_array = tweet_data
            y_pred = self.preprocessDataFrame()
        else:
            if path_to_Tweet_DataFrame :
                self.tweet_df =  pd.read_csv(path_to_Tweet_DataFrame)
                y_pred = self.preprocessDataFrame( override=True )
            else:
                print('ERROR : 404 FILE NOT FOUND!!! Provide valid path name.')
                return
        if not result_filename:
            self.result_filename = result_filename
        self.result[self.tweet_column] = self.original_tweet_df[self.tweet_column]
#         self.result[self.tweet_column] = self.tweet_df.iloc[:,0]
        self.result[self.list_classes] =  (y_pred)
        self.result.to_csv(self.result_filename, index = False )
        print(f"[{ time.time() - self.start_time }] Completed!")
        return (self.result, self.tweet_column, self.list_classes, self.result_filename) 

    
    def preprocessDataFrame(self,override=False):
        if not override:
            print('Inside')
            self.tweet_df = pd.DataFrame( self.tweet_array, columns = [ self.tweet_column ])
#         print(f"\n 1 , {self.tweet_df}")
        self.original_tweet_df = self.tweet_df.copy(deep=True)
        self.tweet_df[self.tweet_column].fillna("no comment")
#         print(f"\n\n 2 , {self.tweet_df}")
        self.raw_tweet_df = self.tweet_df[self.tweet_column].str.lower()
#         print(f"\n\n 3 , {self.tweet_df}")
#         self.tk.fit_on_texts(self.raw_tweet_df)
        self.tweet_df[ self.tweet_column ] = self.tk.texts_to_sequences( self.raw_tweet_df )
#         print(f"\n\n 4 , {self.tweet_df}")
        self.tweet_df = pad_sequences( self.tweet_df[ self.tweet_column ], maxlen = self.max_len )
#         print(f"\n\n 5 , {self.tweet_df}")
        print("Predicting")
        y_pred = self.model.predict(self.tweet_df, verbose=0)
        return y_pred

















if __name__ == "__main__":
    print("Prediction File is Running..")

    #Inappropriate Content Detection Prediction

    # %%time
    start = time.time()
    df = pd.read_csv("./dataset/test.csv")
    tweet = df["comment_text"][0]

    # actual_model_path_for_inapp = '../input/model-for-spam-inappropriate-content-detection1/best_model_FrameWork1_Inappropriate-Content-Detection.hdf5'
    # actual_model_path_for_inapp = './best_model_FrameWork1_Inappropriate-Content-Detection.hdf5'
    # actual_model_path_for_inapp = '../input/inappcontentepoch50model/best_model_FrameWork1_Inappropriate-Content-Detection.hdf5'
    actual_model_path_for_inapp = './Trained_model/best_model_FrameWork1_Inappropriate-Content-Detection.hdf5'
    InappPrediction = PredictorObject(model_Type = "Inappropriate-Content-Detection", 
                                      file_path = "1" ,
                                      n_multiClassificationClasses = 6, 
                                      tweet_column='comment_text'  ,
                                     actual_model_path = actual_model_path_for_inapp,
                                     training_dataset_path = "./dataset/train.csv" 
                                     )


    end = time.time()
    print(f"Time taken to construct Predictor Object : {end-start}")

    test_data_path = './dataset/test.csv'
    result, tweet_column, list_classes, result_filename = InappPrediction.prediction( tweet,isText=True, isArray=False, isDataFrame=False, tweet_column='comment_text'  )

    print(InappPrediction.tweet_array)

    print(InappPrediction.raw_tweet_df[0] )


    print(InappPrediction.raw_tweet_df)


    print(InappPrediction.result.head())


    # %%time
    start = time.time()
    tweet ='you will cry whole life and i will cut you in pieces. Beacause i am monster. Be ready, i am coming for you and when i will find you , i will wash my hand with your blood'
    result, tweet_column, list_classes, result_filename = InappPrediction.prediction( tweet,isText=True, isArray=False, isDataFrame=False, tweet_column='comment_text'  )
    print("\n\n",InappPrediction.tweet_array,"\n\n")
    print(InappPrediction.result.head())
    end = time.time()
    print("Time Taken to predict : ",end - start)



############################## ABOVE SNIPPET is ENOUGH For Prediction of Inappropriate Content Detection ##########################################
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################







    # # %%time
    # start = time.time()
    # # actual_model_path_for_inapp = '../input/model-for-spam-inappropriate-content-detection1/best_model_FrameWork1_Inappropriate-Content-Detection.hdf5'
    # actual_model_path_for_inapp = './best_model_FrameWork1_Inappropriate-Content-Detection.hdf5'
    # InappPrediction = PredictorObject(model_Type = "Inappropriate-Content-Detection", 
    #                                   file_path = "1" ,n_multiClassificationClasses = 6, 
    #                                   tweet_column='comment_text'  ,
    #                                  actual_model_path = actual_model_path_for_inapp)


    # end = time.time()
    # print(f"Time taken to construct Predictor Object : {end-start}")

    # #Checking whether model is loaded correctly or not 
    # from sklearn.metrics import roc_auc_score
    # def predictions(dataset, y_actual,model,on_the_fly=0 ): # On the fly is for those data that is comimg on the fly from any tweet.
    #     y_pred_validation = model.predict(dataset, verbose=1)
    #     score_validation = roc_auc_score(y_actual, y_pred_validation)
    #     print(f"\n ROC-AUC - ON given Dataset - score: {round(score_validation,5)*100}%")
    #     return y_pred_validation



    # ########################################################################
    # print(predictions(InappContentModel1.modelBase.X_train,InappContentModel1.modelBase.Y_train,InappPrediction.model))



    # ########################################################################
    # print(predictions(InappContentModel1.modelBase.X_valid,InappContentModel1.modelBase.Y_valid,InappPrediction.model))





    # from sklearn.model_selection import train_test_split
    # # train_data_path = '../input/jigsaw-toxic-comment-classification-challenge/train.csv'
    # df1 = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
    # df1["comment_text"].fillna("no comment")
    # X_train, X_valid, Y_train, Y_valid = train_test_split(df1, df1[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values, test_size = 0.1)

    # # raw_text_train = X_train["comment_text"].str.lower()
    # df2 = X_train['comment_text'].tolist()
    # df3 =X_valid['comment_text'].tolist()
    # # for i in range(len(X_train['comment_text']))
    # #     df2.append(X_train['comment_text'][i])
    # # for i in range(len(X_valid['comment_text'])):
    # #     df3.append(X_valid['comment_text'][i])



    # print(InappContentModel1.modelBase.X_train)



    # print(X_train['comment_text'][0])

    # print(len(X_train['comment_text']))


    # result2, tweet_column2, list_classes2, result_filename2   = InappPrediction.prediction( df2,isText=False, isArray=True, isDataFrame=False, path_to_Tweet_DataFrame=None,tweet_column='comment_text'  )



    # print(InappPrediction.result.head(12))



    # print(X_train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].head(12))


    # print(X_train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].shape)



    # print(InappPrediction.result[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].shape)

    # print(X_train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].shape == InappPrediction.result[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].shape)



    # from sklearn.metrics import roc_auc_score
    # def predictions(dataset, y_actual,model,on_the_fly=0 ): # On the fly is for those data that is comimg on the fly from any tweet.
    #     y_pred_validation = model.predict(dataset,  batch_size = 1024, verbose = 1)
    #     score_validation = roc_auc_score(y_actual, y_pred_validation)
    #     print(f"\n ROC-AUC - ON given Dataset - score: {round(score_validation,5)*100}%")
    #     return y_pred_validation


    # # predictions( InappPrediction.tweet_df, Y_train, InappPrediction.model)
    # print(predictions( InappPrediction.tweet_df, Y_train, InappContentModel1.model))



    # print(predictions( InappContentModel1.modelBase.X_train,InappContentModel1.modelBase.Y_train, InappContentModel1.model))


    # result3, tweet_column3, list_classes3, result_filename3   = InappPrediction.prediction( df3,isText=False, isArray=True, isDataFrame=False, path_to_Tweet_DataFrame=None,tweet_column='comment_text'  )

    # print(predictions( InappPrediction.tweet_df, Y_valid, InappPrediction.model))



    # #############################################################################################################################################
    # #############################################################################################################################################
    # #############################################################################################################################################
    # #############################################################################################################################################
    # #############################################################################################################################################

    # ##Spam Detection Prediction


    # # %%time
    # start = time.time()
    # df = pd.read_csv("../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")
    # tweet = df[df.columns[0]][0]

    # # actual_model_path_for_inapp = '../input/model-for-spam-inappropriate-content-detection1/best_model_FrameWork1_Inappropriate-Content-Detection.hdf5'
    # # actual_model_path_for_inapp = './best_model_FrameWork1_Inappropriate-Content-Detection.hdf5'
    # actual_model_path_for_inapp = '../input/inappcontentepoch50model/best_model_FrameWork1_Inappropriate-Content-Detection.hdf5'
    # InappPrediction = PredictorObject(model_Type = "Inappropriate-Content-Detection", 
    #                                   file_path = "1" ,n_multiClassificationClasses = 6, 
    #                                   tweet_column='comment_text'  ,
    #                                  actual_model_path = actual_model_path_for_inapp)



    # test_data_path = '../input/jigsaw-toxic-comment-classification-challenge/test.csv'
    # result, tweet_column, list_classes, result_filename = InappPrediction.prediction( tweet,isText=True, isArray=False, isDataFrame=False, tweet_column='comment_text'  )
    # end = time.time()
    # print("Time Taken to predict and also to construct Predictor Object : ", end-start)