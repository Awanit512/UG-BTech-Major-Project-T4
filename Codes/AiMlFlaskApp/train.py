

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
from preprocessor import RocAucEvaluation, Modelbase , Preprocessor
import warnings
warnings.filterwarnings("ignore")
import_time = round((time.time()-start_time)*1000,3)
np.random.seed(32)
# os.environ["OMP_NUM_THREADS"] = "4"


print(f"'\n\nALL LIBRARIES IMPORTED SUCCESSFULLY!!' in TIME : {import_time} msec")




class ML_Model :
    def __init__(self, 
                 ModelBaseInstance = Preprocessor( embed_size = 300, 
                                           max_features = 130000, 
                                           max_len = 220, 
                                           model_Type = "Spam-Detection"),
#                  embed_size = 300, 
#                  max_features = 130000, 
#                  max_len = 220, 
#                  model_Type = "Spam-Detection", 
                 file_path = "1",
                 n_multiClassificationClasses = 1 ,# if Inappropriate then 6 as they are : "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
                 *args, **kwargs):  # for file_path just pass which idea/framework on which this Model is based upon
#         super(Preprocessor, self).__init__()
#         super().__init__( embed_size = 300, 
#                           max_features = 130000, 
#                           max_len = 220, 
#                           model_Type = "Spam-Detection")
        self.modelBase = ModelBaseInstance
        self.model = None
        self.history = None
        self.framework_idea = file_path 
        self.file_path = "best_model_" + "FrameWork" + self.framework_idea + "_" + self.modelBase.model_Type  + ".hdf5" 
        self.check_point = ModelCheckpoint(self.file_path, monitor = "val_loss", verbose = 1, save_best_only = True, mode = "min")
        self.ra_val = None
        #Allowing early stopping
        self.early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)
        self.n_output_nuerons = n_multiClassificationClasses
        
    def change_idea_for_same_object(self,change_idea_to="2", automatic_train = False, *args, **kwargs):
        earlier_idea = self.framework_idea 
        self.framework_idea = change_idea_to
        self.file_path = "best_model_" + "FrameWork" + self.framework_idea + "_" + self.modelBase.model_Type + ".hdf5" 
        self.check_point = ModelCheckpoint(self.file_path, monitor = "val_loss", verbose = 1, save_best_only = True, mode = "min")
        print(f"Framework is changeed succesfully \n Idea/Frameowkr earlier :  {earlier_idea} \n Idea/Framework Now {self.framework_idea } : ")
        print("\n\n PS: This method will keep in handy when you want to test for multiple idea without doing preprocessing step again.")
        # TO-DO  allo training auto to be done for user given parameters
        if automatic_train:
            print(f"Note: The training is requested to be done AUTOMATICALLY as automatic_train flag is {automatic_train}, Note:  This training will be done on default parameters \n i.e lr = 0.0, \n lr_d = 0.0, \n units = 0, \n dr = 0.0, \n epochs=10.")
            print('OR The parameters which was set for earlier IDEA/FRAMEWORK')
            self.train_Model()
        else:
            print("\t Note : WE NEED TO RE-TRAINED THE MODEL.\n\n \t For Help :: USE METHOD --> train_Model for this in order to train the model on new Idea/Framework. \n \t Not training automatically as Automatic train Flag is off.")
        return
    
    def roc_Auc_Valuation(self):
        self.ra_val = RocAucEvaluation(validation_data=(self.modelBase.X_valid, self.modelBase.Y_valid), interval = 1)
        return
        
    def build_model_framework1(self, lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0, epochs=10):
        inp = Input(shape = (self.modelBase.max_len,))
#         x_fastText = Embedding( self.max_features, self.embed_size, weights = [self.embedding_matrix_fastText], trainable = False)(inp)# Let this layer be the FatText input embedding
#         x_glove = Embedding( self.max_features, self.embed_size, weights = [self.embedding_matrix_glove], trainable = False)(inp)# Let this layer be the Glove input embedding
        x_fastText = Embedding(self.modelBase.embedding_matrix_fastText.shape[0], self.modelBase.embed_size, weights = [self.modelBase.embedding_matrix_fastText], trainable = False)(inp)# Let this layer be the FatText input embedding
        x_glove = Embedding(self.modelBase.embedding_matrix_glove.shape[0], self.modelBase.embed_size, weights = [self.modelBase.embedding_matrix_glove], trainable = False)(inp)# Let this layer be the Glove input embedding
        
    
    #Drop-Out Layer
        x1_fastText = SpatialDropout1D(dr)(x_fastText)
        x1_glove = SpatialDropout1D(dr)(x_glove)
        #BI-LSTM BRANCH
        x_glove_lstm = Bidirectional(GRU(units, return_sequences = True))(x1_glove)
        x_fastText_lstm = Bidirectional(GRU(units, return_sequences = True))(x1_fastText)
        #BI-GRU BRNACH
        x_glove_gru = Bidirectional(GRU(units, return_sequences = True))(x1_glove)
        x_fastText_gru = Bidirectional(GRU(units, return_sequences = True))(x1_fastText)
        #Convolutional+Pooling Layer This is Optional Can be commnented later if required for testing purposes
        #Bi-listm
        x_glove_lstm_conv           = Conv1D(int(units/2), kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x_glove_lstm)
        x_glove_lstm_conv_avgPool   = GlobalAveragePooling1D()(x_glove_lstm_conv)
        x_glove_lstm_conv_maxPool   = GlobalMaxPooling1D()(x_glove_lstm_conv)
        #concatenating
        x_glove_lstm_conv_pooled    = concatenate([x_glove_lstm_conv_avgPool, x_glove_lstm_conv_maxPool])

        x_fastText_lstm_conv        = Conv1D(int(units/2), kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x_fastText_lstm)
        x_fastText_lstm_conv_avgPool= GlobalAveragePooling1D()(x_fastText_lstm_conv)
        x_fastText_lstm_conv_maxPool= GlobalMaxPooling1D()(x_fastText_lstm_conv)
        #concatenating
        x_fastText_lstm_conv_pooled = concatenate([x_fastText_lstm_conv_avgPool, x_fastText_lstm_conv_maxPool])
        #Bi-gru
        x_glove_gru_conv            = Conv1D(int(units/2), kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x_glove_gru)
        x_glove_gru_conv_avgPool    = GlobalAveragePooling1D()(x_glove_gru_conv)
        x_glove_gru_conv_maxPool    = GlobalMaxPooling1D()(x_glove_gru_conv)
        #concatenating
        x_glove_gru_conv_pooled     = concatenate([x_glove_gru_conv_avgPool, x_glove_gru_conv_maxPool])
        x_fastText_gru_conv         = Conv1D(int(units/2), kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x_fastText_gru)
        x_fastText_gru_conv_avgPool = GlobalAveragePooling1D()(x_fastText_gru_conv)
        x_fastText_gru_conv_maxPool = GlobalMaxPooling1D()(x_fastText_gru_conv)
        #concatenating
        x_fastText_gru_conv_pooled  = concatenate([x_fastText_gru_conv_avgPool, x_fastText_gru_conv_maxPool])
        #Con-Catenating Bi-Lstm branch
        x_lstm_conv_pooled = concatenate([x_glove_lstm_conv_pooled,  x_fastText_lstm_conv_pooled])
        #Con-Catenating Bi-Gru branch
        x_gru_conv_pooled = concatenate([x_glove_gru_conv_pooled,  x_fastText_gru_conv_pooled])
        #ConCatenating Bi-LSTM and Bi-GRU Branch
        x_lstm_gru_concatenated = concatenate([x_lstm_conv_pooled, x_gru_conv_pooled])
        #Passing concatenated output to dense network
    #     xxxx = Dense(8, activation = "sigmoid")(x_lstm_gru_concatenated)
    #     xxx = Dense(4, activation = "sigmoid")(xxxx)
    #     xx = Dense(2, activation = "sigmoid")(xx)
    #     x = Dense(1, activation = "sigmoid")(xx) 
        
#         if self.model_Type != "Spam-Detection" :
#             assert self.n_output_nuerons==6
#         else:
#             assert self.n_output_nuerons==1

        print(self.n_output_nuerons)
            
        x = Dense(self.n_output_nuerons, activation = "sigmoid")(x_lstm_gru_concatenated)
        self.model = Model(inputs = inp, outputs = x)
        self.model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    #     model.compile(loss = "binary_crossentropy", optimizer = adam_v2.Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    
#         print("*************************************************")
#         print(f"Y_train's size/shape  ={self.modelBase.Y_train.shape} ")
#         print(f"Y_valid's size/shape  ={self.modelBase.Y_valid.shape} ")  
        
        self.history = self.model.fit(self.modelBase.X_train, self.modelBase.Y_train, batch_size = 128, epochs = epochs, validation_data = (self.modelBase.X_valid, self.modelBase.Y_valid), 
                            verbose = 1, callbacks = [self.ra_val, self.check_point, self.early_stop])
        self.model = load_model(self.file_path)
        return
        
    def build_model_framework2(self,lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0, epochs=10):
        inp = Input(shape = (self.modelBase.max_len,))
#         x_fastText = Embedding( self.max_features, self.embed_size, weights = [self.embedding_matrix_fastText], trainable = False)(inp)# Let this layer be the FatText input embedding
#         x_glove = Embedding( self.max_features, self.embed_size, weights = [self.embedding_matrix_glove], trainable = False)(inp)# Let this layer be the Glove input embedding

        x_fastText = Embedding(self.modelBase.embedding_matrix_fastText.shape[0], self.modelBase.embed_size, weights = [self.modelBase.embedding_matrix_fastText], trainable = False)(inp)# Let this layer be the FatText input embedding
        x_glove = Embedding(self.modelBase.embedding_matrix_glove.shape[0], self.modelBase.embed_size, weights = [self.modelBase.embedding_matrix_glove], trainable = False)(inp)# Let this layer be the Glove input embedding
       
    #Drop-Out Layer
        x1_fastText = SpatialDropout1D(dr)(x_fastText)
        x1_glove = SpatialDropout1D(dr)(x_glove)
        #BI-LSTM BRANCH
        x_glove_lstm = Bidirectional(GRU(units, return_sequences = True))(x1_glove)
        x_fastText_lstm = Bidirectional(GRU(units, return_sequences = True))(x1_fastText)
        #BI-GRU BRNACH
        x_glove_gru = Bidirectional(GRU(units, return_sequences = True))(x1_glove)
        x_fastText_gru = Bidirectional(GRU(units, return_sequences = True))(x1_fastText)
        #Convolutional+Pooling Layer This is Optional Can be commnented later if required for testing purposes
        #Bi-listm
        x_glove_lstm_conv           = Conv1D(int(units/2), kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x_glove_lstm)
        x_glove_lstm_conv_avgPool   = GlobalAveragePooling1D()(x_glove_lstm_conv)
        x_glove_lstm_conv_maxPool   = GlobalMaxPooling1D()(x_glove_lstm_conv)
        #concatenating
        x_glove_lstm_conv_pooled    = concatenate([x_glove_lstm_conv_avgPool, x_glove_lstm_conv_maxPool])

        x_fastText_lstm_conv        = Conv1D(int(units/2), kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x_fastText_lstm)
        x_fastText_lstm_conv_avgPool= GlobalAveragePooling1D()(x_fastText_lstm_conv)
        x_fastText_lstm_conv_maxPool= GlobalMaxPooling1D()(x_fastText_lstm_conv)
        #concatenating
        x_fastText_lstm_conv_pooled = concatenate([x_fastText_lstm_conv_avgPool, x_fastText_lstm_conv_maxPool])
        #Bi-gru
        x_glove_gru_conv            = Conv1D(int(units/2), kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x_glove_gru)
        x_glove_gru_conv_avgPool    = GlobalAveragePooling1D()(x_glove_gru_conv)
        x_glove_gru_conv_maxPool    = GlobalMaxPooling1D()(x_glove_gru_conv)
        #concatenating
        x_glove_gru_conv_pooled     = concatenate([x_glove_gru_conv_avgPool, x_glove_gru_conv_maxPool])

        x_fastText_gru_conv         = Conv1D(int(units/2), kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x_fastText_gru)
        x_fastText_gru_conv_avgPool = GlobalAveragePooling1D()(x_fastText_gru_conv)
        x_fastText_gru_conv_maxPool = GlobalMaxPooling1D()(x_fastText_gru_conv)
        #concatenating
        x_fastText_gru_conv_pooled  = concatenate([x_fastText_gru_conv_avgPool, x_fastText_gru_conv_maxPool])
        #Con-Catenating Bi-Lstm branch
        x_lstm_conv_pooled = concatenate([x_glove_lstm_conv_pooled,  x_fastText_lstm_conv_pooled])
        #Con-Catenating Bi-Gru branch
        x_gru_conv_pooled = concatenate([x_glove_gru_conv_pooled,  x_fastText_gru_conv_pooled])
        #ConCatenating Bi-LSTM and Bi-GRU Branch
        x_lstm_gru_concatenated = concatenate([x_lstm_conv_pooled, x_gru_conv_pooled])
        #Passing concatenated output to dense network
    #     xxxx = Dense(8, activation = "sigmoid")(x_lstm_gru_concatenated)
    #     xxx = Dense(4, activation = "sigmoid")(xxxx)
    #     xx = Dense(2, activation = "sigmoid")(xxx)
    #     x = Dense(1, activation = "sigmoid")(xx)
    
        print(self.n_output_nuerons)

        x = Dense(self.n_output_nuerons, activation = "sigmoid")(x_lstm_gru_concatenated)
        self.model = Model(inputs = inp, outputs = x)
        self.model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    #     model.compile(loss = "binary_crossentropy", optimizer = adam_v2.Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
        self.history = self.model.fit(self.modelBase.X_train, self.modelBase.Y_train, batch_size = 128, epochs = epochs, validation_data = (self.modelBase.X_valid, self.modelBase.Y_valid), 
                            verbose = 1, callbacks = [self.ra_val, self.check_point, self.early_stop])
        self.model = load_model(self.file_path)
        return 
    
    def build_model_framework3(self,lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0, epochs=10):
        pass
    
    def build_model(self, lr = 1e-3, lr_d = 0, units = 112, dr = 0.2,epochs = 30):
        self.roc_Auc_Valuation()
        if self.framework_idea == "1":
            self.build_model_framework1( lr, lr_d , units, dr ,epochs)
        elif self.framework_idea == "2":
            self.build_model_framework2( lr, lr_d , units, dr ,epochs)
        else : 
            self.build_model_framework3( lr, lr_d , units, dr ,epochs)
        
    def train_Model(self, lr = 1e-3, lr_d = 0, units = 112, dr = 0.2,epochs = 30):
        #For training use this member function
        print("TRAINING STARTED")
        self.build_model(lr , lr_d, units, dr ,epochs)
        print("TRAINING COMPLETED")
    
    def get_accuracy_for_validation_set(self):
        return self.prediction(self.modelBase.X_valid, self.modelBase.Y_valid)
        
    def get_accuracy_for_training_set(self):
        return self.prediction(self.modelBase.X_train, self.modelBase.Y_train)
         
    def Plot(self, string): # example Object.Plot(history, "accuracy")  or Object.Plot(history, "loss")
        plt.plot(self.history.history[string])
        plt.plot(self.history.history['val_' + string])
        plt.xlabel("EPOCHS")
        plt.ylabel(string)
        plt.legend([string, 'val_' + string ])
        plt.savefig(string + "_framework-" + self.framework_idea  +'_' + self.modelBase.model_Type  + '_'+'.png')
        plt.show()       
        print(f"###### DONE PLOTTING FOR IDEA - {self.framework_idea} ########")
                    
    def prediction(self,dataset, y_actual,on_the_fly=0 ): # On the fly is for those data that is comimg on the fly from any tweet.
        y_pred_validation = self.model.predict(dataset, verbose=1)
        score_validation = roc_auc_score(y_actual, y_pred_validation)
        print(f"\n ROC-AUC - ON given Dataset - score: {round(score_validation,5)*100}%")
        return y_pred_validation
        
##############################################################################################################################################################





if __name__ == "__main__":
    print("Training File is Running..")

    #INAPPROPRIATE CONTENT DETECTION
    start = time.time()
    InappContentModel1Preprocessor = Preprocessor( embed_size = 300, 
                                                   max_features = 130000, 
                                                   max_len = 220, 
                                                   model_Type = "Inappropriate-Content-Detection")

    end = time.time()
    print(f'Time Taken for Creating Preprocessor Object : {end-start}')


    start = time.time()
    #HYPER PARAMETER 1
    validation_size = 0.3
    InappContentModel1Preprocessor.preprocessing( validation_size=validation_size,
                                      embedding_path_fastText = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec",
                                      embedding_path_glove ="../input/glove840b300dtxt/glove.840B.300d.txt" ,
                                      isEmbeddingIndexFileSaved     = False,
                                      isEmbeddingMatrixFileSaved    = False,
                                      wantToSaveEmbeedingIndexFile  = False,
                                      wantToSaveEmbeedingMatrixFile = False )

    end = time.time()
    print(f'Time Taken for Preprocessing : {end-start}')


    start = time.time()
    InappContentModel1 = ML_Model( InappContentModel1Preprocessor,
                                   file_path = "1",
                                   n_multiClassificationClasses = 6 )

    end = time.time()
    print(f'Time Taken for Creating ML Model Object: {end-start}')

    start = time.time()
    #HYPER PARAMETER 2
    learning_weight_decay = 0.1
    #HYPER PARAMETER 3
    dropout = 0.2
    #HYPER PARAMETER 4
    epochs = 50
    InappContentModel1.train_Model(lr = 1e-3, lr_d = learning_weight_decay, units = 112, dr = dropout, epochs = epochs )
    end = time.time()
    print(f'Time Taken for Training For Inapprpriate Content : {end-start}')

   
    import tensorflow as tf
    tf.keras.utils.plot_model(InappContentModel1.model, to_file=f'Plotted_model-{InappContentModel1.modelBase.model_Type}--Framework-{InappContentModel1.framework_idea}-.png')
    InappContentModel1.model.summary()



    print(InappContentModel1.n_output_nuerons )
    print(InappContentModel1.modelBase.embedding_matrix_glove.shape[0])
    print(InappContentModel1.modelBase.max_features)

    start = time.time()
    y_pred_train = InappContentModel1.get_accuracy_for_training_set()
    end = time.time()
    print(f"Time taken for checking accuracy of Training set : {end-start}")

    start = time.time()
    y_pred_valid  = InappContentModel1.get_accuracy_for_validation_set()
    end = time.time()
    print(f"Time taken for checking accuracy of Dev set : {end-start}")

    InappContentModel1.Plot( "loss")

    InappContentModel1.Plot( "accuracy")
    # %%time
    # CHECKING FOR IDEA 2 BY SWITCHING 
    InappContentModel1.change_idea_for_same_object( change_idea_to = "2", automatic_train = True)
    # %%time
    # Note this prediction is  FOR IDEA 2 
    y_pred_valid  = InappContentModel1.get_accuracy_for_validation_set()

    # Note this prediction is  FOR IDEA2
    InappContentModel1.Plot( "accuracy")

    # Note this prediction is  FOR IDEA 2 
    InappContentModel1.Plot( "loss")




    ############################################################################################################################
    ############################################################################################################################
    ############################################################################################################################
    ############################################################################################################################





    #SPAM DETECTION




  
    # %%time
    #HYPER PARAMETER 1
    validation_size = 0.3
    #HYPER PARAMETER 2
    learning_weight_decay = 0.1
    #HYPER PARAMETER 3
    dropout = 0.2
    #HYPER PARAMETER 4
    epochs = 50
    start = time.time()
    spamDetectionModel1Preprocessor = Preprocessor( embed_size = 300, 
                                                   max_features = 130000, 
                                                   max_len = 220, 
                                                   model_Type = "Spam-Detection")

    end = time.time()
    print(f"Time taken for Creating Preprocessor Object : {end-start}")
    start = time.time()
    # %%time
    spamDetectionModel1Preprocessor.preprocessing(  validation_size=validation_size,
                                      embedding_path_fastText = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec",
                                      embedding_path_glove ="../input/glove840b300dtxt/glove.840B.300d.txt" ,
                                      isEmbeddingIndexFileSaved     =False,
                                      isEmbeddingMatrixFileSaved    = False,
                                      wantToSaveEmbeedingIndexFile  = False,
                                      wantToSaveEmbeedingMatrixFile = False )

    end = time.time()
    print(f"Time taken for Preprocessing : {end-start} ")
    # %%time
    start = time.time()
    spamDetectionModel1 = ML_Model( spamDetectionModel1Preprocessor,
                                   file_path = "1",
                                   n_multiClassificationClasses = 1 )
    end = time.time()
    print(f"Time taken for constructing ML Model: {end-start}")

    start = time.time()
    # %%time
    spamDetectionModel1.train_Model(lr = 1e-3, lr_d = learning_weight_decay, units = 112, dr = dropout, epochs = epochs)
    end = time.time()
    print(f"Time taken for Training : {end-start}")

    # %%time
    start = time.time()
    y_pred_valid  = spamDetectionModel1.get_accuracy_for_validation_set()
    end = time.time()
    print(f"Time taken for calculating accuracy for Dev set : {end-start}")
    spamDetectionModel1.Plot( "loss")

    spamDetectionModel1.Plot( "accuracy")
