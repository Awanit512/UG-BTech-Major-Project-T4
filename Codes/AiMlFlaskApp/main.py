########################################################################################
######################          Import packages      ###################################
########################################################################################
#Importing required libraries 
# from numpy import asarray
# from numpy import loadtxt
# from numpy import savetxt
import time
start_time = time.time()
from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_required, current_user
from models import InappropriateContentTweet
from __init__ import create_app, db


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
from prediction import PredictorObject



end = time.time()
import_time = round((end-start_time)*1000,3)
np.random.seed(32)
# os.environ["OMP_NUM_THREADS"] = "4"


print(f"'\n\nALL LIBRARIES IMPORTED SUCCESSFULLY!!' in TIME : {import_time} msec")


########################################################################################
# our main blueprint
main = Blueprint('main', __name__)


def createPredictorObject():
    start = time.time()
    actual_model_path_for_inapp = './Trained_model/best_model_FrameWork1_Inappropriate-Content-Detection.hdf5'
    InappPrediction = PredictorObject(model_Type = "Inappropriate-Content-Detection", 
                                      file_path = "1" ,
                                      n_multiClassificationClasses = 6, 
                                      tweet_column='comment_text'  ,
                                     actual_model_path = actual_model_path_for_inapp,
                                     training_dataset_path = "./dataset/train.csv" 
                                     )


    end = time.time()
    print(f"Time taken to construct Predictor Object : {end-start} seconds")
    return InappPrediction


# InappPrediction =createPredictorObject()


def createNewTweet(**kwargs):
    newTweet = InappropriateContentTweet(   tweet=kwargs["tweet"], 
                                            subjectP=kwargs["P"] ,  
                                            ObjectQ=kwargs["Q"],
                                            ConnectorR=kwargs["R"],
                                            classToxic=kwargs["Toxic"],
                                            classSevereToxic=kwargs["SevereToxic"],
                                            classObscene=kwargs["Obscene"],
                                            classThreat=kwargs["Threat"],
                                            classInsult=kwargs["Insult"],
                                            classIdentityHate=kwargs["IdentityHate"])
    return newTweet

def getInappropriateClasses(result,list_classes):
    params = {}
    params["Toxic"] =  [round(result[list_classes[0]][0] ,2), str(round(100*result[list_classes[0]][0] ,2))+" %"]
    params["SevereToxic"] = [ round(result[list_classes[1]][0] ,2), str(round(100*result[list_classes[1]][0] ,2))+" %"]
    params["Obscene"] = [ round(result[list_classes[2]][0] ,2), str(round(100*result[list_classes[2]][0] ,2)) + " %" ] 
    params["Threat"] = [ round(result[list_classes[-3]][0] ,2), str(round(100*result[list_classes[-3]][0] ,2)) + " %" ]
    params["Insult"] =  [ round(result[list_classes[-2]][0] ,2), str(round(100*result[list_classes[-2]][0] ,2)) + " %"]
    params["IdentityHate"] = [ round(result[list_classes[-1]][0] ,2), str(round(100*result[list_classes[-1]][0] ,2)) + " %"]
    return params

def getPQR(tweet):
    return "None","None","None"





@main.route('/') # home page that return 'index'
def index():
    return render_template('index.html')



@main.route('/checkTweet', methods=['GET', 'POST']) # define check tweet path
def checkTweet(): # define login page fucntion
    if request.method=='GET': # if the request is a GET we return the login page
        return render_template('tweet.html',tweet=None,params=None)
    else: # if the request tweet is POST 
        tweet = request.form.get('tweet')
        print("\n\n---------------------------\n\n",tweet,"\n\n----------------------------------------\n\n")
        # return render_template('tweet.html',tweet=tweet)
        print("Prediction has started...")
        # tweet2 ='you fucking bitch will cry whole life and i will cut you in pieces. Beacause i am monster. Be ready, i am coming for you and when i will find you , i will wash my hand with your blood'
        start = time.time()

        print("############################## ML PART Predictor #################################")
        print("From tweet.html we get this tweet ",tweet)
        InappPrediction = createPredictorObject()
        ##########################################################   ML MODEL PREDICTION ######################################################################################################
        result, tweet_column, list_classes, result_filename = InappPrediction.prediction( tweet,isText=True, isArray=False, isDataFrame=False, tweet_column='comment_text'  )

        for classes in list_classes:
            print(classes,result[classes][0])

        end = time.time()
        print("Time Taken to predict : ",end - start)

        params = getInappropriateClasses(result,list_classes)
        # params = {"Toxic": 0.9, "SevereToxic":0.8,"Obscene" :0.5, "Threat":0.40, "Insult":0.5, "IdentityHate":0.8}
        # appointedColour = {"Toxic": (255,0,0), "SevereToxic":(0,250,5),"Obscene" :(0,0,195), "Threat":(255,255,255), "Insult":(255,255,5), "IdentityHate":(255,0,26)}

        ##########################################################   AI MODEL PREDICTION ######################################################################################################
        print("############################## AI PART Predictor #################################")
        P,Q,R=getPQR(tweet)
        ##########################################################   AI MODEL PREDICTION ######################################################################################################
        newTweet = createNewTweet(tweet = tweet,
                                    P=P,
                                    Q=Q,
                                    R=R,
                                    Toxic=params["Toxic"][0],
                                    SevereToxic = params["SevereToxic"][0], 
                                    Obscene = params["Obscene"][0],
                                    Threat= params["Threat"][0], 
                                    Insult=params["Insult"][0],
                                    IdentityHate=params["IdentityHate"][0] )
        # add the new tweet to the InappropriateContentTweet database
        db.session.add(newTweet)
        db.session.commit()
        return render_template('tweet.html',tweet=tweet,params=params)












#################################################
































        # check if the user actually exists
#         # take the user-supplied password, hash it, and compare it to the hashed password in the database
#         if not user:
#             flash('Please sign up before!')
#             return redirect(url_for('auth.signup'))
#         elif not check_password_hash(user.password, password):
#             flash('Please check your login details and try again.')
#             return redirect(url_for('auth.login')) # if the user doesn't exist or password is wrong, reload the page
#         # if the above check passes, then we know the user has the right credentials
#         login_user(user, remember=remember)
#         return redirect(url_for('main.profile'))
# ########



# @auth.route('/signup', methods=['GET', 'POST'])# we define the sign up path
# def signup(): # define the sign up function
#     if request.method=='GET': # If the request is GET we return the sign up page and forms
#         return render_template('signup.html')
#     else: # if the request is POST, then we check if the email doesn't already exist and then we save data
#         email = request.form.get('email')
#         name = request.form.get('name')
#         password = request.form.get('password')
#         user = User.query.filter_by(email=email).first() # if this returns a user, then the email already exists in database
#         if user: # if a user is found, we want to redirect back to signup page so user can try again
#             flash('Email address already exists')
#             return redirect(url_for('auth.signup'))
#         # create a new user with the form data. Hash the password so the plaintext version isn't saved.
#         new_user = User(email=email, name=name, password=generate_password_hash(password, method='sha256')) #

# ###############








@main.route('/profile') # profile page that return 'profile'
@login_required
def profile():
    return render_template('profile.html', name=current_user.name)

app = create_app() # we initialize our flask app using the __init__.py function

if __name__ == '__main__':
    print("Starting APP...")
    db.create_all(app=create_app()) # create the SQLite database
    app.run(debug=True,port = 7211) # run the flask app on debug mode
    print("Application Started.")
