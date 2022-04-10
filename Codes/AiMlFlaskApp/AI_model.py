
import re
import nltk
from dbCRUD import fetchContentFromTable
from databasePopulator import  tableList, nameOfDB
#nltk.download('punkt')

columnsList = {}
for tableName in tableList:
    columnsList[tableName] = ''.join(tableName.split("_"))



#######################TOXIC#####################
def is_toxic(input_tweet):
    file1 = open("toxic_pred_p.txt", "r")
    file2 = open("toxic_pred_q.txt", "r")
    # print(file1.read())#this prints the whole text file

    pred_p = file1.read().splitlines()  # read the file without newlines..important
    pred_q = file2.read().splitlines()

    # match each token of tweet against the dict.
    # belongs:2 means two pairs of p&q matched.
    belongs_to_ = 0
    for token in input_tweet:
        if token in pred_p:
            for token_word in input_tweet:
                if token_word in pred_q:
                    belongs_to_ += 1
    #print(belongs_to_)

    file1.close()
    file2.close()
    return belongs_to_


#####################THREAT##################
def is_threat(input_tweet):
    file1 = open("threat_pred_p.txt", "r")
    file2 = open("threat_pred_q.txt", "r")
    # print(file1.read())#this prints the whole text file

    pred_p = file1.read().splitlines()  # read the file without newlines..important
    pred_q = file2.read().splitlines()

    # match each token of tweet against the dict.
    # belongs:2 means two pairs of p&q matched.
    belongs_to_ = 0
    for token in input_tweet:
        if token in pred_p:
            for token_word in input_tweet:
                if token_word in pred_q:
                    belongs_to_ += 1
    #print(belongs_to_)

    file1.close()
    file2.close()
    return belongs_to_




#####################IDENTITY HATE##################
##SIMPLE VERSION##
def isInappropriateClass_(  input_tweet, 
                            nameOfDB,
                            tableNameForP, 
                            tableNameForWeakQ, 
                            tableNameForStrongQ, 
                            colNameOfP=None, 
                            colNameWeakOfQ=None, 
                            colNameStrongOfQ=None,
                            colIndexOfP = 0, 
                            colIndexOfWeakOfQ=0,
                            colIndexOfStrongQ=0):
    # file1 = open("identity_hate_p.txt", "r")
    # file2 = open("identity_hate_weak_q.txt", "r")
    # file3 = open("identity_hate_strong_q.txt","r")
    # print(file1.read())#this prints the whole text file

    pred_p = fetchContentFromTable(nameOfDB,tableNameForP, columnsList[ columnsList[tableNameForP] ], colIndex =colIndexOfP ) #file1.read().splitlines()  # read the file without newlines..important
    pred_weak_q = fetchContentFromTable(nameOfDB,tableNameForWeakQ, colIndex =colIndexOfWeakOfQ ) # file2.read().splitlines()
    pred_strong_q =  fetchContentFromTable(nameOfDB,tableNameForStrongQ, colIndex =colIndexOfStrongQ ) #file3.read().splitlines()

    # match each token of tweet against the dict.
    # belongs:2 means two pairs of p&q matched.
    belongs_to_negative = 0
    belongs_to_weak = 0
    belongs_to_strong = 0
    for token in input_tweet:
        if token in ["not", "no", "never", "none"]:
            print("Negative:=",token)
            belongs_to_negative += 1
        if token in pred_p:
            print("Subject=",token)
            for token_word in input_tweet:
                if token_word in pred_weak_q:
                    print("Weak bad=",token_word)
                    belongs_to_weak += 1
            for token_word in input_tweet:
                if token_word in pred_strong_q:
                    print("Strong-bad=",token_word)
                    belongs_to_strong += 1
    #print(belongs_to_)

    file1.close()
    file2.close()
    if (belongs_to_strong > 0) or (belongs_to_weak > 0 and belongs_to_negative == 0):
        return True
    else:
        return False
###############################################




#####################IDENTITY HATE##################
##LOGICAL-NOT VERSION##
def isInappropriateClass(   input_tweet, 
                            nameOfDB,
                            tableNameForP, 
                            tableNameForWeakQ, 
                            tableNameForStrongQ, 
                            colNameOfP=None, 
                            colNameWeakOfQ=None, 
                            colNameStrongOfQ=None,
                            colIndexOfP = 0, 
                            colIndexOfWeakOfQ=0,
                            colIndexOfStrongQ=0):
    # file1 = open("identity_hate_p.txt", "r")
    # file2 = open("identity_hate_weak_q.txt", "r")
    # file3 = open("identity_hate_strong_q.txt","r")
    # print(file1.read())#this prints the whole text file

    pred_p = fetchContentFromTable(nameOfDB,tableNameForP, columnsList[tableNameForP] ,colIndex =colIndexOfP ) #file1.read().splitlines()  # read the file without newlines..important
    pred_weak_q = fetchContentFromTable(nameOfDB,tableNameForWeakQ, columnsList[tableNameForWeakQ],colIndex =colIndexOfWeakOfQ ) # file2.read().splitlines()
    pred_strong_q =  fetchContentFromTable(nameOfDB,tableNameForStrongQ, columnsList[tableNameForStrongQ], colIndex =colIndexOfStrongQ ) #file3.read().splitlines()
    print(pred_p)
    print(pred_weak_q)
    print(pred_strong_q)

    # match each token of tweet against the dict.
    # belongs:2 means two pairs of p&q matched.
    belongs_to_negative = 0
    belongs_to_weak = 0
    belongs_to_strong = 0
    P_R_Q = [None,None,None]
    n=len(input_tweet)
    ##################
    for i in range(n):
        token=input_tweet[i]
        if token in pred_p:
            print("Subject=",token)
            P_R_Q[0] = token
            ####################
            for j in range(i,n):
                token=input_tweet[j]
                if token in ["not", "no", "never", "none"]:
                    print("Negative:=",token)
                    P_R_Q[1] = token
                    belongs_to_negative += 1
                    ####################
                    for k in range(j,n):
                        token=input_tweet[k]
                        if token in pred_weak_q:
                            print("Weak bad=",token)
                            P_R_Q[2] = token
                            belongs_to_weak += 1
                        if token in pred_strong_q:
                            print("Strong-bad=",token)
                            P_R_Q[2] = token
                            belongs_to_strong += 1
            #------------------------------------------
            if (belongs_to_negative>0 and belongs_to_weak>0):
                return False,P_R_Q
            if (belongs_to_negative>0 and belongs_to_strong>0):
            # if (belongs_to_strong>0):
                # file1.close()
                # file2.close()
                return True,P_R_Q
            #-------------------------------------------
            for k in range(i,n):
                token=input_tweet[k]
                if token in pred_weak_q:
                    print("Weak bad=",token)
                    P_R_Q[2] = token
                    belongs_to_weak += 1
                if token in pred_strong_q:
                    print("Strong-bad=",token)
                    P_R_Q[2] = token
                    belongs_to_strong += 1
            #---------------------------------------------
            if (belongs_to_strong>0 or belongs_to_weak>0):
                # file1.close()
                # file2.close()
                return True,P_R_Q

    #print(belongs_to_)

    # file1.close()
    # file2.close()
    if (belongs_to_strong > 0) or (belongs_to_weak > 0 and belongs_to_negative == 0):
        return True,P_R_Q
    else:
        return False,P_R_Q
###############################################




#########################MAIN#######################
def apply_ai(input_tweet):
    # intensity_toxic = is_toxic(input_tweet)
    # intensity_threat = is_threat(input_tweet)
    # intensity_toxic = is_toxic(input_tweet)
    # if (intensity_toxic > 0):
    #     print("Tweet is toxic, can't be published")
    # intensity_threat = is_threat(input_tweet)
    # if (intensity_threat > 0):
    #     print("Tweet is theatening, can't be published")
    probabilities = [0,0,0,0,0,0]
    P__R__Q = [[None,None,None],[None,None,None],[None,None,None],[None,None,None],[None,None,None],[None,None,None]]
    def make_it_True(indx):
        if indx == 0 :
            probabilities[-1] = 1
            return -1
        elif indx == 3:
            probabilities[2] = 1
            return 2
        elif indx == 6:
            probabilities[-2] = 1
            return -2
        elif indx == 9:
            probabilities[-3] = 1
            return -3
        elif indx == 12:
            probabilities[0] = 1
            return 0
        elif indx == 15:
            probabilities[1] = 1
            return 1

    for index in range(0,len(tableList),3):
        res,PRQ = isInappropriateClass(input_tweet,nameOfDB, tableList[index], tableList[index+1], tableList[index+2] )
        if res:
            idx = make_it_True(index)
            P__R__Q[idx][0] , P__R__Q[idx][1] , P__R__Q[idx][2] = PRQ[0] , PRQ[1] , PRQ[2]

    if any(probabilities):
        print("Tweet belongs To--> Inappropriate Class")
        return True,probabilities,P__R__Q 
    else:
        print("Tweet is clean")
        return False,probabilities,P__R__Q 





if  __name__ == "__main__":
    tweet= "hey you bitch came and eat my dick how dare to fuck with me i will cut shed your blood"
    tweet1= "all christians are bastard by birth"#strong-bad => inapp
    tweet2= "the race of muslims is full of terrorists"#strong-bad => inapp
    tweet3 = "all hindu are not assholes"#strong-not-bad => inapp
    tweet4 = "all indians are cunt"#weak-bad => inapp
    tweet5 = "all americans are not dog"#weak-not-bad => app

    tweet1=nltk.word_tokenize(tweet1)
    print(f"Tweet 1 : {tweet1}")
    tweet2=nltk.word_tokenize(tweet2)
    print(f"Tweet 2 : {tweet2}")
    tweet3=nltk.word_tokenize(tweet3)
    print(f"Tweet 3 : {tweet3}")
    tweet4=nltk.word_tokenize(tweet4)
    print(f"Tweet 4 : {tweet4}")
    tweet5=nltk.word_tokenize(tweet5)
    print(f"Tweet 5 : {tweet5}")

    # apply_ai(input_tweet)
    print(f"\n\n1.0 : {tweet1} \n\n")
    print(apply_ai(tweet1))
    print(f"\n\n2.0 : {tweet2} \n\n")
    print(apply_ai(tweet2))
    print(f"\n\n3.0 : {tweet3} \n\n")
    print(apply_ai(tweet3))
    print(f"\n\n4.0 : {tweet4} \n\n")
    print(apply_ai(tweet4))
    print(f"\n\n5.0 : {tweet5} \n\n")
    print(apply_ai(tweet5))


