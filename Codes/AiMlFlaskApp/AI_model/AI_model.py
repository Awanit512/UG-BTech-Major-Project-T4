

#25.03.2022
import re
import nltk
#nltk.download('punkt')
tweet= "hey you bitch came and eat my dick how dare to fuck with me i will cut shed your blood"

tweet1= "all christians are bastard by birth"#strong-bad => inapp
tweet2= "the race of muslims is full of terrorists"#strong-bad => inapp
tweet3 = "all hindu are not assholes"#strong-not-bad => inapp

tweet4 = "all indians are cunt"#weak-bad => inapp
tweet5 = "all americans are not dog"#weak-not-bad => app


tweet1=nltk.word_tokenize(tweet1)
tweet2=nltk.word_tokenize(tweet2)
tweet3=nltk.word_tokenize(tweet3)
tweet4=nltk.word_tokenize(tweet4)
tweet5=nltk.word_tokenize(tweet5)
#print(input_tweet)#tokenized tweet

#dictionary for toxic words,
#created and stored in text file.
toxic_predicate_p = open("toxic_pred_p.txt","a")
toxic_predicate_q = open("toxic_pred_q.txt","a")

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
def is_identity_hate(input_tweet):
    file1 = open("identity_hate_p.txt", "r")
    file2 = open("identity_hate_weak_q.txt", "r")
    file3 = open("identity_hate_strong_q.txt","r")
    # print(file1.read())#this prints the whole text file

    pred_p = file1.read().splitlines()  # read the file without newlines..important
    pred_weak_q = file2.read().splitlines()
    pred_strong_q = file3.read().splitlines()

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
def is_identity_hate_(input_tweet):
    file1 = open("identity_hate_p.txt", "r")
    file2 = open("identity_hate_weak_q.txt", "r")
    file3 = open("identity_hate_strong_q.txt","r")
    # print(file1.read())#this prints the whole text file

    pred_p = file1.read().splitlines()  # read the file without newlines..important
    pred_weak_q = file2.read().splitlines()
    pred_strong_q = file3.read().splitlines()

    # match each token of tweet against the dict.
    # belongs:2 means two pairs of p&q matched.
    belongs_to_negative = 0
    belongs_to_weak = 0
    belongs_to_strong = 0

    n=len(input_tweet)
    ##################
    for i in range(n):
        token=input_tweet[i]
        if token in pred_p:
            print("Subject=",token)
            ####################
            for j in range(i,n):
                token=input_tweet[j]
                if token in ["not", "no", "never", "none"]:
                    print("Negative:=",token)
                    belongs_to_negative += 1
                    ####################
                    for k in range(j,n):
                        token=input_tweet[k]
                        if token in pred_weak_q:
                            print("Weak bad=",token)
                            belongs_to_weak += 1
                        if token in pred_strong_q:
                            print("Strong-bad=",token)
                            belongs_to_strong += 1
            #------------------------------------------
            if (belongs_to_negative>0 and belongs_to_weak>0):
                return False
            elif (belongs_to_negative>0 and belongs_to_strong>0):
                return True
            #-------------------------------------------
            for k in range(i,n):
                token=input_tweet[k]
                if token in pred_weak_q:
                    print("Weak bad=",token)
                    belongs_to_weak += 1
                if token in pred_strong_q:
                    print("Strong-bad=",token)
                    belongs_to_strong += 1
            #---------------------------------------------
            if (belongs_to_strong>0 or belongs_to_weak>0):
                return True

    #print(belongs_to_)

    file1.close()
    file2.close()
    if (belongs_to_strong > 0) or (belongs_to_weak > 0 and belongs_to_negative == 0):
        return True
    else:
        return False
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

    res=is_identity_hate_(input_tweet)
    if res==True:
        print("Tweet is full of identity hate")
    else:
        print("Tweet is clean")

# apply_ai(input_tweet)
apply_ai(tweet1)
apply_ai(tweet2)
apply_ai(tweet3)
apply_ai(tweet4)
apply_ai(tweet5)

