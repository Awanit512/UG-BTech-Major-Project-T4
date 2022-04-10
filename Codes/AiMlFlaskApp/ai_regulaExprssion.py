

# import imp
import re
def is_twisted_word(word_list):
    new_list=[]
    for word in word_list:
        if re.search("^a.*ole$", word):
            word="asshole"
        elif re.search("^a.*@le",word):
            word="asshole"
        elif re.search("^b.*ch$", word):
            word="bitch"
        elif re.search("^b.*bs$", word):
            word="boobs"
        elif re.search("^d.*k$", word):
            word="dick"
        elif re.search("^[d|D].*AD$", word):
            word="dickhead"
        elif re.search("^[d|D].*ad$", word):
            word="dickhead"
        elif re.search("^mot.*ker$", word):
            word="motherfucker"
        elif re.search("^[$&s][h#].*t$", word):
            word="shit"
        elif re.search("^[f|F].*[k|K]$", word):
            word="fuck"
        elif re.search("^[f|F].*ing$", word):
            word="fucking"
        elif re.search("^[f|F].*NG$", word):
            word="fucking"
        elif re.search("^$h.*t$", word):
            word="shit"
        elif re.search("s.*x$",word):
            word  = "sex"
        elif word in ["dck" , "d!k" , "d!ck" , "d**k" , "d*k" , "d?ck" , "d??k"]:
            word="dick"
        elif re.search("^$h.*t$", word):
            word="shit"
        elif re.search("s.*x$",word):
            word  = "sex"
        elif re.search("s.*x$",word):
            word  = "sex"
        elif re.search("^[a@].*ss$",word):
            word  = "ass"
        elif re.search("^[a@].*\$\$$",word):
            word  = "ass"
        elif re.search("^coc.*s",word):
            word  = "cock"
        new_list.append(word)
    return new_list          


if __name__ == "__main__":

    tweet= ["fake", "a$$","you", "b!tch",  "d!ck", "f*#k", "mot#*@ker", "d!ck", "&h!t", "b@@bs", "b()()bs", "ar$eh@le", "a$$hole", "fuk" , "f|_|k", "F|_|K" , "F|_|CK" , "f|_|ck" , "B * 0 **  O B S" , "$#!t" , "coc|<s" , "a$$" , "f*ck" , "sx" , "dck" , "_a_s--_s_s" , "___a_$$"]
    print("\n")
    # print(tweet)
    # print(is_twisted_word(tweet))
    print("word | actual Word \n")
    for m,n in zip(tweet,is_twisted_word(tweet)):
        print(f"{m} \t ---> {n} ")
