

import imp


import re
def is_twisted_word(word_list):
    new_list=[]
    for word in word_list:
        if word.isalpha():
            new_list.append(word)
        else:
            # for c in word:
            #     if c=="@":
            #         c="a"
            #     elif c=="!":
            #         c="i"
            #     elif c=="$":
            #         c="s"
            #     elif c=="&":
            #         c="s"

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
            elif re.search("^d.*ad$", word):
                word="dickhead"
            elif re.search("^mot.*ker$", word):
                word="motherfucker"
            elif re.search("^sh.*t$", word):
                word="shit"
            elif re.search("^f.*k$", word):
                word="fuck"
            elif re.search("^$h.*t$", word):
                word="shit"
        
        new_list.append(word)
    return new_list          


tweet= [ "you", "b!tch",  "d!ck", "f*#k", "mot#*@ker", "d!ck", "&h!t", "b@@bs", "b()()bs", "ar$eh@le", "a$$hole"]

print(is_twisted_word(tweet))