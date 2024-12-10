#!/usr/bin/python
"""
The issue is that when we remove outputs with to many syllables
we may get a word which doesnt map to anything. 
We are currently handling this by setting the new word to Alice.

Another thing that could be improved is to use scipy sparse matrices.

Vi skulle också kunna överväga att göra om vissa av variablerna (de som aldrig ändras) till konstanter, så att vi slipper ha med dem i varje funktionsanrop
"""
import nltk
from nltk.corpus import cmudict
import numpy as np
import syllables as s
import random
from collections import Counter
from sys import stderr

ORDER = 2
N_TRAIN_WORDS = 2000
PUNCTUATION_MARKS = [".", "!", "?", ";", ":", "--"]

#nltk.download("cmudict")
cmudict_dict = cmudict.dict()

def main():
    random.seed()
    #f = open("asv_formatted.txt", "r")
    f = open("alice_in_wonderland.txt", "r")
    corpus=f.read()
    f.close()
    print("generating matrix (extremely inefficient)", file=stderr)
    words, mat, word_tuples, possible_final_words, possible_start_indices = create_transition_matrix(corpus)
    print("generated matrix\n", file=stderr)
    print("matrix multiplying... (also extremely inefficient)\n", file=stderr)
    print(create_haikus(10, mat, words, word_tuples, possible_final_words, possible_start_indices))
    return

def syllables(word):
    word = ''.join(c for c in word if c.isalnum()).lower()
    # from SE
    # https://datascience.stackexchange.com/questions/23376/how-to-get-the-number-of-syllables-in-a-word
    if word in cmudict_dict:
        return [len(list(y for y in x if y[-1].isdigit())) for x in cmudict_dict[word]][0]
    else:
        return s.estimate(word) # fallback


# returns state index
def next_stateindex(transition_matrix, current_state):
    v = np.matmul(transition_matrix, current_state)

    if(sum(map(abs,v)) <= 0): #placeholder
        return None
    i = random.choices(range(0, len(v)), v)[0]
    return i

# Set some rows of matrix to 0 and reweight columns to 1
def mutate_matrix(mat, indices):
    for i in indices:
        mat[i, :] = 0
    for i in range(mat.shape[1]):
        colsum=sum(mat[:, i])
        if(colsum != 0):
            mat[:, i] = mat[:,i] / colsum
    return mat

# we could make it more efficient by removing indices from 
# matrices already removed from but cba
def create_syllable_matrixes(transition_matrix, words):
    mat_list = []
    for i in range(1,8): # python ranges are not inclusive dumbo
        li = unacceptable_indices(i, words)
        mat_list.append(mutate_matrix(transition_matrix.copy(), li))
    return mat_list

def create_haikus(n_haikus, transition_matrix, words, word_tuples, possible_final_words, possible_start_indices): 
    haikus = []
    mat_list = create_syllable_matrixes(transition_matrix, words)
    while(len(haikus) < n_haikus):
        initial_stateindex = random.choice(possible_start_indices)
        t = create_haiku_internal(mat_list, words, initial_stateindex, word_tuples, possible_final_words)
        if t != None:
            haikus.append(t)
    return "\n\n".join(haikus)+"\n"

# when random initials it is most likely that it is the initial condition which messes things up
def create_haiku_internal(mat_list, words, initial_stateindex, word_tuples, possible_final_words): 
    state=np.zeros((mat_list[0].shape[1], 1))
    i = word_tuples.index((words[initial_stateindex],))
    state[i]=1
    lines = [[words[initial_stateindex]]]
    previous_words = [words[initial_stateindex]]
    sylls_used = syllables(words[initial_stateindex])
    l=[5,7,5]
    for k in range(0,3):
        while(sylls_used < l[k]):
            j = i
            i = next_stateindex(mat_list[l[k]-1-sylls_used], state)
            if i == None:
                return None
                #return create_haiku_internal(mat_list, words, initial_stateindex, word_tuples, possible_final_words)
            sylls_used+=syllables(words[i])
            lines[k].append(words[i])     
            previous_words.append(words[i])    #eftersom de innehåller samma element kan vi eventuellt överväga att göra så att vi bara har listan previous words och inte listan lines
            state[j]=0
            for m in range(-ORDER,0):
                try:
                    #print("hello there, word is ", word_tuples[i])
                    i = word_tuples.index(tuple(previous_words[m:]))
                    #print("we got word ", word_tuples[i])
                    break
                except ValueError:       #om tupeln inte finns med i listan
                    pass

            state[i]=1      
        lines.append([])
        sylls_used=0
    if not (previous_words[-1].endswith(tuple(PUNCTUATION_MARKS)) or tuple(previous_words[-2:]) in possible_final_words):
        return None
        #return create_haiku_internal(mat_list, words, initial_stateindex, word_tuples, possible_final_words)
    lines.pop(-1)    #tar bort den tomma raden i slutet
    if lines[-1][-1][-1] in [",", ";", ":", "--"]:
        lines[-1][-1] = lines[-1][-1][:-1]     #tar bort skiljetecknet i slutet av sista ordet om det är , eller ;
    lines = map(lambda l: " ".join(l), lines)
    return "\n".join(lines)
    

# get indices of words which have too many syllables
def unacceptable_indices(max_syllables, words):
    indices=[]
    for i in range(len(words)):
        if(syllables(words[i]) > max_syllables):
            indices.append(i)
    return indices

# This returns a matrix A where each column sums to one, assuming the word the column represents
# is not both unique and the last word (then it does not transform to a new word)
# Ax where x is a vector with some word gives us a new vector which is of the probabilities of each word
def create_transition_matrix(corpus):
    corpus = "".join(char for i, char in enumerate(corpus) 
                     if (char.isalpha() or char in ([" ", "\n", "-"] + PUNCTUATION_MARKS) or (char == "'" and i-1 >= 0 and corpus[i-1].isalpha())))
    corpus=corpus.split()
    corpus=corpus[:N_TRAIN_WORDS] # cut the training data to the size we want
    possible_final_words = []
    for i in range(1, len(corpus)):
        if corpus[i].strip()[-1] in PUNCTUATION_MARKS:
            possible_final_words.append((corpus[i-1], corpus[i][:-1]))    #lägger till ordet som avslutas med ett skiljetecken och ordet före det

                
    d = {}
    for i in range(1, len(corpus)):
        for nr_of_words in range(1, ORDER+1):
            if i-nr_of_words < 0:
                break
            tuple_of_words = tuple([corpus[j] for j in range(i-nr_of_words, i)])
            if not tuple_of_words in d:
                d[tuple_of_words] = Counter()
            d[tuple_of_words][corpus[i]]+=1

    d_new = {}
    for (key, c) in d.items():
        s = sum(c.values())
        c_new = Counter()
        for (w, n) in c.items():
            c_new[w] = n/s
        d_new[key]=c_new
    d=d_new

    
    unique_single_words = list(dict.fromkeys(corpus))
    unique_tuples = list(d.keys())


    arr = np.empty((len(unique_single_words), len(unique_tuples)))
    for xcount, x in enumerate(unique_tuples):
        for ycount, y in enumerate(unique_single_words):
            arr[ycount,xcount] = d[x][y]

    possible_start_indices = []
    for i in range(0, len(unique_single_words)):
        if unique_single_words[i][0].isupper() and syllables(unique_single_words[i]) <= 5:
            possible_start_indices.append(i)    #adds word beginning with capital letter

    return unique_single_words, arr, unique_tuples, possible_final_words, possible_start_indices

main()
