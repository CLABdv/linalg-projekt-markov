#!/usr/bin/python
"""
The issue is that when we remove outputs with to many syllables
we may get a word which doesnt map to anything. 
We are currently handling this by setting the new word to Alice.

Also, make it so that it remembers more previous states.
Currently hardcoded to only one previous state

Another thing that could be improved is to use scipy sparse matrices.
"""
import syllables
import numpy as np
import random
from collections import Counter
from sys import stderr

def main():
    f = open("alice_in_wonderland.txt", "r")
    corpus=f.read()
    f.close()
    print("generating matrix (extremely inefficient)", file=stderr)
    words, mat = create_transition_matrix(20000, corpus)
    print("generated matrix\n", file=stderr)
    print("matrix multiplying... (also extremely inefficient)\n", file=stderr)
    print(create_haikus(1, mat, words, 0))
    return

def generate_text(n_words, words, transition_matrix, initial_state):
    text=words[initial_state]
    state=np.zeros((transition_matrix.shape[0], 1))
    i=initial_state
    for _ in range(n_words):
        state[:]=0
        state[i]=1
        state=np.matmul(transition_matrix, state)
        if (sum(state)==0):
            print("no connections from state=\"%s\"", words[i], file=stderr)
            i=random.randint(0,n_words)
        else:
            i = random.choices(range(0, len(state)), list(state))[0]
        text = " ".join([text, words[i]])
    return text

# returns state index
def next_stateindex(transition_matrix, current_state):
    v = np.matmul(transition_matrix, current_state)

    if(sum(map(abs,v)) <= 0): #placeholder
        return 0
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

def create_haikus(n_haikus, transition_matrix, words, initial_stateindex): 
    state=np.zeros((transition_matrix.shape[0], 1))
    i = initial_stateindex
    state[i]=1
    lines=[]
    lines.append([words[i]])
    mat_list = create_syllable_matrixes(transition_matrix, words)
    sylls_used = syllables.estimate(words[i])
    l=[5,7,5]
    for k in range(0,3):
        while(sylls_used < l[k]):
            j = i
            i = next_stateindex(mat_list[l[k]-1-sylls_used], state)
            sylls_used+=syllables.estimate(words[i])
            state[j]=0 #state[:]=0
            state[i]=1
            lines[k].append(words[i])
        lines[k].append("\n")
        lines.append([])
        if(sylls_used != l[k]): #WARNING: PLACEHOLDER. This is if we get an unconnected word which makes Alice to long of a word
            return create_haikus(n_haikus, transition_matrix, words, initial_stateindex)
        sylls_used=0

    lines = map(lambda l: " ".join(l), lines)
    return "".join(lines)
    

# get indices of words which have too many syllables
def unacceptable_indices(max_syllables, words):
    indices=[]
    for i in range(len(words)):
        if(syllables.estimate(words[i]) > max_syllables):
            indices.append(i)
    return indices
# This returns a matrix A where each column sums to one, assuming the word the column represents
# is not both unique and the last word (then it does not transform to a new word)
# Ax where x is a vector with some word gives us a new vector which is of the probabilities of each word
#
#TODO: we do not need d_new and c_new
def create_transition_matrix(n_words, corpus):
    corpus=corpus.split()
    corpus=corpus[:n_words]
    d = {}
    for w in corpus:
        d[w]=Counter()
    for i in range(1, len(corpus)):
        d[corpus[i-1]][corpus[i]]+=1
    d_new = {}
    for (key, c) in d.items():
        s = sum(c.values())
        c_new = Counter()
        for (w, n) in c.items():
            c_new[w] = n/s
        d_new[key]=c_new
    d=d_new

    uniques = list(dict.fromkeys(corpus))

    arr = np.empty((len(uniques), len(uniques)))
    xcount=0
    for x in uniques:
        ycount=0
        for y in uniques:
            arr[ycount,xcount] = d[x][y]
            ycount+=1
        xcount+=1
    return (uniques, arr)

main()
