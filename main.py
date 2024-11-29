#!/usr/bin/python
"""
The issue is that when we remove outputs with to many syllables
we may get a word which doesnt map to anything. I do not know how we 
should handle this.

Also, make it so that it remembers more previous states.
Currently hardcoded to only one previous state
"""
import syllables
#import pandas as pd
import numpy as np
import re
from random import choices
from collections import Counter

def main():
    """
    dat = pd.read_csv("elonmusk.csv", sep=",", header=0)
    dat=dat["Text"].apply(clean_text)
    dat = dat.loc[dat.apply(lambda x: bool(x.strip()))] #empty string evaluates to false
    dat = dat[:1000]
    print("joining...")
    corpus = " ".join(dat)
    print("joined")
    """
    f = open("alice_in_wonderland.txt", "r")
    corpus=f.read()
    f.close()
    print("generating matrix (extremely inefficient)")
    words, mat = create_transition_matrix(20000, corpus)
    print("generated matrix")
    initial_state = np.zeros((mat.shape[0], 1))
    initial_state[0]=1
    print(generate_text(90, words, mat, initial_state))
    return

def clean_text(text):
    # Remove URLs (http://, https://, www)
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove words starting with '@'
    text = re.sub(r'\s?@\w+', '', text)
    return remove_emojis(text)

def generate_text(nwords, words, transition_matrix, initial_state):
    text=""
    state=initial_state
    for _ in range(nwords):
        if (sum(state)==0):
            return text
        i = choices(range(0, len(state)), state)[0]
        text = " ".join([text, words[i]])
        #text = " ".join([text, choices(words, state)[0]])
        state[:]=0
        state[i]=1
        state=np.matmul(transition_matrix, state)
    return text

# Set some rows of matrix to 0 and reweight columns to 1
def mutate_matrix(mat, indices):
    for i in indices:
        mat[i, :] = 0
    for i in range(mat.shape[1]):
        colsum=sum(mat[:, i])
        if(colsum != 0):
            mat[:, i] = mat[:,i] / colsum

# get indices of words which have too many syllables
def unacceptable_indices(max_syllables, words):
    indices=[]
    for i in range(len(words)):
        if(syllables.estimate(words[i]) > max_syllables):
            indices.append(i)
    return indices


def remove_emojis(text):
    # Regex pattern to match emojis based on their Unicode ranges
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    return re.sub(emoji_pattern, '', text)

# This returns a matrix A where each column sums to one, assuming the word the column represents
# is not both unique and the last word (then it does not transform to a new word)
# Ax where x is a vector with some word gives us a new vector which is of the probabilities of each word
def create_transition_matrix(nwords, corpus):
    corpus=corpus.split()
    corpus=corpus[:nwords]
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
