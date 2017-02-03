"""
This script is what created the dataset pickled.

1) You need to download this file and put it in the same directory as this file.
https://github.com/moses-smt/mosesdecoder/raw/master/scripts/tokenizer/tokenizer.perl . Give it execution permission.

2) Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/ and extract it in the current directory.

3) Then run this script.
"""

dataset_path='C:/HiwiModules/Lstm/twitter_1_6M/dataset/'
#C:/HiwiModules/Lstm/twitter_1_6M/dataset/
#C:/HiwiModules/Lstm/twitter_50thousand/
#C:/HiwiModules/TwitterNNetworkSmall/
#C:/HiwiModules/TwiterrForNNetwork/
#'C:/HiwiModules/ImdbDatasets/aclImdb'
#C:/HiwiModules/ImdbDatasets/aclImdbShort
#dataset_path='/Tmp/bastienf/aclImdb/'

import numpy
import cPickle as pkl

from collections import OrderedDict

import glob
import os

from subprocess import Popen, PIPE

# tokenizer.perl is from Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
# C:/HiwiModules/Scripts/NewTry/tokenizer.perl
# './tokenizer.perl'
tokenizer_cmd = ['perl', 'tokenizer.perl', '-l', 'en', '-q', '-']


def tokenize(sentences):

    print 'Tokenizing..',
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    tok_text, _ = tokenizer.communicate(text)
    toks = tok_text.split('\n')[:-1]    
    print 'Done'

    return toks


def build_dict(path):
    print(str(path))
    sentences = []
    currdir = os.getcwd()
    os.chdir('%s/pos/' % path)
    print(os.getcwd())
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            for line in f:
                sentences.append(line.strip())
            #print("AAAAA: "+f.readline().strip())
            #sentences.append(f.readline().strip())
            #sentences.append([line.rstrip() for line in f])
    os.chdir('%s/neg/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            for line in f:
                sentences.append(line.strip())
            #print("BBBBB: "+f.readline().strip())            
            #sentences.append(f.readline().strip())
    os.chdir(currdir)

    sentences = tokenize(sentences)

    print 'Building dictionary..',
    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            #print("word: "+str(w))
            if w not in wordcount:                
                wordcount[w] = 1
                #print("value: "+str(wordcount[w]))
            else:                
                wordcount[w] += 1
                #print("value: "+str(wordcount[w]))
            #print("key: "+str(w)+"value: "+str(wordcount[w]))

    counts = wordcount.values()
    #print("These are the counts: ")
    #print(counts)
    keys = wordcount.keys()
    #print("These are the keys: ")
    #print(keys)
    #{'all': 2, 'not': 1, ',':32, 'just':1}  key = word in the reviews and value the frequency within the reviews 

    #Sort all the values in a descendent way using the divide and conquer approach the quicksort method keys["alvaro",".","hi","this","is","the"]
    # a = [1,5,3,1,2,20]  applying the line below will get a_sor_[::-1] = [5,1,2,4,3,0]
    sorted_idx = numpy.argsort(counts)[::-1]

    worddict = dict()

    #input for enumerate a_sort = [5,1,2,4,3,0] output enumerate prints index and value [(0,5),(1,1),(2,2),(3,4),(4,3),(5,0)]
    #worddict  {"the":2, ".":3, "hi": 4 .... } and so on 
    for idx, ss in enumerate(sorted_idx):
        #print("idx: "+str(idx)+" ss: "+str(ss))
        #print("key: "+str(keys[ss])+" value: "+str(idx+2))
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK) this is because when there is no term it will assign 1

    print numpy.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict


def grab_data(path, dictionary):
    sentences = []
    currdir = os.getcwd()
    os.chdir(path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            for line in f:
                sentences.append(line.strip())
            #print("UUULL: "+str(f.readline().strip()))
            #sentences.append(f.readline().strip())
            #print("sentences 0 "+str(sentences))
    os.chdir(currdir)
    
    sentences = tokenize(sentences)
    #print("len sentences: "+str(len(sentences)))
    #print("sentences 0 "+str(sentences[0]))
    #print("sentences 1 "+str(sentences[1]))
    #For instance 2 reviews  therefore len sentences = 2 
    #sentence [0] = "test review movie 1"
    #sentence [1] = "test review movie 2"

    #each review is read and all the words that compose a review each word are check within the dictionary and retrieved the value of the dictionary

    seqs = [None] * len(sentences)
    #seqs of None the seqs has the same lenght as the sentence length
    #enumerate [(0, "test review movie 1"), (1, "test review movie 2")]
    for idx, ss in enumerate(sentences):
        #each element in the array represents a line of the review #split into spaces
        words = ss.strip().lower().split()
        #print("Idx: "+str(idx)+" words: "+ str(ss.strip().lower()))
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]
    #print("length of seqs: "+str(len(seqs)))
    #for i in seqs:
    #    print("aa"+str(i))

            
    return seqs


def main():
    # Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/
    path = dataset_path
    #Create dictionary from the train set
    dictionary = build_dict(os.path.join(path, 'train'))
    print(len(dictionary))    
    # print(dictionary.keys())
    # print(dictionary.values())
    # print("alvaro__")
    # print(dictionary['.'],dictionary['the'],dictionary[','] )   

    # train x = [reviews positives represeented each word to the dictionary]     + [reviews negatives]
    # train x = [[3,4,2,90], [6,7,8,45]]
    # train y = [1,0 ]

    
    train_x_pos = grab_data(path+'train/pos', dictionary)
    train_x_neg = grab_data(path+'train/neg', dictionary)
    train_x = train_x_pos + train_x_neg
    #print("AA: "+ str(len(train_x)))
    #print("Possitive: "+str(train_x[0])+" Negative: "+str(train_x[400001]))
    train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)
    #print("BB: "+ str(len(train_y)))
    #print("Possitive: "+str(train_y[0])+" Negative: "+ str(train_y[400001]))


    
    test_x_pos = grab_data(path+'test/pos', dictionary)
    test_x_neg = grab_data(path+'test/neg', dictionary)
    test_x = test_x_pos + test_x_neg
    test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

    # 
    #f = open('imdb.pkl', 'wb')
    f = open('twitter_1_6M.pkl', 'wb')
    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((test_x, test_y), f, -1)
    f.close()

    #f = open('imdb.dict.pkl', 'wb')
    f = open('twitter_1_6M.dict.pkl', 'wb')
    pkl.dump(dictionary, f, -1)
    f.close()
    
    
if __name__ == '__main__':
    main()