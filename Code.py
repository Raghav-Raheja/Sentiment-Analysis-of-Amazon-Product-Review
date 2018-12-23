# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:10:57 2018

@author: admin
"""
import re
import nltk
import pandas as pd
#import string
from nltk.corpus import stopwords
from nltk.corpus import sentence_polarity
from nltk.corpus import sentiwordnet as swn
#from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


f = open('baby.txt','r')
rawfile = f.read()
a = re.compile('[\s]{2,}')
rawfile = a.sub(" ",rawfile)
pword = re.compile('reviewText:\w+.*\w+')
g = re.findall(pword,rawfile)
g = [s.replace("reviewText:","") for s in g]
"""
pword2 = re.compile('reviewTime:[0-9]+\s[0-9]+\,\s[0-9]+')
g2 = re.findall(pword2,rawfile)
"""


g[:5]
len(g)
t = g[:5000] # Take first 5000 reviews
st = ''.join(t)
len(st)


star = nltk.sent_tokenize(st)
gstar = star[:30]
#sid = SentimentIntensityAnalyzer()

#negative = list()
#positive = list()
#for sentence in gstar:
#    print(sentence)
#    ss = sid.polarity_scores(sentence)
#    for k in sorted(ss):
#        print('{0}: {1}, '.format(k, ss[k]), end='')
#        print()

negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']
stopwords = nltk.corpus.stopwords.words('english')
newstopwords = [word for word in stopwords if word not in negationwords]
newstopword = [word for word in newstopwords if not word.endswith("n't")]
len(newstopword)


sar = nltk.word_tokenize(st)
nonchar = [w for w in sar if w.isalpha()]
nonstopnewlower = [w.lower() for w in nonchar]
nonstopnew = [w for w in nonstopnewlower if w not in newstopword]
#nonstoplower = [w.lower() for w in nonstopnew]

wnl= nltk.WordNetLemmatizer()
lw= [wnl.lemmatize(w) for w in nonstopnew]


# Find the positive and negative words from the file provided to us
text_pos = nltk.pos_tag(set(lw))
text_positive = []
text_negative = []


for i in range(0, len(text_pos)):
    word = text_pos[i][0]
    tag = text_pos[i][1]
    synlist = []
    if (tag == 'NN' or tag == 'NNS'):
        synlist = list(swn.senti_synsets(word, 'n'))
    elif (tag == 'VB' or tag == 'VBD' or tag == 'VBN' or tag == 'VBP' or tag == 'VBZ' or tag == 'VBG'):
        synlist = list(swn.senti_synsets(word, 'v'))
    elif (tag == 'JJ' or tag == 'JJR' or tag == 'JJS'):
        synlist = list(swn.senti_synsets(word, 'a'))
    elif (tag == 'RB' or tag == 'RBR' or tag == 'RBS'):
        synlist = list(swn.senti_synsets(word, 'r'))
    if (len(synlist) > 0):
        pscore = synlist[0].pos_score()
        nscore = synlist[0].neg_score()
        oscore = synlist[0].obj_score()
        if (pscore > nscore and pscore > oscore):
            text_positive.append(word)
        elif (nscore > pscore and nscore > oscore):
            text_negative.append(word)

combined = ([(name, 'pos') for name in text_positive] + [(name, 'neg') for name in text_negative])
random.shuffle(combined)


# Create a list of positive, negative and neutral sentences
posi = list()
nega = list()
neut = list()
for sent in star:
    p = 0
    n = 0
    ps = 0
    ns = 0
    g=[]
    g = nltk.word_tokenize(sent)
    for word in g:
        if word in text_positive:
#           synlist= list(swn.senti_synsets(word))
#            ps = ps + synlist[0].pos_score()
            p = p + 1
        elif word in text_negative:
#            synlist= list(swn.senti_synsets(word))
#            ns = ns + synlist[0].neg_score()
            n = n + 1
    if p > n:
        posi.append(sent)
    elif n > p:
        nega.append(sent)
    else:
        neut.append(sent)
"""
p = ""        
for sent in posi:
    p = p + '\n' + sent
 
n = ""    
for sent in nega:
    n = n + '\n' + sent

nu = ""
for sent in neut:
    nu = nu + '\n' + sent



f = open("positive.txt","w")
f.write(p)
f.close()

f = open("negative.txt","w")
f.write(n)
f.close()
        
f = open("neutral.txt","w")
f.write(nu)
f.close()
     
"""   
f = open('positive.txt','r')
ptive = f.read()
         
f = open('negative.txt','r')
ntive = f.read()

a = re.compile('[\s]{2,}')
ntive = a.sub(" ",ntive)

pstive = ''.join(ptive)
nstive = ''.join(ntive)

psent = nltk.sent_tokenize(pstive)


trial = psent[:5]

combine = ([(nltk.word_tokenize(sent),'pos') for sent in posi] + 
           [(nltk.word_tokenize(sent),'neg') for sent in nega])

random.shuffle(combine)
#nsent = nltk.sent_tokenize(nstive)


# Before Removing Stop Words
all_words_list = [word for (sent,cat) in combine for word in sent]
all_words = nltk.FreqDist(all_words_list)
word_items = all_words.most_common(2000)
word_features = [word for (word, freq) in word_items]

def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
train = int(len(combine)*0.5)
featuresets = [(document_features(d,word_features), c) for (d,c) in combine]
train_set, test_set = featuresets[train:], featuresets[:train]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print (nltk.classify.accuracy(classifier, test_set)) #85.9// # 0.93319

from nltk.metrics import *

reflist = []
testlist = []
for (features, label) in test_set:
    reflist.append(label)
    testlist.append(classifier.classify(features))


cm = ConfusionMatrix(reflist, testlist)
print(cm)
      
refneg = set()
refpos = set()
testneg = set()
testpos = set()

for i, label in enumerate(reflist):
    if label == 'neg': refneg.add(i)
    if label == 'pos': refpos.add(i)

for i, label in enumerate(testlist):
    if label == 'neg': testneg.add(i)
    if label == 'pos': testpos.add(i)

def printmeasures(label, refset, testset):
    print(label, 'precision:', precision(refset, testset))
    print(label, 'recall:', recall(refset, testset)) 
    print(label, 'F-measure:', f_measure(refset, testset))

printmeasures('neg', refneg, testneg)
printmeasures('pos', refpos, testpos)


#Not Feature

def NOT_features(document, word_features, negationwords):
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = False
        features['contains(NOT{})'.format(word)] = False
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['contains(NOT{})'.format(document[i])] = (document[i] in word_features)
        else:
            features['contains({})'.format(word)] = (word in word_features)
    return features
      
NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in combine]      
train_set, test_set = NOT_featuresets[train:], NOT_featuresets[:train]
classifier4 = nltk.NaiveBayesClassifier.train(train_set)
print('Testing Accuracy: ',nltk.classify.accuracy(classifier4, test_set)) #0.88895

reflist = []
testlist = []
for (features, label) in test_set:
    reflist.append(label)
    testlist.append(classifier4.classify(features))


cm = ConfusionMatrix(reflist, testlist)
print(cm)
      
refneg = set()
refpos = set()
testneg = set()
testpos = set()

for i, label in enumerate(reflist):
    if label == 'neg': refneg.add(i)
    if label == 'pos': refpos.add(i)

for i, label in enumerate(testlist):
    if label == 'neg': testneg.add(i)
    if label == 'pos': testpos.add(i)

def printmeasures(label, refset, testset):
    print(label, 'precision:', precision(refset, testset))
    print(label, 'recall:', recall(refset, testset)) 
    print(label, 'F-measure:', f_measure(refset, testset))

printmeasures('neg', refneg, testneg)
printmeasures('pos', refpos, testpos)

## Subjectivity Features

def readSubjectivity(path):
    flexicon = open(path, 'r')
    # initialize an empty dictionary
    sldict = { }
    for line in flexicon:
        fields = line.split()   # default is to split on whitespace
        # split each field on the '=' and keep the second part as the value
        strength = fields[0].split("=")[1]
        word = fields[2].split("=")[1]
        posTag = fields[3].split("=")[1]
        stemmed = fields[4].split("=")[1]
        polarity = fields[5].split("=")[1]
        if (stemmed == 'y'):
            isStemmed = True
        else:
            isStemmed = False
        # put a dictionary entry with the word as the keyword
        #     and a list of the other values
        sldict[word] = [strength, posTag, isStemmed, polarity]
    return sldict

SLpath = 'subjclueslen1-HLTEMNLP05.tff'

SL = readSubjectivity(SLpath)

def SL_features(document, word_features, SL):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    # count variables for the 4 classes of subjectivity
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg)      
    return features

SL_featuresets = [(SL_features(d, word_features, SL), c) for (d,c) in combine]
train_set, test_set = SL_featuresets[train:], SL_featuresets[:train]
classifier5 = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier5, test_set)) #0.939341

reflist = []
testlist = []
for (features, label) in test_set:
    reflist.append(label)
    testlist.append(classifier4.classify(features))


cm = ConfusionMatrix(reflist, testlist)
print(cm)
      
refneg = set()
refpos = set()
testneg = set()
testpos = set()

for i, label in enumerate(reflist):
    if label == 'neg': refneg.add(i)
    if label == 'pos': refpos.add(i)

for i, label in enumerate(testlist):
    if label == 'neg': testneg.add(i)
    if label == 'pos': testpos.add(i)

printmeasures('neg', refneg, testneg)
printmeasures('pos', refpos, testpos)
     
#After Removing Stop Words

all_words_list = [word for (sent,cat) in combine for word in sent]
words = [w.lower() for w in all_words_list if w.isalpha()]
all_words_list = [word for word in words if word not in stopwords]
all_words = nltk.FreqDist(all_words_list)
word_items = all_words.most_common(2000)
word_features = [word for (word, freq) in word_items]

def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
train = int(len(combine)*0.5)
featuresets = [(document_features(d,word_features), c) for (d,c) in combine]
train_set, test_set = featuresets[train:], featuresets[:train]
classifier2 = nltk.NaiveBayesClassifier.train(train_set)
print (nltk.classify.accuracy(classifier2, test_set)) 

reflist = []
testlist = []
for (features, label) in test_set:
    reflist.append(label)
    testlist.append(classifier2.classify(features))


cm = ConfusionMatrix(reflist, testlist)
print(cm)
      
refneg = set()
refpos = set()
testneg = set()
testpos = set()

for i, label in enumerate(reflist):
    if label == 'neg': refneg.add(i)
    if label == 'pos': refpos.add(i)

for i, label in enumerate(testlist):
    if label == 'neg': testneg.add(i)
    if label == 'pos': testpos.add(i)

printmeasures('neg', refneg, testneg)
printmeasures('pos', refpos, testpos)

NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in combine]      
train_set, test_set = NOT_featuresets[train:], NOT_featuresets[:train]
classifier4 = nltk.NaiveBayesClassifier.train(train_set)
print('Testing Accuracy: ',nltk.classify.accuracy(classifier4, test_set)) 

reflist = []
testlist = []
for (features, label) in test_set:
    reflist.append(label)
    testlist.append(classifier4.classify(features))


cm = ConfusionMatrix(reflist, testlist)
print(cm)
      
refneg = set()
refpos = set()
testneg = set()
testpos = set()

for i, label in enumerate(reflist):
    if label == 'neg': refneg.add(i)
    if label == 'pos': refpos.add(i)

for i, label in enumerate(testlist):
    if label == 'neg': testneg.add(i)
    if label == 'pos': testpos.add(i)

printmeasures('neg', refneg, testneg)
printmeasures('pos', refpos, testpos)

      
SL_featuresets = [(SL_features(d, word_features, SL), c) for (d,c) in combine]
train_set, test_set = SL_featuresets[train:], SL_featuresets[:train]
classifier5 = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier5, test_set)) 

reflist = []
testlist = []
for (features, label) in test_set:
    reflist.append(label)
    testlist.append(classifier5.classify(features))


cm = ConfusionMatrix(reflist, testlist)
print(cm)
      
refneg = set()
refpos = set()
testneg = set()
testpos = set()

for i, label in enumerate(reflist):
    if label == 'neg': refneg.add(i)
    if label == 'pos': refpos.add(i)

for i, label in enumerate(testlist):
    if label == 'neg': testneg.add(i)
    if label == 'pos': testpos.add(i)

printmeasures('neg', refneg, testneg)
printmeasures('pos', refpos, testpos)

# Using movie review corpus and top words from amazon baby review
all_sent = nltk.word_tokenize(st)
all_sent_list = [nltk.word_tokenize(sent) for sent in all_sent]
words = [w.lower() for w in all_sent if w.isalpha()]
stopwords = nltk.corpus.stopwords.words('english')
all_words_list = [word for word in words if word not in stopwords]
all_words = nltk.FreqDist(all_words_list)
word_items = all_words.most_common(2000)
word_features = [word for (word, freq) in word_items]


      
sentences = sentence_polarity.sents()
documents = [(sent, cat) for cat in sentence_polarity.categories() for sent in sentence_polarity.sents(categories=cat)]

random.shuffle(documents)

featuresets = [(document_features(d,word_features), c) for (d,c) in documents]
traning = int(len(featuresets)*0.5)
train_set, test_set = featuresets[traning:], featuresets[:traning]
classifier3 = nltk.NaiveBayesClassifier.train(train_set)
print ('Testing Accuracy: ',nltk.classify.accuracy(classifier3, test_set)) # 0.63421// 0.6450

      
reflist = []
testlist = []
for (features, label) in test_set:
    reflist.append(label)
    testlist.append(classifier3.classify(features))


cm = ConfusionMatrix(reflist, testlist)
print(cm)
      
refneg = set()
refpos = set()
testneg = set()
testpos = set()

for i, label in enumerate(reflist):
    if label == 'neg': refneg.add(i)
    if label == 'pos': refpos.add(i)

for i, label in enumerate(testlist):
    if label == 'neg': testneg.add(i)
    if label == 'pos': testpos.add(i)

printmeasures('neg', refneg, testneg)
printmeasures('pos', refpos, testpos)  

NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in documents]      
train_set, test_set = NOT_featuresets[traning:], NOT_featuresets[:traning]
classifier0 = nltk.NaiveBayesClassifier.train(train_set)
print ('Testing Accuracy: ',nltk.classify.accuracy(classifier0, test_set))

reflist = []
testlist = []
for (features, label) in test_set:
    reflist.append(label)
    testlist.append(classifier0.classify(features))


cm = ConfusionMatrix(reflist, testlist)
print(cm)
      
refneg = set()
refpos = set()
testneg = set()
testpos = set()

for i, label in enumerate(reflist):
    if label == 'neg': refneg.add(i)
    if label == 'pos': refpos.add(i)

for i, label in enumerate(testlist):
    if label == 'neg': testneg.add(i)
    if label == 'pos': testpos.add(i)

printmeasures('neg', refneg, testneg)
printmeasures('pos', refpos, testpos)      

SL_featuresets = [(SL_features(d, word_features, SL), c) for (d,c) in documents]
train_set, test_set = SL_featuresets[traning:], SL_featuresets[:traning]
classifier8 = nltk.NaiveBayesClassifier.train(train_set)
print ('Testing Accuracy: ',nltk.classify.accuracy(classifier8, test_set)) 


sentences = nltk.sent_tokenize(st)
positive = []
negative = []

for sent in sentences:
   a = str() 
   a = classifier8.classify(SL_features(nltk.word_tokenize(sent),word_features, SL))      
   if a == 'pos':
       positive.append(sent)
   elif a == 'neg':
       negative.append(sent)
 """
neg = ""    
for sent in negative:
    neg = neg + '\n' + sent

pos = ""
for sent in positive:
    pos = pos + '\n' + sent   
    
f = open("positive_classify.txt","w")
f.write(pos)
f.close()

f = open("negative_classify.txt","w")
f.write(neg)
f.close()
  
"""

     
reflist = []
testlist = []
for (features, label) in test_set:
    reflist.append(label)
    testlist.append(classifier8.classify(features))


cm = ConfusionMatrix(reflist, testlist)
print(cm)
      
refneg = set()
refpos = set()
testneg = set()
testpos = set()

for i, label in enumerate(reflist):
    if label == 'neg': refneg.add(i)
    if label == 'pos': refpos.add(i)

for i, label in enumerate(testlist):
    if label == 'neg': testneg.add(i)
    if label == 'pos': testpos.add(i)

printmeasures('neg', refneg, testneg)
printmeasures('pos', refpos, testpos)

