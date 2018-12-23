# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 01:42:38 2018

@author: admin
"""
import gzip
import pandas as pd
import numpy as np
import nltk
import random
from nltk.corpus import sentiwordnet as swn

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def parse_gz(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def convert_to_DF(path):
    i = 0
    df = {}
    for d in parse_gz(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

baby = convert_to_DF('reviews_Baby_5.json.gz')
baby.count()
#format in time format
baby["reviewTime"] = pd.to_datetime(baby["reviewTime"])
baby.tail(3)

crit1 = baby['reviewTime'].map(lambda x : x.year == 2013)

baby1 = baby[crit1]
baby1.head(3)
baby1.count()
reviews = baby['reviewText']
reviews = list(reviews)
m = reviews[:5000] # Take first 5000 reviews
re = ''.join(m)

negationword = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']
stopword = nltk.corpus.stopwords.words('english')
newstopword = [word for word in stopword if word not in negationword]
newstopwor = [word for word in newstopword if not word.endswith("n't")]
len(newstopwor)


tre = nltk.word_tokenize(re)
noncha = [w for w in tre if w.isalpha()]
nonstopnewlowe = [w.lower() for w in noncha]
nonstopne = [w for w in nonstopnewlowe if w not in newstopwor]

wn= nltk.WordNetLemmatizer()
revie= [wn.lemmatize(w) for w in nonstopne]

# Find the positive and negative words from the file provided to us
text_po = nltk.pos_tag(set(revie))
t_positive = []
t_negative = []


for i in range(0, len(text_po)):
    word = text_po[i][0]
    tag = text_po[i][1]
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
            t_positive.append(word)
        elif (nscore > pscore and nscore > oscore):
            t_negative.append(word)

# Create a list of positive, negative and neutral sentences
sen = nltk.sent_tokenize(re)
posit = list()
negat = list()
neutr = list()
for sent in sen:
    p = 0
    n = 0
    g = nltk.word_tokenize(sent)
    for word in g:
        if word in t_positive:
            p = p + 1
        elif word in t_negative:
            n = n + 1
    if p > n:
        posit.append(sent)
    elif n > p:
        negat.append(sent)
    else:
        neutr.append(sent)

"""
po = ""        
for sent in posit:
    po = p + '\n' + sent
 
ne = ""    
for sent in negat:
    ne = ne + '\n' + sent

nut = ""
for sent in neutr:
    nut = nut + '\n' + sent



f = open("positive2.txt","w")
f.write(po)
f.close()

f = open("negative2.txt","w")
f.write(ne)
f.close()
        
f = open("neutral2.txt","w")
f.write(nut)
f.close()

"""
# Before Removing Stop Words

combined = ([(nltk.word_tokenize(sent),'pos') for sent in posit] + 
           [(nltk.word_tokenize(sent),'neg') for sent in negat])
random.shuffle(combined)

all_words_lis = [word for (sent,cat) in combined for word in sent]
all_word = nltk.FreqDist(all_words_lis)
word_item = all_word.most_common(2000)
word_features = [word for (word, freq) in word_item]

def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

trai = int(len(combined)*0.5)
featureset = [(document_features(d,word_features), c) for (d,c) in combined]
train_se, test_se = featureset[trai:], featureset[:trai]
classifie = nltk.NaiveBayesClassifier.train(train_se)
print (nltk.classify.accuracy(classifie, test_se)) #85.9// # 0.93319

from nltk.metrics import *

reflis = []
testlis = []
for (features, label) in test_se:
    reflis.append(label)
    testlis.append(classifie.classify(features))      

cm = ConfusionMatrix(reflis, testlis)
print(cm)
     
refne = set()
refpo = set()
testne = set()
testpo = set()

for i, label in enumerate(reflis):
    if label == 'neg': refne.add(i)
    if label == 'pos': refpo.add(i)

for i, label in enumerate(testlis):
    if label == 'neg': testne.add(i)
    if label == 'pos': testpo.add(i)

def printmeasures(label, refset, testset):
    print(label, 'precision:', precision(refset, testset))
    print(label, 'recall:', recall(refset, testset)) 
    print(label, 'F-measure:', f_measure(refset, testset))

printmeasures('neg', refne, testne)
printmeasures('pos', refpo, testpo)

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

NOT_featureset = [(NOT_features(d, word_features, negationword), c) for (d, c) in combined]      
train_se, test_se = NOT_featureset[trai:], NOT_featureset[:trai]
classifie4 = nltk.NaiveBayesClassifier.train(train_se)
print('Testing Accuracy: ',nltk.classify.accuracy(classifie4, test_se))

reflis = []
testlis = []
for (features, label) in test_se:
    reflis.append(label)
    testlis.append(classifie4.classify(features))      

cm = ConfusionMatrix(reflis, testlis)
print(cm)

refne = set()
refpo = set()
testne = set()
testpo = set()

for i, label in enumerate(reflis):
    if label == 'neg': refne.add(i)
    if label == 'pos': refpo.add(i)

for i, label in enumerate(testlis):
    if label == 'neg': testne.add(i)
    if label == 'pos': testpo.add(i)

printmeasures('neg', refne, testne)
printmeasures('pos', refpo, testpo)

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

SL_featureset = [(SL_features(d, word_features, SL), c) for (d,c) in combined]
train_se, test_se = SL_featureset[trai:], SL_featureset[:trai]
classifie5 = nltk.NaiveBayesClassifier.train(train_se)
print(nltk.classify.accuracy(classifie5, test_se)) #0.939341

reflis = []
testlis = []
for (features, label) in test_se:
    reflis.append(label)
    testlis.append(classifie5.classify(features))      

cm = ConfusionMatrix(reflis, testlis)
print(cm)

refne = set()
refpo = set()
testne = set()
testpo = set()

for i, label in enumerate(reflis):
    if label == 'neg': refne.add(i)
    if label == 'pos': refpo.add(i)

for i, label in enumerate(testlis):
    if label == 'neg': testne.add(i)
    if label == 'pos': testpo.add(i)

printmeasures('neg', refne, testne)
printmeasures('pos', refpo, testpo)

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

NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in combine]      
train_set, test_set = NOT_featuresets[train:], NOT_featuresets[:train]
classifier4 = nltk.NaiveBayesClassifier.train(train_set)
print('Testing Accuracy: ',nltk.classify.accuracy(classifier4, test_set))
      
SL_featuresets = [(SL_features(d, word_features, SL), c) for (d,c) in combine]
train_set, test_set = SL_featuresets[train:], SL_featuresets[:train]
classifier5 = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier5, test_set))


baby1['pos_neg'] = [1 if x > 3 else 0 for x in baby1.overall]         
         
x_train, x_test, y_train, y_test = train_test_split(baby1.reviewText, baby1.pos_neg, random_state=0)

# Vectorize X_train
vectorizer = CountVectorizer(min_df=5).fit(x_train)
X_train = vectorizer.transform(x_train)

feature_names = vectorizer.get_feature_names()

## Logistic Regression
scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
print("Cross Validation Score: {:.3f}".format(np.mean(scores)))

logreg = LogisticRegression(C=0.1).fit(X_train, y_train)

X_test = vectorizer.transform(x_test)

log_y_pred = logreg.predict(X_test)

logreg_score = accuracy_score(y_test, log_y_pred)
print("Accuracy:   {:.3f}".format(logreg_score))


## Multinomial Bayes
mnb = MultinomialNB(alpha=.01)
mnb.fit(X_train, y_train)

mnb_y_pred = mnb.predict(X_test)

mnb_score = accuracy_score(y_test, mnb_y_pred)
print("Accuracy:   {:.3f}".format(mnb_score))

