# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

""" Information retreival"""

from nltk.tokenize import word_tokenize
import os
import math
from scipy import spatial

def docsPath(path):        # importing desired text files 
    os.chdir(path)
    docs = []               
    noOfDocs = 0
    
    for file in os.listdir():
        if file.endswith('.txt'):
            file_path = path + '/' + file
            f = open(file_path, 'r')
            docs.append(f.read())
            noOfDocs+=1
    
    return docs,noOfDocs

Docs, numOfDocs = docsPath('/Put your documents path Here') #

def tokenize_and_Casefolding(docs): 
        for index in   range(len(docs)):
            #Tokenization or 1-gram
            docs[index] = list(word_tokenize(docs[index].lower()))
            #lemmatizing words
            
        return docs
    
#Docs = preprocessDocs(Docs)

Docs = tokenize_and_Casefolding(Docs)

def uniqueTerms(docs):
    terms = []
    for docId in range(numOfDocs):
        for word in docs[docId]:
            terms.append(word)
    return terms
terms = uniqueTerms(Docs)
terms = set(terms)


def TF(docs,terms):
    TermF = {}
    
    for docId in range(numOfDocs):
        document = "Document " + str(docId + 1)
        TermF[document] = dict.fromkeys(terms,0) 
        
        for word in docs[docId]:
         TermF[document][word]= docs[docId].count(word) / len(docs[docId])
            
    return TermF

termFreq= TF(Docs,terms)

def documentFrequency(docs):
    DctFreq = {}
    
    for word in terms:
        DctFreq[word] = 0
        
        for docId in range(numOfDocs):
            document = "Document "+ str(docId + 1)
            
            try:
                DctFreq[word] += docs[docId].count(word)
            except (KeyError):
                 continue
            
    return DctFreq
dctFreq = documentFrequency(Docs)

def IDF(dctFreq, numOfDocs):
    idf_dict = {}
    
    for word in dctFreq:
        idf_dict[word] = math.log10(numOfDocs / dctFreq[word])
    return idf_dict
idf = IDF(dctFreq, numOfDocs)

def TFIDF(trmfrq, Idf):
    tfidf = {}
    for docId in trmfrq.keys():
        tfidf[docId] = {}
        for word in trmfrq[docId]:
            tfidf[docId][word] = trmfrq[docId][word] * Idf[word]
    return tfidf
tfidf = TFIDF(termFreq, idf)


def CosineSimiliarity(firstDocNumber, secondDocNumber,tfidf): 
    tfidf_Doc1 = []
    tfidf_Doc2 = []
    document1 = "Document " + str(firstDocNumber)
    document2 = "Document " + str(secondDocNumber)
    
    for value1 in tfidf[document1].values():
        tfidf_Doc1.append(value1)
        
    for value2 in tfidf[document2].values():
        tfidf_Doc2.append(value2)
        
    return 1 - spatial.distance.cosine(tfidf_Doc1, tfidf_Doc2)


print("Similarity between Selected Documents = "+ str(CosineSimiliarity(1, 2, tfidf)))
