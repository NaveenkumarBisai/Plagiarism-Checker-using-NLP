#!/usr/bin/env python
# coding: utf-8

# In[1]:


#all imported packages
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet


# In[52]:


#all the functions and declarations

#object creation for stemming
porter = PorterStemmer()

#to check for nouns
is_noun = lambda pos: pos[:2] == 'NN'

###################################################pre-processing steps#################################################


#sentence_tokenization
def sen_tokenize(text1):
    text = text1.read()
    sen_list = nltk.tokenize.sent_tokenize(text)
    return sen_list

#word_tokenization
def wor_tokenize(sen_list):  
    word_list = []
    for terms in sen_list:
        text_tokens = nltk.tokenize.word_tokenize(terms)
        word_list.append(text_tokens)
    return word_list


#to remove the stop words      
def stopword_removal(a_list):
    word_list = []
    for terms in a_list:
        tokens_without_sw = [word for word in terms if not word in stopwords.words()]
        word_list.append(tokens_without_sw)
    return word_list


#to remove the punctuation
def remove_punctuation(a_list):
    word_list = []
    for terms in a_list:
        tokens_without_sw = [word for word in terms if word.isalnum()]
        word_list.append(tokens_without_sw)
    return word_list


#concept extraction: to extract the important features like nouns in our case
def concept_extraction(a_list):
    b_list=[]
    
    for terms in a_list:
        #extracting nouns
        nouns = [word for (word, pos) in nltk.pos_tag(terms) if is_noun(pos)] 
        tokens_without_sw1=[]
        for word in nouns:
            #stemming
            tokens_without_sw1.append(porter.stem(word))
            #adding synonyms to the list so that it can catch alternative words used
            synset=wordnet.synsets(word)
            tokens_without_sw1.append(synset[0].lemmas()[0].name())
        b_list.append(tokens_without_sw1)
        #print(b_list)
    return b_list

#creating topic signature node
def get_topic_signature(b_list):
    topic_signature = list(set().union(*b_list))
    return topic_signature


# In[71]:


######################processing original document#############################

text1 = open("original.txt","r")

print("Original Document")

b_list=sen_tokenize(text1)
print("After Sentence Tokenization")
print(b_list)
print()

b_list=wor_tokenize(b_list)
print("After Word Tokenization")
print(b_list)
print()

b_list=stopword_removal(b_list)
print("After removing stop words")
print(b_list)
print()

b_list=remove_punctuation(b_list)
print("After removing punctuations")
print(b_list)
print()

b_list=concept_extraction(b_list)
print("The concepts list extracted from original document")
print(b_list)
print()

ts_1=get_topic_signature(b_list)
print("Topic signature of original document")
print(ts_1)


# In[72]:


######################processing plagiarised document#############################


text2 = open("plagiarised.txt","r")

print("Suspected Document")
c_list=sen_tokenize(text2)
print("After Sentence Tokenization")
print(c_list)
print()

c_list=wor_tokenize(c_list)
print("After Word Tokenization")
print(c_list)
print()

c_list=stopword_removal(c_list)
print("After removing stop words")
print(c_list)
print()

c_list=remove_punctuation(c_list)
print("After removing punctuations")
print(c_list)
print()

c_list=concept_extraction(c_list)
print("The concepts list extracted from suspected document")
print(c_list)
print()

ts_2=get_topic_signature(c_list)
print("Topic signature of suspected document")
print(ts_2)


# In[74]:


plag_percentage = len(list(set(ts_1) & set(ts_2)))/len(list(set(ts_1).union(set(ts_2))))*100
print("percentage of plagiarism found",plag_percentage)
print()
print("The copied concepts are")
print(list(set(ts_1) & set(ts_2)))


# In[80]:


#setting the tolerance level of plagiarism
tolerance=30
copied_content = list(set(ts_1) & set(ts_2))

#function for localization of copied content within the document
def localization(sen_list):    
    copied_sen = []
    print("plagiarism percentage of each sentence from start")
    for lists in sen_list:
        plag_percentage1 = len(list(set(copied_content) & set(lists)))/len(list(set(copied_content).union(set(lists))))*100
        print(" ",plag_percentage1)
        if plag_percentage1 > tolerance:
            copied_sen.append(sen_list.index(lists)+1)
    return copied_sen


# In[81]:


#calling localization for original document
copied_sen1 = localization(b_list)
print(copied_sen1,"sentences are copied with high accuracy from original document")

#calling localization for suspected document
copied_sen2 = localization(c_list)
print(copied_sen2,"sentences are copied with high accuracy in suspected document")


# In[82]:


print("The most copied sentences are",list(set(copied_sen1) & set(copied_sen2)))


# In[ ]:




