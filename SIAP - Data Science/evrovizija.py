#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import string

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[2]:


import nltk

#Uncomment these for the first run only!
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

#Helper function to translate tags
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
    
#This function splits the text of lyrics to single words, does lemmatization for each, and removes stopwords & punctuation
def predprocess(document):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(document.lower()))
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    stop_words = set(stopwords.words('english'))
    exclude = set(string.punctuation + '’')
    stopwordRemoval = " ".join([i for i in lemmatized_sentence if i not in stop_words])
    punctuationRemoval = ''.join(ch for ch in stopwordRemoval if ch not in exclude)
    return punctuationRemoval.split()

#Returns a list with only one ofs
def unique(list):
     unique_list = []
     for x in list:
         if x not in unique_list:
              unique_list.append(x)
     return unique_list

def mostCommon(textList, count):
    if count==0:
        return " ".join(textList)
    
    wordCounter = {}
    for word in textList:
        if word in wordCounter:
            wordCounter[word] += 1
        else:
            wordCounter[word] = 1
        popularWords = sorted(wordCounter, key = wordCounter.get, reverse = True)
    return " ".join(popularWords[:count])


# In[3]:


df = pd.read_json('eurovision-lyrics.json')


# In[4]:


#Transponed table to make more sense
df = df.T


# In[5]:


#Remove new lines and commas
df['Lyrics'] = df['Lyrics'].str.replace("\n"," ")
df['Lyrics translation'] = df['Lyrics translation'].str.replace("\n"," ")


# In[6]:


#Move lyrics value from 'Lyrics' to 'Translated lyrics' for songs in English
df.loc[df['Lyrics translation'] == 'English', 'Lyrics translation'] = df.loc[df['Lyrics translation'] == 'English', 'Lyrics']


# In[7]:


#Remove songs in imaginaray language with no translation
df.drop(df.loc[df['Lyrics translation'] == ''].index, inplace=True)


# In[8]:


#Reduce each song to main language value only
df['Language'] = df['Language'].str.split('/').str[0]


# In[9]:


#Remove songs that dont have values in Place fields
df.drop(df.loc[df['Pl.']=='-'].index, inplace=True)


# In[10]:


#Convert year and placement fields to numbers
df['Year'] = df['Year'].astype(int)
df['Pl.'] = df['Pl.'].astype(int)


# In[11]:


#Remove irelevant columns
df.drop('#', axis=1, inplace=True)
df.drop('#.1', axis=1, inplace=True)
df.drop('Artist', axis=1, inplace=True)
df.drop('Sc.', axis=1, inplace=True)
df.drop('Eurovision_Number', axis=1, inplace=True)
df.drop('Host_City', axis=1, inplace=True)
df.drop('Lyrics', axis=1, inplace=True)


# In[12]:


#Create filtered list with option to set amount of most common words eg. 5 most common, pass 0 to use all words
df['Words'] = df['Lyrics translation'].apply(lambda x: predprocess(x)).apply(lambda x: mostCommon(x,0))


# In[13]:


df['Pl.'] = df['Pl.'].apply(lambda x: 1 if x <= 10 else 0)


# In[14]:


#Rename columns
df.rename({'Pl.': 'Placement', 'Host_Country': 'Host', 'Lyrics translation': 'Lyrics'}, axis=1, inplace=True)


# In[15]:


#Reset index and print sample data
df.reset_index(drop=True, inplace=True)
print(df.isnull().sum())
df


# In[75]:


#Machine learning usage to test predictions of models

dfNew = df[['Words','Placement']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( df['Words'], df['Placement'], test_size=0.2, random_state=40)

#tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='word' , stop_words='english',)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

#bag of words
#from sklearn.feature_extraction.text import CountVectorizer
#vectorizer = CountVectorizer()
#X_train = vectorizer.fit_transform(X_train)
#X_test = vectorizer.transform(X_test)

#Naive Bayes model
from sklearn.naive_bayes import GaussianNB
ml = GaussianNB()
ml.fit(X_train.toarray(), y_train)
predictions = ml.predict(X_test.toarray())

#LogisticRegression model
#from sklearn.linear_model import LogisticRegression
#ml = LogisticRegression(solver='lbfgs', max_iter=400)
#ml.fit(X_train,y_train)
#predictions = ml.predict(X_test)

#Random Forest Model
#from sklearn.ensemble import RandomForestClassifier
##Create a Gaussian Classifier
#clf=RandomForestClassifier(n_estimators=100)
##Train the model using the training sets y_pred=clf.predict(X_test)
#clf.fit(X_train,y_train)
#predictions=clf.predict(X_test)


from sklearn.metrics import confusion_matrix,classification_report
new = np.asarray(y_test)
confusion_matrix(y_test,predictions)
print(classification_report(y_test,predictions))


# In[ ]:




