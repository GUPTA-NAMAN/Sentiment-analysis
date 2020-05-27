#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[ ]:


def stanford_train() :
    top_dir = os.getcwd()
    os.chdir("aclImdb/train")
    os.chdir('neg')
    x=[]
    y=[]
    for text in os.listdir() :
     if not text == '.DS_Store' :
        fo = open(text) 
        data=fo.read()
        x.append(data)
        y.append(0)
    print("number of negative reviews ",len(x))
    os.chdir('..')
    os.chdir('pos')
    numb=0
    for text in os.listdir() :
     if not text == '.DS_Store' :
        fo = open(text) 
        data=fo.read()
        x.append(data)
        y.append(1)
        numb=numb+1
    print('positive review s',numb)
    os.chdir( top_dir )
    return [x,y,2]



def stanford_test() :
    top_dir = os.getcwd()
    os.chdir("aclImdb/test")
    os.chdir('neg')
    x=[]
    y=[]
    for text in os.listdir() :
        if not text == '.DS_Store' :
            fo = open(text)
        data=fo.read()
        x.append(data)
        y.append(0)
    print("number of negative reviews ",len(x))
    os.chdir('..')
    os.chdir('pos')
    numb=0
    for text in os.listdir() :
        if not text == '.DS_Store' :
            fo = open(text)
            data=fo.read()
            x.append(data)
            y.append(1)
            numb=numb+1
    print('positive review s',numb)
    os.chdir( top_dir )
    return [x,y,2]

