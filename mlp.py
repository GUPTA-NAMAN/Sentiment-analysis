#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import os
import matplotlib
import random
import pickle

import loader
import utils
import copy

                            # fraction of data need to be in training set


# In[2]:


def split(x,y,frac) :
    
    
    tr_x=[]
    tr_y=[]
    va_x=[]
    va_y=[]
    
    tr_xn=0
    tr_xp=0
    va_xn=0
    va_xp=0
    
    index=0
    
    for i in range(2) :
        index=0
        for j in range( len(x) ) :
            if y[j] == i :
                if index < int(frac*12500)  :
                    tr_x.append( x[j] )
                    tr_y.append( y[j] )
                    if i==0 :
                        tr_xn=tr_xn+1
                    else :
                        tr_xp=tr_xp+1
                else :
                    va_x.append( x[j] )
                    va_y.append( y[j] )
                    if i==0 :
                        va_xn = va_xn +1
                    else :
                        va_xp=va_xp+1
                index=index+1
    print( tr_xn,tr_xp,va_xn,va_xp )
    return [tr_x,tr_y,va_x,va_y]


# In[3]:


[x,y,utils.total_number_class] = loader.stanford_train()


# In[35]:


[x_test,y_test,t] = loader.stanford_test() 


# In[6]:


print(utils.total_number_class)


# In[7]:


import utils
import numpy


# In[8]:


frq = numpy.zeros( [ utils.total_number_class ] )


# In[9]:


for i in range( len(y) ) :
    frq[ y[i] ] = frq[ y[i] ] + 1


# In[10]:


[tr_x,tr_y,va_x,va_y] = split( x,y,0.8 )


# In[11]:


print(len(tr_x))


# In[12]:


train_x=[]
Model = []


# In[13]:


for sentiment in range( utils.total_number_class ) :
    sent = sentiment 
    size_train_set=0
    temp_train_x=[]
    for i in range( len( tr_x ) ) :
            if tr_y[i] == sent :
                print(i , sent)
                utils.Train_Model(tr_x[i],i,sent,Model)
                temp_train_x.append( i ) 
            
    train_x.append( temp_train_x )


# In[14]:


for i in range( utils.total_number_class ) :
    print( len(train_x[i])  )


# In[15]:


print(len(Model))


# In[16]:


ordered_feature_tf=feature_tf = utils.Sort_tfidf( Model )


# In[17]:


feature_tf


# In[18]:


ordered_feature_mi=feature_mi = utils.Feature_MI( Model , train_x) 


# In[19]:


feature_mi


# In[20]:


model = Model


# In[21]:


class dictionary_entery :

    def __init__(self,token) :
        self.token = token
        self.term_frq_list=[]
        self.docs_frq_list=[]
        self.Rank_tfidf = -1
        self.Rank_mi = -1
        self.last_acess = "NA"
        for i in range(5) :
            self.term_frq_list.append(0)
            self.docs_frq_list.append(0)


# In[22]:



size_of_feature = 500                 # number of top features used
selected_mi_feature = ordered_feature_mi[: size_of_feature ]    # selecting top features
selected_mi_feature.sort()                                      # sorting above selected feature on the basis of alphabatic oreder
selected_tf_feature = ordered_feature_tf[: size_of_feature ]    # same as above but with tf-idf feature
selected_tf_feature.sort()


# In[23]:


Reduced_Model_tf = []
Reduced_Model_mi = []


# In[24]:


print("reducing model-------")
for token_rank in range( size_of_feature ) :
    index = utils.Find_Index(model,selected_tf_feature[token_rank])
    Reduced_Model_tf.append( copy.deepcopy( model[index[1] ] ) )
    
    index = utils.Find_Index(model,selected_mi_feature[token_rank])
    Reduced_Model_mi.append( copy.deepcopy( model[index[1] ] ) )
    


# In[48]:


reduced_train_Data_tf = []
reduced_val_Data_tf = []
reduced_test_Data_tf =[]
reduced_train_Data_mi = []
reduced_val_Data_mi = []
reduced_test_Data_mi = []


# In[49]:


original_vector =[]
for i in range( size_of_feature ) :
    original_vector.append(0)


# In[50]:


for sentiment in range( utils.total_number_class ) :
    temp_data_tf = []
    temp_data_mi = []
    for l in range( len( tr_x ) ) :
        if   tr_y[l] == sentiment  :
            vector_tf = copy.deepcopy(original_vector)
            vector_mi = copy.deepcopy(original_vector)
            tokens = utils.Pre_process( tr_x[l] )
            for i in range( len(tokens) ) :
                tokens[i] = tokens[i].lower()
                
            for token in tokens :
                Ispst_index = utils.Find_Index(Reduced_Model_tf,token )
                if( Ispst_index[0] == 1 ) :
                    index = Reduced_Model_tf[ Ispst_index[1] ].Rank_tfidf - 1
                    vector_tf[ index ] = vector_tf[  index ]   + 1
                Ispst_index = utils.Find_Index(Reduced_Model_mi,token)
                if( Ispst_index[0] == 1 ) :
                    index = Reduced_Model_mi[ Ispst_index[1] ].Rank_mi - 1
                    vector_mi[ index ] = vector_mi[ index ]  + 1
            temp_data_tf.append(vector_tf)
            temp_data_mi.append(vector_mi)
                
            
    reduced_train_Data_tf.append(temp_data_tf)
    reduced_train_Data_mi.append(temp_data_mi)
    
    


# In[51]:


for i in range( len(reduced_train_Data_tf) ) :
    print(i , len(reduced_train_Data_tf[i]) )


# In[52]:


for sentiment in range( utils.total_number_class ) :
    temp_data_tf = []
    temp_data_mi = []
    for l in range( len( va_x ) ) :
        if   va_y[l] == sentiment  :
            vector_tf = copy.deepcopy(original_vector)
            vector_mi = copy.deepcopy(original_vector)
            tokens = utils.Pre_process( va_x[l] )
            for i in range( len(tokens) ) :
                tokens[i] = tokens[i].lower()
                
            for token in tokens :
                Ispst_index = utils.Find_Index(Reduced_Model_tf,token )
                if( Ispst_index[0] == 1 ) :
                    index = Reduced_Model_tf[ Ispst_index[1] ].Rank_tfidf - 1
                    vector_tf[ index ] = vector_tf[  index ]   + 1
                Ispst_index = utils.Find_Index(Reduced_Model_mi,token)
                if( Ispst_index[0] == 1 ) :
                    index = Reduced_Model_mi[ Ispst_index[1] ].Rank_mi - 1
                    vector_mi[ index ] = vector_mi[ index ]  + 1
            temp_data_tf.append(vector_tf)
            temp_data_mi.append(vector_mi)
                
            
    reduced_val_Data_tf.append(temp_data_tf)
    reduced_val_Data_mi.append(temp_data_mi)
    
    


# In[53]:


for i in range( len(reduced_val_Data_tf) ) :
    print(i , len(reduced_val_Data_tf[i]) )
print( len(reduced_val_Data_tf[i][0]) )


# In[54]:


for sentiment in range( utils.total_number_class ) :
    temp_data_tf = []
    temp_data_mi = []
    for l in range( len( x_test ) ) :
        if   y_test[l] == sentiment  :
            vector_tf = copy.deepcopy(original_vector)
            vector_mi = copy.deepcopy(original_vector)
            tokens = utils.Pre_process( x_test[l] )
            for i in range( len(tokens) ) :
                tokens[i] = tokens[i].lower()
                
            for token in tokens :
                Ispst_index = utils.Find_Index(Reduced_Model_tf,token )
                if( Ispst_index[0] == 1 ) :
                    index = Reduced_Model_tf[ Ispst_index[1] ].Rank_tfidf - 1
                    vector_tf[ index ] = vector_tf[  index ]   + 1
                Ispst_index = utils.Find_Index(Reduced_Model_mi,token)
                if( Ispst_index[0] == 1 ) :
                    index = Reduced_Model_mi[ Ispst_index[1] ].Rank_mi - 1
                    vector_mi[ index ] = vector_mi[ index ]  + 1
            temp_data_tf.append(vector_tf)
            temp_data_mi.append(vector_mi)
                
            
    reduced_test_Data_tf.append(temp_data_tf)
    reduced_test_Data_mi.append(temp_data_mi)
    
    


# In[55]:


for i in range( len(reduced_test_Data_tf) ) :
    print(i , len(reduced_test_Data_tf[i]) )
print( len(reduced_test_Data_tf[i][0]) )


# In[56]:



file_name_train_matrix_tf  = "train_matrix_tf.data"
file_name_train_matrix_mi  = "train_matrix_mi.data"
file_name_val_matrix_tf   = "val_matrix_tf.data"
file_name_val_matrix_mi   = "val_matrix_mi.data"
file_name_test_matrix_tf  = "test_matrix_tf.data"
file_name_test_matrix_mi = "test_matrix_mi.data"

file_name_reduced_model_tf = "reduced_model_tf.data"
file_name_reduced_model_mi = "reduced_model_mi.data"


# In[57]:


print("saving data")


# In[58]:


with open( file_name_train_matrix_tf,"wb") as file_obj :
  pickle.dump(reduced_train_Data_tf,file_obj)
with open( file_name_train_matrix_mi,"wb") as file_obj :
  pickle.dump(reduced_train_Data_mi,file_obj)
with open( file_name_val_matrix_tf,"wb") as file_obj :
  pickle.dump(reduced_val_Data_tf,file_obj)
with open( file_name_val_matrix_mi,"wb") as file_obj :
  pickle.dump(reduced_val_Data_mi,file_obj)
with open( file_name_test_matrix_tf,"wb") as file_obj :
  pickle.dump(reduced_test_Data_tf,file_obj)
with open( file_name_test_matrix_tf,"wb") as file_obj :
  pickle.dump(reduced_test_Data_mi,file_obj)
  
    
with open( file_name_reduced_model_tf,"wb") as file_obj :
  pickle.dump(Reduced_Model_tf,file_obj)

with open( file_name_reduced_model_mi,"wb") as file_obj :
  pickle.dump(Reduced_Model_mi,file_obj)


# In[59]:


print("result stored")


# In[60]:


import numpy


# In[61]:


len(reduced_train_Data_tf)


# In[112]:


def ny(ns) :
    l = numpy.zeros( [ns,2] )
    for i in range( ns ) :
        if i<int(ns/2) :
            l[i][0]=1 
        else :
            l[i][1] = 1
    return l


# In[113]:


def l_na(l) :
    ns = len(l[0])+len(l[1])
    print("number samples ",ns , " number feature ",len(l[0]),len(l[1]),len(l[0][0]))
    m = numpy.zeros( [ ns , len(l[0][0]) ] )
    return [m,ny(ns)]


# In[114]:


[nmtrx , ntry ]= l_na( reduced_train_Data_mi )

[nmvax , nvay] = l_na( reduced_val_Data_mi )
[nmtex , ntey] = l_na( reduced_test_Data_mi )


# In[146]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[153]:


model = Sequential()
model.add(Dense(500, input_dim=500, activation='sigmoid'))
model.add(Dense(400, activation='sigmoid'))
model.add(Dense(300, activation='sigmoid'))

model.add(Dense(2, activation='softmax'))


# In[ ]:





# In[154]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[155]:


model.summary()


# In[156]:


from keras.models import *
from keras.layers import *
from keras.callbacks import *


# In[164]:


history = model.fit( nmtrx ,ntry,batch_size=500,epochs=2,validation_data=(nmvax,nvay),verbose=1)


# In[169]:


sets = [ [nmtrx,ntry] , [nmvax,nvay] , [nmtex , ntey ] ]
for i in range(3) :
    _,acc=model.evaluate( sets[i][0] , sets[i][1] )
    print(acc)


# In[ ]:




