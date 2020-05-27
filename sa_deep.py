#!/usr/bin/env python
# coding: utf-8

# In[36]:


import gensim
import nltk
import numpy
import tensorflow
import torch
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.models import *
from keras.layers import *
from keras.callbacks import *
import random
import matplotlib.pyplot as plt 
from keras.models import Model
from keras.layers import Dense, Input, LSTM, GRU, Conv1D, MaxPooling1D, Concatenate  ,SimpleRNN
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.models import Sequential 
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


def split_train_val(x,y,ratio) :
    in_out = []
    
    p_xy = [] 
    n_xy=[]
    
    for i in range( len(x) ) :
        if y[i]== 0 :
            n_xy.append( [x[i],y[i]] )
        else :
            p_xy.append( [ x[i] , y[i] ] )
    
    
    print("debugging meassage -   negative ",len(n_xy)  , " postivie ",len(p_xy))
    
  
    a=random.shuffle( p_xy )
    b=random.shuffle( n_xy )
    print("debugging message  types ", type(a), type(b))
    
    x_val = []
    y_val = []
    x_train = []
    y_train = []
    
    for i in range( len(n_xy )) :
        if i < int(len(n_xy)*ratio) :
            x_train.append( n_xy[i][0] )
            y_train.append( n_xy[i][1] )
        else :
            x_val.append( n_xy[i][0] )
            y_val.append( n_xy[i][1] )
    
    for i in range( len(p_xy) ) :
        if i < int( len(p_xy) * ratio ) :
            x_train.append( p_xy[i][0] )
            y_train.append( p_xy[i][1] )
        else :
            x_val.append( p_xy[i][0] )
            y_val.append( p_xy[i][1] )
    
    
    return [x_train,y_train,x_val,y_val]


# In[38]:


def  visualize( x , y ) :
    
    t=150
    d=20
    
    frq_n = numpy.zeros( [t] ) 
    frq_p = numpy.zeros( [t] )
    frq_t = numpy.zeros( [t] )
    
    
    
    length_list=[]
    lens=d
    
    while(lens<=d*t) :
        length_list.append( lens )
        lens = lens + d
    n_p=0
    n_n=0
    print(len(length_list))
    for i in range( len(x) ) :
        length = len( x[i] )
        index = int(length/d)
        if y[i]== 0  :
            n_n=n_n+1
            frq_n[index] = frq_n[index]+1
        else :
            n_p=n_p+1
            frq_p[index] = frq_p[index]+1
            
    frq_t = frq_p + frq_n
         
    sum_n=0
    sum_p=0
    print("positive sample ",n_p," negative sample ",n_n)
    for i in range(t) :
        sum_n = sum_n + frq_n[i]
        sum_p = sum_p + frq_p[i]
        print(  i,length_list[i]  , sum_n/n_n , sum_p/n_p ,( sum_n + sum_p )/(n_n+n_p) )
    
    plt.plot( length_list , frq_p , color= 'red' )
    plt.plot( length_list , frq_n , color = 'green' )
    plt.plot( length_list , frq_t , color = 'blue' )
    
    plt.show()
        
    return [ frq_n , frq_p , frq_t ]


# loading pretrained word2vec  ' trained by negative subsampling '

# In[39]:


word2vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  


# getting words for which embedding is present

# In[40]:


vocab_word2vec = word2vec.wv.vocab.keys()


# In[41]:


from loader import stanford_train , stanford_test
print("details about training and validation sets combined")
[x_train_val,y_train_val,c] = stanford_train()
print("details on test set")
[x_test,y_test,c] =  stanford_test()


# In[42]:


all_data = x_train_val + x_test


# In[43]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(all_data))


# In[44]:


tokenizer.word_index


# In[45]:


size_of_vocabulary_dataset=len(tokenizer.word_index) + 1 #+1 for padding
print(size_of_vocabulary_dataset)


# In[46]:


embedding_matrix = numpy.zeros(( size_of_vocabulary_dataset , 300))


# In[47]:


for word, i in tokenizer.word_index.items():
    #print(i,word)
    if word in vocab_word2vec :
        embedding_matrix[i] =  word2vec[word]
        


# pre-pairing dataset for training , validation and testing set

# In[48]:


[x_train,y_train,x_val,y_val] = split_train_val( x_train_val , y_train_val , 0.7 )


# In[49]:


x_train_sq = tokenizer.texts_to_sequences( x_train )
x_val_sq = tokenizer.texts_to_sequences( x_val )
x_test_sq = tokenizer.texts_to_sequences( x_test )


# In[50]:


all_train_test_val = []
for i in range( len(x_train_sq) ) :
    all_train_test_val.append( len(x_train_sq[i]) )
for i in range( len(x_val_sq) ) :
    all_train_test_val.append( len(x_val_sq[i]) )
for i in range( len(x_test_sq) ) :
    all_train_test_val.append( len(x_test_sq[i]) )
maximum = max(all_train_test_val)


# In[51]:


maximum


# In[52]:


[frq_train_n,frq_train_p,frq_train_t] = visualize(x_train_sq, y_train )


# In[53]:


[frq_train_n,frq_train_p,frq_train_t] = visualize(x_val_sq, y_val )


# In[54]:


[frq_train_n,frq_train_p,frq_train_t] = visualize(x_test_sq, y_test )


# In[55]:


for i in range(len(y_train)) :
    y_train[i] = float(y_train[i])
for i in range(len(y_val)) :
    y_val[i] = float(y_val[i])


# In[56]:


threshold = 1000
x_train_sq_pd = pad_sequences( x_train_sq , maxlen = threshold )
x_val_sq_pd = pad_sequences( x_val_sq , maxlen = threshold )
x_test_sq_pd = pad_sequences( x_test_sq , maxlen = threshold)


# # shallow vanila RNN

# In[22]:


model = Sequential()
model.add(Embedding(size_of_vocabulary_dataset,300,weights=[embedding_matrix],input_length= threshold ,trainable=False)) 
model.add( SimpleRNN ( 64 , activation = 'sigmoid' ) )
model.add( Dense( 1 , activation = 'sigmoid' ) )

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'] )

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)  
mc=ModelCheckpoint('best_model_shallow_VRNN.h', monitor='val_acc', mode='max', save_best_only=True,verbose=1)  

#Print summary of model
print(model.summary())


# In[ ]:


history = model.fit(numpy.array(x_train_sq_pd),numpy.array(y_train),batch_size=128,epochs=10,validation_data=(numpy.array(x_val_sq_pd),numpy.array(y_val)),verbose=1,callbacks=[es,mc])


# In[ ]:


model = load_model('best_model_shallow_VRNN.h')

#evaluation 
_,train_acc = model.evaluate(x_train_sq_pd,y_train, batch_size=128)
_,val_acc = model.evaluate( x_val_sq_pd , y_val , batch_size = 128 )
_,test_acc = model.evaluate( x_test_sq_pd , y_test , batch_size=128 )
print("train acc ",test_acc)
print("val acc ",val_acc)
print("test acc ",test_acc)


# # DEEP VANILA RNN

# In[ ]:


model = Sequential()
model.add(Embedding(size_of_vocabulary_dataset,300,weights=[embedding_matrix],input_length= threshold ,trainable=False)) 
model.add( SimpleRNN ( 64 , activation = 'tanh' , retur_sequences = True ) )
model.add( SimpleRNN(64,activation = 'tanh') , return_sequences = False )

model.add( Dense(32,activation = 'sigmoid' ) )
model.add( Dense( 1 , activation = 'sigmoid' ) )

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'] )

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)  
mc=ModelCheckpoint('best_model_deep_VRNN.h', monitor='val_acc', mode='max', save_best_only=True,verbose=1)  

#Print summary of model
print(model.summary())



# In[ ]:


history = model.fit(numpy.array(x_train_sq_pd),numpy.array(y_train),batch_size=128,epochs=10,validation_data=(numpy.array(x_val_sq_pd),numpy.array(y_val)),verbose=1,callbacks=[es,mc])


# In[ ]:


model = load_model('best_model_deep_VRNN.h')

#evaluation 
_,train_acc = model.evaluate(x_train_sq_pd,y_train, batch_size=128)
_,val_acc = model.evaluate( x_val_sq_pd , y_val , batch_size = 128 )
_,test_acc = model.evaluate( x_test_sq_pd , y_test , batch_size=128 )
print("train acc ",train_acc)
print("val acc ",val_acc)
print("test acc ",test_acc)


# # shallow GRU 

# In[164]:


model = Sequential()
model.add(Embedding(size_of_vocabulary_dataset,300,weights=[embedding_matrix],input_length= threshold ,trainable=False)) 

model.add(GRU(64,return_sequences=False,dropout=0.2))
#model.add(GlobalMaxPooling1D())

#model.add(Dense(64,activation='relu')) 
model.add(Dense(1,activation='softmax')) 



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'] )

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)  
mc=ModelCheckpoint('best_model_shallow_GRU.h', monitor='val_acc', mode='max', save_best_only=True,verbose=1)  

#Print summary of model
print(model.summary())


# In[166]:


history = model.fit(numpy.array(x_train_sq_pd),numpy.array(y_train),batch_size=128,epochs=10,validation_data=(numpy.array(x_val_sq_pd),numpy.array(y_val)),verbose=1,callbacks=[es,mc])


# In[34]:


model = load_model('best_model_shallow_GRU.h')

#evaluation 
_,train_acc = model.evaluate(x_train_sq_pd,y_train, batch_size=128)
_,val_acc = model.evaluate( x_val_sq_pd , y_val , batch_size = 128 )
_,test_acc = model.evaluate( x_test_sq_pd , y_test , batch_size=128 )
print("train acc ",train_acc)
print("val acc ",val_acc)
print("test acc ",test_acc)


# In[35]:


train_acc


# # deep GRU

# In[59]:


model = Sequential()
model.add(Embedding(size_of_vocabulary_dataset,300,weights=[embedding_matrix],input_length= threshold ,trainable=False)) 

model.add(GRU(64,return_sequences=True,dropout=0.2))
model.add ( GRU(64,return_sequences=False, dropout=0.2) )

model.add(Dense(32,activation='softmax')) 
model.add(Dense(1,activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'] )

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)  
mc=ModelCheckpoint('best_model_deep_GRU.h', monitor='val_acc', mode='max', save_best_only=True,verbose=1)  

print(model.summary())


# In[60]:


history = model.fit(numpy.array(x_train_sq_pd),numpy.array(y_train),batch_size=128,epochs=10,validation_data=(numpy.array(x_val_sq_pd),numpy.array(y_val)),verbose=1,callbacks=[es,mc])


# In[ ]:


model = load_model('best_model_deep_GRU.h')

#evaluation 
_,train_acc = model.evaluate(x_train_sq_pd,y_train, batch_size=128)
_,val_acc = model.evaluate( x_val_sq_pd , y_val , batch_size = 128 )
_,test_acc = model.evaluate( x_test_sq_pd , y_test , batch_size=128 )
print("train acc ",train_acc)
print("val acc ",val_acc)
print("test acc ",test_acc)


# # shallow LSTM

# In[27]:


model = Sequential()
model.add(Embedding(size_of_vocabulary_dataset,300,weights=[embedding_matrix],input_length= threshold ,trainable=False)) 

model.add ( LSTM(64,return_sequences=False) )

model.add(Dense(32,activation='sigmoid')) 
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'] )

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)  
mc=ModelCheckpoint('best_model_shallow_LSTM.h', monitor='val_acc', mode='max', save_best_only=True,verbose=1)  

print(model.summary())


# In[28]:


history = model.fit(numpy.array(x_train_sq_pd),numpy.array(y_train),batch_size=128,epochs=10,validation_data=(numpy.array(x_val_sq_pd),numpy.array(y_val)),verbose=1,callbacks=[es,mc])


# In[29]:


model = load_model('best_model_shallow_LSTM.h')

#evaluation 
_,train_acc = model.evaluate(x_train_sq_pd,y_train, batch_size=128)
_,val_acc = model.evaluate( x_val_sq_pd , y_val , batch_size = 128 )
_,test_acc = model.evaluate( x_test_sq_pd , y_test , batch_size=128 )
print("train acc ",train_acc)
print("val acc ",val_acc)
print("test acc ",test_acc)


# # deep LSTM

# In[57]:


model = Sequential()
model.add(Embedding(size_of_vocabulary_dataset,300,weights=[embedding_matrix],input_length= threshold ,trainable=False)) 

model.add ( LSTM(64,return_sequences=True) )
model.add( LSTM(64,return_sequences=False) )

model.add(Dense(32,activation='sigmoid')) 
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'] )

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)  
mc=ModelCheckpoint('best_model_deep_LSTM.h', monitor='val_acc', mode='max', save_best_only=True,verbose=1)  

print(model.summary())


# In[58]:


history = model.fit(numpy.array(x_train_sq_pd),numpy.array(y_train),batch_size=128,epochs=10,validation_data=(numpy.array(x_val_sq_pd),numpy.array(y_val)),verbose=1,callbacks=[es,mc])


# In[ ]:


model = load_model('best_model_deep_LSTM.h')

#evaluation 
_,train_acc = model.evaluate(x_train_sq_pd,y_train, batch_size=128)
_,val_acc = model.evaluate( x_val_sq_pd , y_val , batch_size = 128 )
_,test_acc = model.evaluate( x_test_sq_pd , y_test , batch_size=128 )
print("train acc ",train_acc)
print("val acc ",val_acc)
print("test acc ",test_acc)


# In[ ]:




