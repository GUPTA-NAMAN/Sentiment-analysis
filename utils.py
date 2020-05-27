import os
import nltk
import copy
import math
import numpy
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

punction_marks=['`','~','!','@','#','$','%','^','&','*','(',')','-','_','+','=','{','[',']','}','|']
punction_marks = punction_marks + [']',';',':','"',"'",'<',',','>','.','?','/',' ']


lemmatizer = WordNetLemmatizer()
stopWords = set(stopwords.words('english'))

total_number_class = - 1



"""
token          :  string  -  used to store tokens
term_frq_list  :  list    -  its size is equal to utils.total_number_class , each entry stores the frequency of token in particular class . such that first stores for                                        directories[0](which is comp.graphics)  which is defined in file .used in calculating tf-idf score
docs_frq_list  :  list    -  stores number of files contains the token in class ,  as same manner in above.used in calculating mi score
Rank_tfidf     :  integer -  used to store rank of tf-idf ,means term with highest tf-idf will have 2 , second highest will have 2.
Rank_mi        :  integer -  same as above but for mi score .
last_access    :  string  -  used to store details about which file has accessed it while filling details about particular token . used to speed up calculation .
"""
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
     
     
#   this function find out all directories and files in current directories
#   It is used to get all files name and directories name also
def find_files_and_directories() :
        All =  os.listdir(os.getcwd())
        if('.DS_Store' in All ) :
            All.remove(".DS_Store")
        
        if('a1' in All) :
           All.remove("a1")
           All.remove("a2")
           All.remove("a3")
        
        return All


# takes list of strings and string as input does binary search , it string is present return a list first value 1( it means string is present ) second value is it's index.
# if string is not present return list with first value as -1 and second value a index where the string should be inserted without destroying its sorted order.
def Find_index(list,token) :

    lower = 0
    upper = len(list) - 1
    
    if( len(list) == 0 ) :
        return 0
        
    while(True) :
        mid = (int)( (lower+upper)/2 )
        index_token = list[mid]
        if( not ( lower == upper ) ) :
            if( index_token == token ) :
                return mid
            else :
                if ( index_token < token ) :
                    upper = mid - 1
                    if ( upper<0 ) :
                        upper = 0
                else :
                    lower = mid + 1
                    if(lower>upper) :
                        lower = upper
        else :
            if( index_token == token ) :
                return lower
            elif ( index_token < token ) :
                return lower
            else :
                return lower + 1


# it takes list of objects of dictionary entry and token(string )as input and do the same job as above function with dictionary.entry[index].token 
def Find_Index(Model,token) :
    
    lower = 0
    upper = len(Model)-1
    
    if(len(Model)==0) :
        return [-1,0]
    while(True) :
      mid = (int)((lower+upper)/2)
      index_token = Model[mid].token
      if ( not (lower  == upper) ) :
        if( index_token == token  ) :
            return [1,mid]
        else :
            if( index_token < token ) :
                lower = mid+1
                if (lower>upper) :
                    lower = upper
            else :
                upper = mid - 1
                if(upper<0) :
                    upper = 0
      else :
        if(index_token == token ) :
            return [1,lower]
        elif ( index_token <  token ) :
            return [-1,lower+1]
        else :
            return [-1,lower]
      
      
# enter token of a particular class into model and incease the contribution of class by one , if it is done by a new file incease docs frequncy by one
def Register_token_in_Model(Model,token,file,class_of_file) :

    index_in_Model = Find_Index(Model,token)
    if( index_in_Model[0] == -1 ) :
        temp_dictionary_entery = dictionary_entery(token)
        Model.insert( index_in_Model[1] , temp_dictionary_entery )
    Model[ index_in_Model[1] ].term_frq_list[ class_of_file ] = Model[ index_in_Model[1] ].term_frq_list[ class_of_file ]  + 1
    
    if( index_in_Model[0] == -1 ) :
        Model[ index_in_Model[1] ].last_acess =  file
        Model[ index_in_Model[1] ].docs_frq_list[ class_of_file ] = Model[ index_in_Model[1] ].docs_frq_list[ class_of_file ] + 1
    else :
        if ( not ( Model[ index_in_Model[1] ].last_acess == file )  ) :
            Model[ index_in_Model[1] ].last_acess =  file
            Model[ index_in_Model[1] ].docs_frq_list[ class_of_file ] = Model[ index_in_Model[1] ].docs_frq_list[ class_of_file ] + 1
        
   
            
            
# train the model
def Train_Model(train_text,text_id,sentiment,Model) :
    
             print( len(Model) )
             tokens =   Pre_process( train_text )
             for token in tokens :
                Register_token_in_Model(Model,token,text_id,sentiment)
        


# take file name as input and returns processed token as output
def Pre_process(train_text) :

    file_data = train_text
    
    file_data = file_data.lower()
   
    document_vector = []
    token = nltk.word_tokenize(file_data)
    
    
    temp_token = token
    token=[]
    for each in temp_token :
        if ( each not in punction_marks ) :
            token.append(each)
    
    temp_token=token
    token=[]
    for each in temp_token :
       lemmatize_token = lemmatizer.lemmatize(each)
       token.append(lemmatize_token)
    
    
    temp_token=token
    token=[]
    for each in temp_token :
        if each not in stopWords :
            token.append(each)
            
    return token
  
  
# take feature and class as input and returns MI score
def MI_score(Model,feature_id,class_id,Train_Data) :
   
   n = numpy.zeros( [2,2] )
   
   n[1][1] = Model[ feature_id ].docs_frq_list[ class_id ]
   n[0][1] = len( Train_Data[ class_id ] ) - n[1][1]
   n[1][0] = sum( Model[ feature_id ].docs_frq_list ) - n[1][1]
   all_docs = 0
   for i in range( len(Train_Data) ) :
    all_docs = all_docs + len( Train_Data[i] )
   n[0][0] = all_docs - len(Train_Data[ class_id ]) - n[1][0]
   #print(" mi score calculator ",feature_id,class_id)
   #print(n)
   #with open("matrix","wb") as f :
   # pickle.dump(n,f)
   score = 0
   if ( not n[1][1] == 0  ) :
     score = score +  ( n[1][1]/all_docs  )*math.log( (all_docs*n[1][1])/(  ( n[1][1] + n[0][1] )*( n[1][1] + n[1][0] )  )  )
   if ( not  n[0][1] == 0  ) :
     score = score +  ( n[0][1]/all_docs  )*math.log( (all_docs*n[0][1])/(  ( n[0][1] + n[0][0] )*( n[1][1] + n[0][1] )  )  )
   if ( not n[1][0] == 0 ) :
     score = score +  ( n[1][0]/all_docs  )*math.log( (all_docs*n[1][0])/(  ( n[1][0] + n[1][1] )*( n[0][0] + n[1][0] )  )  )
   if ( not n[0][0] == 0 ) :
     score = score +  ( n[0][0]/all_docs  )*math.log( (all_docs*n[0][0])/(  ( n[0][1] + n[0][0] )*( n[1][0] + n[0][0] )  )  )
   
   return score
   
   
# arrange vocab terms in decreasing order of MI score
def Feature_MI( Model , Train_Data) :

    Mi_score_list = []
    feature_name = []
    total_number_class = 2
    for i in range( len(Model) ) :
        score_of_ith_token = 0
        for j in range( total_number_class ) :
            score_of_ith_token = score_of_ith_token + MI_score( Model , i , j  , Train_Data )
        
        index = Find_index( Mi_score_list , score_of_ith_token )
        Mi_score_list.insert( index , score_of_ith_token )
        feature_name.insert( index , Model[i].token )
        
    for i in range( len(feature_name) ) :
        token = feature_name[i]
        index = Find_Index( Model , token )
        Model[ index[1] ].Rank_mi = i + 1
        
    return feature_name
   
   
   
#  arrange vocab terms in decreasing order of tf-idf
def Sort_tfidf( Model ) :
    
    feature_name = []
    tf_idt_score_list = []
    
    for i in range(len(Model)) :
        score_of_ith_token = 0
        number_classes_have_th_token=0
        for j in range( total_number_class ) :
            if ( Model[i].term_frq_list[j] > 0 ) :
                number_classes_have_th_token = number_classes_have_th_token + 1
        idf = math.log( total_number_class / number_classes_have_th_token  )
        for j in range( total_number_class ) :
            score_of_ith_token = score_of_ith_token +  Model[i].term_frq_list[j]
        score_of_ith_token = score_of_ith_token*idf
        
        index = Find_index(tf_idt_score_list,score_of_ith_token)
        tf_idt_score_list.insert(index,score_of_ith_token)
        feature_name.insert(index,Model[i].token)
           
    
    for i in range( len(feature_name) ) :
        token = feature_name[i]
        index = Find_Index(Model,token)
        Model[ index[1] ].Rank_tfidf = i+1
    
    
    return feature_name
