#https://archive.ics.uci.edu/ml/machine-learning-databases/00228
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

#SMSSPamCollection is the file name
#and names are the column of our new formd data
#download the file and place it at home directory

df=pd.read_csv('SMSSpamCollection',sep='\t',names=['Status','Message'])

#'ham' are not spam whereas 'spam' are spam
print "the data is something like ..",df.head()
print "of which we have  spam =" ,len(df[df.Status=='spam'])

df.loc[df["Status"]=='spam',"Status"]=0
df.loc[df["Status"]=='ham',"Status"]=1

print "after chaniging label",df.head()

df_x=df["Message"]
df_y=df["Status"]


x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=4)
#converting those 0 and 1 to int 
y_train=y_train.astype('int')

#counter vector which make bag of words using presence or absence of word
cv=CountVectorizer()
#transferring the string into bag of words
x_traincv2=cv.fit_transform(x_train)
a=x_traincv2.toarray()

print "here our bags of words ",a

print "te total posiibility of words ", len(a[0])
print "the total word present ",len(a)

print "writng the string from bag of words ",cv.inverse_transform(a[0])

#the countveactor only tell us about the presence
# of word so we go with tfidf which assign the weight to each word as per there occurence in all document

tfcv=TfidfVectorizer(min_df=1,stop_words="english")
#stop_words are the word common in language so of no use
#transferring the string into bag of words
x_traincv2=tfcv.fit_transform(x_train)
xtestcv=tfcv.transform(x_test)

a2=x_traincv2.toarray();

print "so our new bag of words with weight",a2
print "and the words in our weighted bag is ",tfcv.get_feature_names()

#using multinomail classifer for spam classification
mnb=MultinomialNB()
mnb.fit(x_traincv2,y_train)

pred=mnb.predict(xtestcv)
print "so here is our prediction",pred


real=np.array(y_test)

print "y test",y_test
print "actual label in list format ",real
print "prediction" ,pred

count=0



for i in range (len(pred)):
	if real[i]==pred[i]:
		count=count+1


print "the correct prediction are ",count
print "out of ",len(y_test)			
