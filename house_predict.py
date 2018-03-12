from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split

boston=load_boston()

print "here is the data",boston

df_x=pd.DataFrame(boston.data,columns=boston.feature_names)
df_y=pd.DataFrame(boston.target)

print "description of features",df_x.describe()
print "description of taget",df_y.describe() 

x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=4)


reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)

print "the regression coefficient are",reg.coef_

a=reg.predict(x_test)

print "the predcition for ",df_y ,"are ",a

df_a=pd.DataFrame(a)

print "the squred error is",np.mean((a-y_test)**2)
