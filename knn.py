from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris=datasets.load_iris()
x=iris.data
y=iris.target


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

print "here we go with our features",x,"length =",len(x)
print "and labels are",y,"length =",len(y)
print "which is divided into 75-25 train-test "

print "train data",x_train,y_train
print "testing data",x_test,y_test
clf=KNeighborsClassifier()
clf.fit(x_train,y_train)

p=clf.predict(x_test)

print "so here are our prediction",p
print "and the true labels are" ,y_test


print "accuracy of our algo is",accuracy_score(y_test,p)

