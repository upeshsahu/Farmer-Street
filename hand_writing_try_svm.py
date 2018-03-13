from sklearn import svm
from sklearn.datasets import load_digits
import random 


digits=load_digits()







#defining aribales
n_samples=len(digits.images)
#n_sample contain number of case we have 
#1D array instead:
x=digits.images.reshape((n_samples,-1))    #n_samples row unknown column  n_sampes*8*8=n_sample*64
y=digits.target

#so we have n_samples at all
sam_in=random.sample(range(len(x)),len(x)/5)
#dividing data in 20-80 sets
val_in=[i for i in range(len(x)) if i not in sam_in]

sam_img=[x[i] for i in sam_in]
val_img=[x[i] for i in val_in]

sam_target=[y[i] for i in sam_in]
val_target=[y[i] for i in val_in]

print("with default kernel ie rbf")
clf = svm.SVC()
clf.fit(sam_img,sam_target)
score=clf.score(val_img,val_target)
print ('Score is\t '+str(score))
print("with  kernel ie sigmoid")
clf = svm.SVC(kernel='sigmoid')
clf.fit(sam_img,sam_target)
score=clf.score(val_img,val_target)
print ('Score is\t '+str(score))
print("with  kernel poly")
clf = svm.SVC(kernel='poly')
clf.fit(sam_img,sam_target)
score=clf.score(val_img,val_target)
print ('Score is\t '+str(score))