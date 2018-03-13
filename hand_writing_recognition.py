#https://www.youtube.com/watch?v=PO4hePKWIGQ

import matplotlib.pyplot as plt
import pylab as pl
from sklearn.datasets import load_digits
import random 
from sklearn import ensemble
#load_digits is our data for practise contain images in array 
#form where element of array shows the pixels value
#1797 images with 8*8 featiures

digits=load_digits()

#lets see a image 
pl.gray()
pl.matshow(digits.images[0])
pl.show()
#its shows the image but what python see is 
digits.images[0]
#its showing an arrya 8*8 dividing horizontal and vertical 
#length into 8 sub parts 
# its an 8 bit pixel image with value 0-255

#visyua;ise first 15 images
images_and_labels=list(zip(digits.images,digits.target))
plt.figure(figsize=(5,5))
for index, (image,label) in enumerate(images_and_labels[:15]):
	plt.subplot(3,5,index+1)
	plt.axis("off")
	plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
	plt.title('%i'%label) 

plt.show()


#Ensemble methods are learning algorithms that construct a. set of classifiers and then classify new data points by taking a (weighted) vote of their predictions. The original ensemble method is 
#Bayesian aver- aging, but more recent algorithms include error-correcting output coding, Bagging, and boosting.	


#defining aribales
n_samples=len(digits.images)
#n_sample contain number of case we have 
#1D array instead:
x=digits.images.reshape((n_samples,-1))    #n_samples row unknown column  n_sampes*8*8=n_sample*64
#this will result into a vector 
#we have 8*8 =64 dimension for sinle image
# -1 simply means that it is an unknown dimension and we want numpy to figure it out. And numpy will figure
# this by looking at the 'length of the array and remaining dimensions' and making sure it satisfies the above mentioned criteria

#Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

# >>> digits.images[0]  (8*8)
# array([[  0.,   0.,   5.,  13.,   9.,   1.,   0.,   0.],
#        [  0.,   0.,  13.,  15.,  10.,  15.,   5.,   0.],
#        [  0.,   3.,  15.,   2.,   0.,  11.,   8.,   0.],
#        [  0.,   4.,  12.,   0.,   0.,   8.,   8.,   0.],
#        [  0.,   5.,   8.,   0.,   0.,   9.,   8.,   0.],
#        [  0.,   4.,  11.,   0.,   1.,  12.,   7.,   0.],
#        [  0.,   2.,  14.,   5.,  10.,  12.,   0.,   0.],
#        [  0.,   0.,   6.,  13.,  10.,   0.,   0.,   0.]])
# >>> x[0] (64*1)
# array([  0.,   0.,   5.,  13.,   9.,   1.,   0.,   0.,   0.,   0.,  13.,
#         15.,  10.,  15.,   5.,   0.,   0.,   3.,  15.,   2.,   0.,  11.,
#          8.,   0.,   0.,   4.,  12.,   0.,   0.,   8.,   8.,   0.,   0.,
#          5.,   8.,   0.,   0.,   9.,   8.,   0.,   0.,   4.,  11.,   0.,
#          1.,  12.,   7.,   0.,   0.,   2.,  14.,   5.,  10.,  12.,   0.,
#          0.,   0.,   0.,   6.,  13.,  10.,   0.,   0.,   0.])

# >>> len(x[0])
# 64
# >>> len(digits.images[0])
# 8


y=digits.target

#so we have n_samples at all
sam_in=random.sample(range(len(x)),len(x)/5)
#dividing data in 20-80 sets
val_in=[i for i in range(len(x)) if i not in sam_in]

sam_img=[x[i] for i in sam_in]
val_img=[x[i] for i in val_in]

sam_target=[y[i] for i in sam_in]
val_target=[y[i] for i in val_in]

#sam_img and target is list object
#we are using random tree classisier whoich run different 
#decision treee and extract result from them

#so we are fitting the data as a image and label i e a sinle image has 8*8 pixel so 64 features for a image
classifier=ensemble.RandomForestClassifier()
classifier.fit(sam_img,sam_target)

print ("so lets see out put for this image")

#i is the number of datasets 
i=5
pl.gray()
pl.matshow(digits.images[i])
pl.show()


#classifier.predict(x[0])
#this will not run
b=x[i].reshape(1,-1)  #1 row unknown column
print ("the predtion f image is ",classifier.predict(b))
# >>> b
# array([[  0.,   0.,   5.,  13.,   9.,   1.,   0.,   0.,   0.,   0.,  13.,
#          15.,  10.,  15.,   5.,   0.,   0.,   3.,  15.,   2.,   0.,  11.,
#           8.,   0.,   0.,   4.,  12.,   0.,   0.,   8.,   8.,   0.,   0.,
#           5.,   8.,   0.,   0.,   9.,   8.,   0.,   0.,   4.,  11.,   0.,
#           1.,  12.,   7.,   0.,   0.,   2.,  14.,   5.,  10.,  12.,   0.,
#           0.,   0.,   0.,   6.,  13.,  10.,   0.,   0.,   0.]])
# >>> x[0]
# array([  0.,   0.,   5.,  13.,   9.,   1.,   0.,   0.,   0.,   0.,  13.,
#         15.,  10.,  15.,   5.,   0.,   0.,   3.,  15.,   2.,   0.,  11.,
#          8.,   0.,   0.,   4.,  12.,   0.,   0.,   8.,   8.,   0.,   0.,
#          5.,   8.,   0.,   0.,   9.,   8.,   0.,   0.,   4.,  11.,   0.,
#          1.,  12.,   7.,   0.,   0.,   2.,  14.,   5.,  10.,  12.,   0.,
#          0.,   0.,   0.,   6.,  13.,  10.,   0.,   0.,   0.])
# >>> len(b)
# 1
# >>> len(x[0])
# 64


score=classifier.score(val_img,val_target)


print ('Random Tree Classifier:\n')
print ('Score\t '+str(score))