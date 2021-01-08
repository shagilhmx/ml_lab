# import the required packages
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score



# Load dataset 
iris=datasets.load_iris() 
print("Iris Data set loaded...")
# Split the data into train and test samples
x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.1) 
print("Dataset is split into training and testing...")
print("Size of trainng data and its label",x_train.shape,y_train.shape) 
print("Size of trainng data and its label",x_test.shape, y_test.shape)

# Create object of KNN classifier
classifier=KNeighborsClassifier(n_neighbors=8,p=3,metric='euclidean')

classifier.fit(x_train,y_train)

#predict the test resuts
y_pred=classifier.predict(x_test)

cm=confusion_matrix(y_test,y_pred)
print('Confusion matrix is as follows\n',cm)
print('Accuracy Metrics')
print(classification_report(y_test,y_pred))
print(" correct predicition",accuracy_score(y_test,y_pred))
print(" worng predicition",(1-accuracy_score(y_test,y_pred)))
