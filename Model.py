# This is done using google colab and the following libraries are needed. Pandas and numpy are generally inbuilt with it

!pip install sklearn
!pip install pandas
!pip install numpy

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np

# Dataset to pandas dataframe

train=pd.read_csv('/content/train_data.txt',sep=':::',names=['id','title','genre','description'])  # here the txt has data with ':::' as separator
test_result=pd.read_csv('/content/test_data_solution.txt',sep=':::',names=['id','title','genre','description'])

# Creating training and testing datasets

x_train=train['description']
y_train=train['genre']
x_test=test_result['description']
y_test=test_result['genre']

# Using the TfidfVectorizer to convert text features to numerical data

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train = vectorizer.fit_transform(x_train)[:15000]
y_train = y_train[:15000]
X_test = vectorizer.transform(x_test)[:1000]
y_test = y_test[:1000]

# Using a SVC model for this task. We set the kernel to sigmoid in order to  obtain non-linear calssifications if possible.

clf = SVC(kernel='sigmoid')
clf.fit(X_train, y_train)

# Finding the accuracy of the model

y_pred=clf.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
