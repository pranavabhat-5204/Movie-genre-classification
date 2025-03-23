!pip install matplotlib.pyplot
!pip install sklearn
!pip install pandas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)


train=pd.read_csv('/content/train_data.txt',sep=':::',names=['id','title','genre','description'])
test_result=pd.read_csv('/content/test_data_solution.txt',sep=':::',names=['id','title','genre','description'])
x_train=train['description']
y_train=train['genre']
x_test=test_result['description']
y_test=test_result['genre']
X_train = vectorizer.fit_transform(x_train)[:15000]
y_train = y_train[:15000]
X_test = vectorizer.transform(x_test)[:1000]
y_test = y_test[:1000]
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
