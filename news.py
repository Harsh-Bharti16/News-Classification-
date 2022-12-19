
import pandas as pd # used for data cleaning and analysis in ML
import numpy as np #NumPy aims to provide an array object that is up to 50x faster than traditional Python lists
from sklearn.feature_extraction.text import CountVectorizer # It provides a selection of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction via a consistence interface in Python.
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/bbc-news-data.csv", sep='\t')
print(data.head())

data.isnull().sum()     #to check it has null valuse or not

data["category"].value_counts()


data = data[["title", "category"]] # category - sport buisness politics tech entertainment

x = np.array(data["title"]) #Training a news classification model
y = np.array(data["category"])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)  # prepare the data for the task of training a news classification model
 
model = MultinomialNB() #MULTINOMIAL NAIVE BAYES
model.fit(X_train,y_train)  

user = input("Enter a Text: ") # to check today news headlines 
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)