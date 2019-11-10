import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('xxx.txt', delimiter='\t', header=None)
y, x_train = df[0], df[1]

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(x_train)

lr = linear_model.LogisticRegression()
lr.fit(x, y)

testX = vectorizer.transform(['URGENT! your mobile NO.1231232 was awarded', 'Hey honey, whats up'])

print(lr.predict(testX))

