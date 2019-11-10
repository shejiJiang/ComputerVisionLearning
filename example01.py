from sklearn import linear_model

x = [[20, 3], [23, 7], [31, 10], [42, 13], [50, 7], [60, 5]]
y = [0, 1, 1, 1, 0, 0]

lr = linear_model.LogisticRegression()
lr.fit(x, y)

testX = [[22, 2]]
print("predict", lr.predict(testX))
print("predict_proba", lr.predict_proba(testX))