import pandas as pd
import utils
from sklearn import linear_model, preprocessing


train = pd.read_csv("train.csv")

utils.clean_data(train)

y_train = train.Survived

#features = ["Pclass","Age","Sex","SibSp","Parch"]

features = ["Pclass","Sex","SibSp"]


X_train = train[features]

model = linear_model.LogisticRegression()

model_ = model.fit(X_train,y_train)


print "\n >> Logistic Regression acuracy:"
print model_.score(X_train,y_train)


print "\n >> Logistic Regression coefficients:"
co =  model.coef_
for i in range(len(features)):
    print "%15s :: %f "%(features[i],co[0][i])


poly = preprocessing.PolynomialFeatures(degree=2)
poly_X_train = poly.fit_transform(X_train)

model_ = model.fit(poly_X_train,y_train)
print "\n >> Poly(2) Logistic Regression acuracy:"
print model_.score(poly_X_train,y_train)

print "\n >> Logistic Regression coefficients:"
co =  model.coef_
print co

## TESTING

test = pd.read_csv("test.csv")
utils.clean_data(test)


X_test = test[features]

poly_X_test= poly.fit_transform(X_test)

y_test = model.predict(poly_X_test)

my_submission = pd.DataFrame({'PassengerId':test.PassengerId,"Survived":y_test})

my_submission.to_csv("subm_poly_logreg_2.csv",index=False)


