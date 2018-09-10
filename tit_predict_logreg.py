import pandas as pd
import utils
from sklearn import linear_model, model_selection

## TRAINING

train = pd.read_csv("train.csv")

utils.clean_data(train)

y_train = train.Survived

#features = ["Pclass","Age","Sex","SibSp","Parch"]

features = ["Pclass","Sex","SibSp"]


X_train = train[features]

model = linear_model.LogisticRegression()

model_ = model.fit(X_train,y_train)


print "\n >>Logistic Regression acuracy:"
print model_.score(X_train,y_train)


print "\n >> Logistic Regression coefficients:"
co =  model.coef_
for i in range(len(features)):
    print "%15s :: %f "%(features[i],co[0][i])

scores = model_selection.cross_val_score(model,X_train,y_train, scoring
="accuracy",cv=50 )

print "\n >> Cross Validation:"

print scores
print "Average score:", scores.mean()


## TESTING

test = pd.read_csv("test.csv")
utils.clean_data(test)


X_test = test[features]
y_test = model.predict(X_test)

my_submission = pd.DataFrame({'PassengerId':test.PassengerId,"Survived":y_test})

my_submission.to_csv("subm_logreg.csv",index=False)


