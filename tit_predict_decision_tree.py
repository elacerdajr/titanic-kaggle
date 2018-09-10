import pandas as pd
import utils
from sklearn import tree, model_selection


train = pd.read_csv("train.csv")

utils.clean_data(train)

y_train = train.Survived

features = ["Pclass","Age","Sex","SibSp","Parch"]
#features = ["Pclass","Sex","SibSp"]

X_train = train[features]

decision_tree = tree.DecisionTreeClassifier(
    random_state =1,
    max_depth=7,
    min_samples_split=2
)

decision_tree_ = decision_tree.fit(X_train,y_train)

print decision_tree_.score(X_train,y_train)

scores = model_selection.cross_val_score(decision_tree,X_train,y_train, scoring
="accuracy",cv=50 )

print scores
print scores.mean()

tree.export_graphviz(decision_tree_,feature_names=features, out_file='tree.dot')

## TESTING

test = pd.read_csv("test.csv")
utils.clean_data(test)


X_test = test[features]


y_test = decision_tree.predict(X_test)

my_submission = pd.DataFrame({'PassengerId':test.PassengerId,"Survived":y_test})

my_submission.to_csv("subm_decision_tree.csv",index=False)


