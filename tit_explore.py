import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# My fisrt model for submission.
# I intent to start always with a linear regression, 
# which in my view is the Hartree Fock method of data science, 
# and it has a lot to teach. 



train = pd.read_csv("train.csv")

print train.shape

print train.count()

#print train.describe()

fig = plt.figure(figsize=(18,6))

plt.subplot2grid((2,3),(0,0))
train.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Survived")

plt.subplot2grid((2,3),(0,1))
train.Pclass.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Class")


plt.subplot2grid((2,3),(1,0))
for x in [1,2,3]:
	train.Age[train.Pclass == x].plot(kind='kde')
plt.title("Class wrt Age")
plt.legend(['1st','2nd','3rd'])



plt.subplot2grid((2,3),(1,1))
train.Sex[train.Survived==1].value_counts(normalize=True).plot(kind="bar",alpha=0.5,color=['r','b'])
plt.title("Survived by Sex")


plt.subplot2grid((2,3),(1,2))
train.Pclass[train.Survived==1].value_counts(normalize=True).plot(kind="bar",alpha=0.5,color=['r','b','g'])
plt.title("Survived by class")



plt.show()
