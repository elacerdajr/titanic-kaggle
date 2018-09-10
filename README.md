# titanic-kaggle

Models built to predict who have survived to the tragic sinking of the RMS Titanic ship. Kaggle competition: [titanic](https://www.kaggle.com/c/titanic), my profile [elacerdajr](https://www.kaggle.com/elacerdajr/competitions) 

---


## Models

- **LogReg**: Logistic Regression. Features:  PClass, Sex, Sibsp.
- **LogReg2**: Logistic Regression with polynomial feartures (degree =2). Features: PClass, Sex, Sibsp. 
- **LogReg2+**: Logistic Regression with polynomial feartures (degree =2). Features: PClass, Sex, Age, Sibsp, Parch. 
- **DTree**: Decision Tree.



## Results
A sumary of the results are presented at the table: 

| Model       | Accurracy        |           | 
|-------------| --------------|-----------|
|      | Train       | Test |
| LogReg   | 0.80029       | 0.77033 |
| LogReg2  | 0.7991      | 0.77511 |
| LogReg2+      |  0.8305      | 0.78947 |
| Decision Tree       | 0.8039  (avg)    | 0.74162 |
 
## Conclusions 

 I noted that:

1. Looking at the coefficints at the LogReg model is a great way to infer which variables are more important to the predictions.
2. The accuracy is always lower for the test set, compeared to the trrain set.
3. The Decision Tree model represented the higher difference between train and test sets.
4. The LogReg2+ improved somehow the results compared to the LogReg2 and LogReg models, altough they are quite similar.

