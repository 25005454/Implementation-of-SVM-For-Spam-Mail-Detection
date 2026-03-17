# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: E.Vamsi krishna
RegisterNumber: 21225230065  
*/
```
```

import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
<img width="1052" height="57" alt="image" src="https://github.com/user-attachments/assets/0660d418-535b-4a63-bdc7-d010829ce178" />
<img width="900" height="260" alt="image" src="https://github.com/user-attachments/assets/07619d77-3908-4fd5-8fc3-df4f9b976405" />
<img width="525" height="340" alt="image" src="https://github.com/user-attachments/assets/54cb3ad7-dd2b-4d06-837b-e7428d1aaa33" />
<img width="395" height="177" alt="image" src="https://github.com/user-attachments/assets/c19a0337-6b02-4082-8aa4-5ac6f0bdd4c7" />
<img width="1047" height="113" alt="image" src="https://github.com/user-attachments/assets/be59c70f-e5c7-465d-8ac7-a9b87eda2d32" />
<img width="352" height="52" alt="image" src="https://github.com/user-attachments/assets/e8559f15-8a0e-444c-a421-99ccdd85c825" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
