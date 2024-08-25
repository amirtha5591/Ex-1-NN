<H3>ENTER YOUR NAME         :AMIRTHAVARSHINI.R.D</H3>
<H3>ENTER YOUR REGISTER NO. :212223040013</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```py
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))


```


## OUTPUT:
### Dataset :
![neu1](https://github.com/user-attachments/assets/456eb898-67f5-49ec-98c4-dcb78a549d80)

### X values:
![neu2](https://github.com/user-attachments/assets/4b9afcb2-ce75-4aa3-94c5-d057b98c47fc)
### Y values :
![neu3](https://github.com/user-attachments/assets/71771607-8ff3-43bd-abcc-1d9c89cbe36f)
### Null values:
![neu4](https://github.com/user-attachments/assets/18437b98-8df1-475c-b4b3-fbf74d10b8cc)
### Duplicated values:
![neu5](https://github.com/user-attachments/assets/3b9b1ff3-8dd0-4939-b8cd-ce31820c55ee)
### Description:
![neu6](https://github.com/user-attachments/assets/bd7ac819-22ab-4a0c-8f55-3e81e44ce777)
### Normalized dataset:

![neu7](https://github.com/user-attachments/assets/7dbe113b-f01d-4bf1-9047-aaed22003457)
### Training data:
![neu8](https://github.com/user-attachments/assets/07d6626c-a3c0-4c0a-88fc-7215d13209b4)
### Testing data:
![neu9](https://github.com/user-attachments/assets/28815bb1-1d67-4ff0-b2c0-ac1991c3a6e1)











## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


