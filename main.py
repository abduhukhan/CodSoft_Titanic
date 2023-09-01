import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#Lets solve this using logistic regression!

data = pd.read_csv('tested.csv')
#Check out the length of dataset which provides us the number of rows
print(len(data))
#Check out the head and names of columns of the dataset
print(data.head())
print(data.columns)
#Check out the info of dataset
print(data.info())
#Check out the statistical values of the dataset
print(data.describe())

#DATA VISUALIZATION USING SEABORN

#Check out how many survived and how many couldnt survive

plot_1 = sns.countplot(x="Survived",data=data)
plot_1.set_xticklabels(["Died","Survived"])
plt.show()

#DATA CLEANING

#NULL VALUES
print(data.isna())
print(data.isna().sum())

#Now the results show that there are 327 null values in Cabin column and 86 null values in Age column
#and 1 null value in Fare column

print((data['Cabin'].isna().sum()/len(data['Cabin'])*100)) #Percentage = 78.23%
print((data['Age'].isna().sum()/len(data['Age'])*100))     #Percentage = 20.57%
print((data['Fare'].isna().sum()/len(data['Fare'])*100))   #Percentage = 0.24%

#Fill age column

data['Age'].fillna(data['Age'].mean(),inplace=True)
print("The null values in Age column are as follows: ")
print(data['Age'].isna().sum())

#Fill fare column

data['Fare'].fillna(data['Fare'].mean(),inplace=True)
print("The null values in Fare column are as follows: ")
print(data['Fare'].isna().sum())

#Since there are approximately 78.23% null values in Cabin column so we will drop it

data.drop('Cabin',axis=1,inplace=True)
print(data.columns)

#CONVERSION OF NON-NUMERICAL COLUMNS

print(data.dtypes)
#There are 4 such columns which are Name,Sex,Ticket and Embarked
#Drop Name,Ticket and Embarked columns
data.drop(['Name','Ticket','Embarked'],axis=1,inplace=True)
print(data.columns)

#Convert Sex Column datatype
sex = pd.get_dummies(data['Sex'],drop_first=True)
data['Sex']=sex
data['Sex'] = data['Sex'].astype(int)
print(data.head())
print(data.dtypes)

#Extracting Dependent and Independent Variables
x=data[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare']]
y=data[['Survived']]

#LOGISTIC REGRESSION
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)

reg=LogisticRegression()
reg.fit(x_train,y_train)
predict=reg.predict(x_test)

#CLASSIFICATION REPORT
print(classification_report(y_test,predict))





