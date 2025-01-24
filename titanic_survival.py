# import necessary libraries
import pandas as pd

# load titanic dataset
df=pd.read_csv('titanic.csv')

# display the first 5 rows of the dataset
print(df.head())

# get basic info about the dataset(number of rows, column types, missing values)
print(df.info())

# display summary statistics for numerical columns
print(df.describe())

# check the number of missing values in each column
print(df.isnull().sum())

# drop unnecessary columns that are not useful for prediction
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# fill missing values in the 'age' column with the mean age
df['Age']=df['Age'].fillna(df['Age'].mean())

# fill missing value in 'Embarked' with the most frequent value
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

# fill missing values in 'sex' with the most frequent value(mode)
df['Sex']= df['Sex'].fillna(df['Sex'].mode()[0])

# convert 'sex' column from text to numbers(0 = male, 1 = female)
df['Sex']= df['Sex'].map({'male': 0, 'female': 1})

df['Embarked']= df['Embarked'].map({'C':0, 'Q':1, 'S':2})

# display the cleaned dataset
print(df.head())

# verify that there are no missing values left
print(df.isnull().sum())

# import necessary library for train_test split
from sklearn.model_selection import train_test_split

# define features(x) and target variable (y)
X=df.drop(columns=['Survived'])#all columns except 'survived'
y=df['Survived']# the target variable

# split the dataset into 70% training and 30% testing data
X_train, X_test, Y_train, Y_test=train_test_split(X, y, test_size=0.3, random_state=42)

# print the shape of the training and testing sets
print(f"training set size:{X_train.shape}, testing set size:{X_test.shape}")

#check for missing values in training and testing data
print('Missing values in x_train:\n', X_train.isnull().sum())
print('Missing values in x_test:\n', X_test.isnull().sum())

# import necessary libraries for logistic regression
from sklearn.linear_model import LogisticRegression

# create the logistic regression model
model=LogisticRegression(max_iter=200)

# train the model on training data
model.fit(X_train, Y_train)

# make prediction on the test set
y_Pred=model.predict(X_test)

# print fist 10 actual vs predicted values
print('Actual values:   ', Y_test.values[:10])
print('predicted values: ', y_Pred[:10])

# import accuracy metric
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_test, y_Pred)
print(f"model accuracy:{accuracy:.2f}")#higher is better

# import confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix= confusion_matrix(Y_test, y_Pred)
print("confusion matrix:\n",conf_matrix)

# import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
#plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',xticklabels=['Died', 'Survived'],yticklabels=['Died', 'Survived'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('confusion matrix')


# import classification report
from sklearn.metrics import classification_report

#generate classification report
report = classification_report(Y_test, y_Pred)
print('classification report:\n', report)

# get feature importance(coefficients)
feature_importance=pd.DataFrame({'feature':X_train.columns, 'coefficient':model.coef_[0]})

# sort feature by absolute importance
feature_importance['Abs_coefficient']=feature_importance['coefficient'].abs()
feature_importance=feature_importance.sort_values(by='Abs_coefficient', ascending=False)

# display feature importance
print('feature importance:\n',feature_importance[['feature', 'coefficient']])

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.barh(feature_importance['feature'], feature_importance['coefficient'], color='teal')
plt.xlabel('coefficient value')
plt.ylabel('feature')
plt.title('feature importance in titanic survival prediction')
plt.gca().invert_yaxis()


from model_utils import save_model, load_model
from predict import predict_survival
save_model(model)
loaded_model = load_model()
passenger_info = {
    "Pclass": 1, "Sex": 1, "Age": 28, "SibSp": 0, "Parch": 0, "Fare": 50, "Embarked": 0
}
prediction_result = predict_survival(loaded_model, passenger_info)
print("survival prediction:", prediction_result)

plt.show()
