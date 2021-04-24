import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
plt.style.use('ggplot')

df = pd.read_csv('Loan_Train.csv').dropna().drop(['Loan_ID'], axis=1)

def clean_data():
    global df
    # changes gender column
    df.replace(to_replace='Male', value=1, inplace=True)
    df.replace(to_replace='Female', value=0, inplace=True)

    # changes married column
    df['Married'].replace(to_replace='Yes', value=1, inplace=True)
    df['Married'].replace(to_replace='No', value=0, inplace=True)

    # changes to dependents column
    df['Dependents'].replace(to_replace='0', value=0, inplace=True)
    df['Dependents'].replace(to_replace='1', value=1, inplace=True)
    df['Dependents'].replace(to_replace='2', value=2, inplace=True)
    df['Dependents'].replace(to_replace='3+', value=3, inplace=True)

    # changes to education column
    df['Education'].replace(to_replace='Graduate', value=1, inplace=True)
    df['Education'].replace(to_replace='Not Graduate', value=0, inplace=True)

    # changes to self employed
    df['Self_Employed'].replace(to_replace='No', value=0, inplace=True)
    df['Self_Employed'].replace(to_replace='Yes', value=1, inplace=True)

    # changes to applicant income
    for income in df['ApplicantIncome']:
        df['ApplicantIncome'].replace(to_replace=income, value=np.log(income), inplace=True)

    # changes to co-applicant income
    for income in df['CoapplicantIncome']:
        df['CoapplicantIncome'].replace(to_replace=income, value=np.log(income), inplace=True)

    # loan amount term
    df['Loan_Amount_Term'].replace(to_replace=360, value=0, inplace=True)
    df['Loan_Amount_Term'].replace(to_replace=120, value=1, inplace=True)
    df['Loan_Amount_Term'].replace(to_replace=180, value=2, inplace=True)
    df['Loan_Amount_Term'].replace(to_replace=60, value=3, inplace=True)
    df['Loan_Amount_Term'].replace(to_replace=300, value=4, inplace=True)
    df['Loan_Amount_Term'].replace(to_replace=480, value=5, inplace=True)
    df['Loan_Amount_Term'].replace(to_replace=240, value=6, inplace=True)
    df['Loan_Amount_Term'].replace(to_replace=36, value=7, inplace=True)
    df['Loan_Amount_Term'].replace(to_replace=84, value=8, inplace=True)

    # changes to property area
    df['Property_Area'].replace(to_replace='Rural', value=0, inplace=True)
    df['Property_Area'].replace(to_replace='Urban', value=1, inplace=True)
    df['Property_Area'].replace(to_replace='Semiurban', value=2, inplace=True)

    # loan status changes
    df['Loan_Status'].replace(to_replace='N', value=0, inplace=True)
    df['Loan_Status'].replace(to_replace='Y', value=1, inplace=True)

    # apply logistic regression

    df.dropna(axis=0, inplace=True)

def logistic_regression():
    log_model = LogisticRegression()

    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    log_model.fit(X_train,y_train)

    predctions = log_model.predict(X_test)

    print('----------- Logistic Regression -----------')
    print('Classification Report: \n', classification_report(y_test, predctions))
    print('Confusion Matrix: \n', confusion_matrix(y_test, predctions))

def nearest_neighbors():
    knn = KNeighborsClassifier()

    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    knn.fit(X_train, y_train)

    pred = knn.predict(X_test)

    print('----------- K Nearest Neighbors -----------')
    print('Classification Report: \n',classification_report(y_test, pred))
    print('Confusion Matrix: \n',confusion_matrix(y_test, pred))


clean_data()
logistic_regression()
nearest_neighbors()
