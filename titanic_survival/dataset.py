# coding=utf-8

'''
    Survived：0代表否，1代表是）
    Pclass：社会阶级（1代表上层阶级，2代表中层阶级，3代表底层阶级）
    Name：船上乘客的名字
    Sex：船上乘客的性别
    Age：船上乘客的年龄（可能存在 NaN）
    SibSp：乘客在船上的兄弟姐妹和配偶的数量
    Parch：乘客在船上的父母以及小孩的数量
    Ticket：乘客船票的编号
    Fare：乘客为船票支付的费用
    Cabin：乘客所在船舱的编号（可能存在 NaN）
    Embarked：乘客上船的港口（C 代表从 Cherbourg 登船，Q 代表从 Queenstown 登船，S 代表从 Southampton 登船）
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class dataset(object):

    def __init__(self,path='titanic_data.csv'):
        self.path = path
        self.data = pd.read_csv(path)
        self.preprocessing()

    def preprocessing(self):
        # Missing data:  Age Cabin
        # remove Cabin first -- useless attribute
        self.data = self.data.drop(['Cabin','Name','Ticket'], axis=1)
        # one hot encoding
        objfeatures = ['Sex','Pclass','Embarked','Survived']
        self.data = pd.get_dummies(self.data,columns=objfeatures,prefix = objfeatures)
        # self.data = self.data.drop(['Sex_female'], axis=1)
        # process missing data of column 'Age'
        # straightforward way survived/unsurvived
        self.data['Age'] = self.data['Age'].fillna(self.data['Age'].mean())


    def split_data(self,training=0.8):
        m = len(self.data.columns)
        print (self.data)
        y = self.data.iloc[:,m-2:m].values
        x = self.data.iloc[:,1:m-2].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-training, random_state=42)
        return x,y,x_train, x_test, y_train, y_test


    def get_data(self):
        return self.split_data()






#if __name__ == '__main__':
#    ds = dataset()
#    ds.get_data()

