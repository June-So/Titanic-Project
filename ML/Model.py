# ---- IMPORT DES MODULES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

class Model:
    def __init__(self):
        self.features = ['Sex','Pclass']
        self.model = LogisticRegression()

        self.train = [] # Initial train dataset
        self.test = [] # Initial test dataset
        self.all = [] # Merge between train and test, modify for cleaning
        self.train_clean = [] # train dataset after process
        self.test_clean = [] # test dataset after process

        self.X_train = [] # create with split_data
        self.X_test = [] # create with split_data
        self.y_train = [] # create with split_data
        self.y_test = [] # create with split_data
        self.X_dev = [] # dataset to predict for kaggle

    def load_data(self,path_train='data/train.csv',path_test='data/test.csv',index="PassengerId",y='Survived'):
        print('chargement des données..')
        self.train = pd.read_csv(path_train, index_col=index)
        self.test = pd.read_csv(path_test, index_col=index)
        self.all = pd.concat([self.train.drop(y,axis=1), self.test])

    def reconstitute_data(self,y='Survived'):
        print('reconstitution des données..')
        self.train_clean = self.all[self.all.index.isin(self.train.index.tolist())]
        self.train_clean = pd.merge(self.train_clean, self.train[[y]], left_index=True, right_index=True)
        self.test_clean = self.all[self.all.index.isin(self.test.index.tolist())]
        self.train_clean.to_csv('data/train_clean.csv',index=False)
        self.test_clean.to_csv('data/test_clean.csv',index=False)

    def select_features(self,cols_features=False):
        print('sélection des features..')
        if cols_features:
            self.features = cols_features

    def split_data(self,y='Survived'):
        print('préparation des données pour le modèle..')
        X = self.train_clean[self.features]
        Y = self.train_clean[y]
        self.X_dev = self.test_clean[self.features]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,Y,test_size=0.2)

    def train_model(self):
        print('entrainement du modèle..')
        self.model.fit(self.X_train,self.y_train)

    def predict_Xdev(self):
        print('exportation des résultats pour kaggle..')
        kaggle = pd.DataFrame({'PassengerId':self.X_dev.index,'Survived':self.model.predict(self.X_dev)})
        kaggle.to_csv('data/submission.csv',index=False)

    def get_predicts(self):
        print('prédictions du modèle..')
        predictions_test = self.model.predict(self.X_test)
        predictions_train = self.model.predict(self.X_train)
        predictions_kaggle = self.model.predict(self.X_dev)

        predictions_proba_train = self.model.predict_proba(self.X_train)[:,1]
        predictions_proba_test = self.model.predict_proba(self.X_test)[:,1]

        print('enregistrement des résultats..')
        results_train = self.X_train.copy()
        results_test = self.X_test.copy()
        self.set_results(results_train,predictions_train,predictions_proba_train,self.y_train)
        self.set_results(results_test,predictions_test,predictions_proba_test,self.y_test)
        return results_train,results_test

    def set_results(self,results,predictions,predictions_proba,y):
        results['Predict'] = predictions
        results['PredictProba'] = predictions_proba
        results['Survived'] = y
        results['Good'] = results['Survived'] == results['Predict']
