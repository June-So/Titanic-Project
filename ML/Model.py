# ---- IMPORT DES MODULES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

title = ['Capt', 'Col', 'Don', 'Dona', 'Dr','Jonkheer', 'Lady', 'Major', 'Master', 'Miss', 'Mlle', 'Mme', 'Mr','Mrs', 'Ms', 'Rev', 'Sir', 'th']
fare_class = ['Poor','Mediocre','Average','HighPrice','Rich','Luxurious']
class Model:
    def __init__(self):
        self.features = ['Sex','Pclass','Deck','AgeClass','FullFamily'] + fare_class + title

        self.model = LogisticRegression()
        #self.model = LogisticRegressionCV(Cs=100)
        #self.model = LinearSVC()
        #self.model = RandomForestClassifier(n_estimators=1400,min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=90,max_depth=90,bootstrap=False)

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
        self.train_clean.to_csv('data/train_clean.csv',index=True)
        self.test_clean.to_csv('data/test_clean.csv',index=True)

    def select_features(self,cols_features=False):
        print('sélection des features..')
        if cols_features:
            self.features = cols_features

    def split_data(self,y='Survived'):
        print('préparation des données pour le modèle..')
        X = self.train_clean[self.features]
        Y = self.train_clean[y]
        self.X_dev = self.test_clean[self.features]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

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

    # a,b=stacking(train,target,test)
    # a,b=pd.DataFrame(np.array(a).T),pd.DataFrame(np.array(b).T)
    # print(a,b)
    def stacking(features,label,test):
        models_test=[]
        a=[]
        b=[]
        models_test.append(('BaggingClf      ' , BaggingClassifier( oob_score=False, n_estimators=200, bootstrap=True, bootstrap_features= False, max_samples=9, max_features=37)))
        models_test.append(('RandomForestClf ' , RandomForestClassifier(n_estimators=2000, min_samples_split= 2, min_samples_leaf=2, max_features='sqrt', max_depth=50, bootstrap=True)))
        models_test.append(('LogReg' , LogisticRegression(penalty='l1',C=2.929292929292929)))
        for name, model in models_test:
            model.fit(features, label)
            predictions_train = model.predict(train)
            predictions_test = model.predict(test)
            a.append(predictions_train)
            b.append(predictions_test)

        return a,b

    def search_hyperparameters(self):
        print('recherche d\'hyperparametres..')
        #Create Hyperparameter Search Space
        penalty = ['l1', 'l2']
        # Create regularization hyperparameter distribution using uniform distribution
        C = np.random.normal(loc=10, scale=4,size=100)
        C = list(np.linspace(0,5, 100)).pop(0)
        #Create hyperparameter options
        hyperparameters = dict(C=[0.000001,0.0001,0.0001,0.1,1,10,100,1000], penalty=penalty)
        # Create randomized search 5-fold cross validation and 100 iterations
        randomlog = RandomizedSearchCV(self.model, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)
        # Fit randomized search
        randomlog.fit(self.X_train,self.y_train)
        best_params = randomlog.best_params_
        self.model = LogisticRegression(C=best_params['C'],penalty=best_params['penalty'])

    def search_param_decisionTree(self):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        max_leaf_nodes= [int(x) for x in np.linspace(10, 110, num = 11)]
        max_leaf_nodes.append(None)
        rf_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap,
                  'max_leaf_nodes':max_leaf_nodes}
        {'bootstrap': [True, False],
         'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
         'max_features': ['auto', 'sqrt'],
         'min_samples_leaf': [1, 2, 4],
         'min_samples_split': [2, 5, 10],
         'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}


        rf = RandomForestClassifier()
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = rf_grid, n_iter = 100, cv = 3, verbose=2, random_state=5, n_jobs = -1)
        # Fit the random search model
        rf_random.fit(self.X_train, self.y_train)
