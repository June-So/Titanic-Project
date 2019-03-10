# ---- IMPORT DES MODULES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
# -- Module personel
import LibVisualisation.plotModel as pltM
import ML.ProcessData as ProcessData
import ML.Model as ModelManagement

Model = ModelManagement.Model()
print('\n---------------------------------------------------------------------')
print("\tBienvenue à bord du Titanic")
print('-----------------------------------------------------------------------\n')
# --- CHARGEMENT DES DONNEES
Model.load_data()
# --- TRAITEMENTS DES DONNEES MANQUANTES -------------
# --- TRANSFORMATION DE FEATURES ---------------------
# --- CREATE FEATURES --------------------------------
ProcessData.processRegressionLogistic(Model.all)
# --- RECONSTITUTION DES DONNEES -----------
Model.reconstitute_data()
# --- SELECTION DES DONNEES POUR NOTRE MODELE
Model.select_features()
# --- ENTRAINEMENT DE LA REGRESSION LOGISTIQUE
Model.split_data()
#Model.search_hyperparameters()
#Model.search_param_decisionTree()
Model.train_model()
# --- PREDICTIONS & PROBABILITE
results_train,results_test = Model.get_predicts()
# --- RESULTATS DU MODELE REGRESSION LOGISTIQUE
print('exportation des résultats en csv..')
results_test.to_csv('data/results_test.csv',index=False)
results_train.to_csv('data/results_train.csv',index=False)
# --- SOUMISSION KAGGLE
Model.predict_Xdev()

#_________________________________________________________
# -- Exportation visualisation de résultats
# Matrice de confusion
print('exportation visualisation des scores..')
cols_features = Model.features
lr = Model.model

pltM.plot_all_roc_curve(results_train,results_test,cols_features)
results_train[['Survived','Predict']] = results_train[['Survived','Predict']].replace({0:'Mort',1:'Vivant'})
results_test[['Survived','Predict']] = results_test[['Survived','Predict']].replace({0:'Mort',1:'Vivant'})
pltM.plot_all_confusion_matrix(results_train,results_test,cols_features)
pltM.plot_coefficients(lr,cols_features)

try:
    cols_plot_barplot = ['Sex']
    pltM.plot_all_TPR(results_train,results_test,cols_plot_barplot,cols_features)
except:
    print('Les barplots ne marcheront pas cette fois :/')
    #print(ValueError)
#__________________________________________________________
# FIN
print('\n---------------------------------------------------------------------')
print("\tL'équipe du Titanic vous souhaite un bon voyage.")
print('-----------------------------------------------------------------------\n')
