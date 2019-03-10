import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

def create_file_name(cols_features):
    folder = '_'.join(cols_features)
    if len(folder) > 20:
        folder = folder[20:]
    return folder

# crosstab une matrice de confusion a partir d'un tableau de resulstats
def plot_confusion_matrix(results,subplot=111,title='A PRECISER',normalize=True):
    cm_model = pd.crosstab(results['Survived'],results['Predict'],normalize=normalize).round(3)
    plt.subplot(subplot)
    plt.title(title)
    sns.heatmap(cm_model,annot=True, fmt='g')

def plot_roc_curve(y_true,y_pred,y_proba,title_dataset,cols_features):
    fpr, tpr, thresholds = roc_curve(y_true, np.round(y_proba,2),pos_label=1)
    roc_auc = roc_auc_score(y_true, y_proba)
    accuracy = accuracy_score(y_true,y_pred)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (AUC = {roc_auc.round(4)})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate\n(Morts déclarés Vivants)')
    plt.ylabel('True Positive Rate\n(Vivants déclarés Vivants)')
    plt.title(f"{title_dataset} - Accuracy {accuracy_score(y_true,y_pred).round(5)}\nFeatures : {cols_features}")
    plt.legend(loc="lower right")

def plot_all_confusion_matrix(results_train,results_test,cols_features,normalize="index"):
    folder = create_file_name(cols_features)
    sns.set(font_scale=2)

    plt.figure(figsize=(16,6))
    plt.suptitle(f"Matrice de confusion\nFeatures : {cols_features}")
    plot_confusion_matrix(results_train,121,"TRAIN",normalize=normalize)
    plot_confusion_matrix(results_test,122,"TEST",normalize=normalize)
    plt.savefig(f'img/{folder}_confusion_matrix')

def plot_all_roc_curve(results_train,results_test,cols_features):
    folder = create_file_name(cols_features)
    sns.set(font_scale=1.5)
    #print('--------- TRAIN --------')
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plot_roc_curve(results_train['Survived'],results_train['Predict'],results_train['PredictProba'],'TRAIN',cols_features)
    #print('--------- TEST --------')
    plt.subplot(122)
    plot_roc_curve(results_test['Survived'],results_test['Predict'],results_test['PredictProba'],'TEST',cols_features)
    plt.savefig(f'img/{folder}_roc_curve')

def plot_coefficients(lr,cols_features):
    folder = create_file_name(cols_features)
    coefs = list(lr.intercept_) + list(lr.coef_[0])
    colnames = ['intercept'] + cols_features
    intercepts = pd.DataFrame({'varname':colnames,'intercept_':coefs})
    intercepts.plot.barh(y='intercept_',x="varname",color="purple",figsize=(15,20),fontsize=16)
    plt.ylabel('')
    plt.axvline(x=0, linestyle='--', color='black', linewidth=4)
    plt.title('Coefficents des features')
    plt.savefig(f'img/{folder}_coefficents')

def plot_all_TPR(results_train,results_test,cols_plot,cols_suptitle):
    # le '_' après le titre est pour que les fichiers
    # ne soient pas écrasé lors de la visualisation
    sns.set(font_scale=1)
    plot_TPR(results_train,cols_plot,'TRAIN',see_predicts=True,legend=True,cols_suptitle=cols_suptitle)
    plot_TPR(results_train,cols_plot,'TRAIN_',see_predicts=False,cols_suptitle=cols_suptitle)
    plot_TPR(results_test,cols_plot,'TEST',see_predicts=True,legend=True,cols_suptitle=cols_suptitle)
    plot_TPR(results_test,cols_plot,'TEST_',see_predicts=False,cols_suptitle=cols_suptitle)

def plot_one_tpr(results,title,axe,fig,top=0,bottom=0,left=0,see_predicts=False,legend=False):
    if see_predicts:
        df_display = pd.crosstab(results['Predict'],results['Survived'],normalize='index')
        fig.legend(['Mort Réel','Vivant Réel'],loc='lower center',ncol=2)
        #legend=True
    else:
        df_display = pd.crosstab(results['Survived'],results['Good'],normalize='index')
        fig.legend(['Mauvaises prédictions','Bonnes prédictions'],loc='lower center',ncol=2)
    df_display.plot(kind="bar", rot=0, color=['crimson','darkcyan'],legend=legend, stacked=True, title=title,ax=axe)
    axe.set_xlabel(None)
    plt.subplots_adjust(top=top,bottom=bottom,left=left)

def plot_TPR(results,cols_features,dataset_title="A PRECISER",ncols=4,nrows=1,figsize=(16,10),see_predicts=False,legend=False,cols_suptitle=''):
    folder = create_file_name(cols_features)
    y = 0
    i = 0
    if (len(cols_features) > 1):# and ((ncols == 4) and (nrows != 1)):
        ncols = len(results[cols_features[-1]].unique())
        nrows = len(results[cols_features[0]].unique())
    fig, axs = plt.subplots(nrows,ncols, figsize=figsize, facecolor='w', edgecolor='k')
    fig.suptitle(f"Résultats des prédictions ['{dataset_title}']\n Features: {cols_suptitle}")
    # -- LEVEL 1 : une variable explicative -----------------------------------------
    if (type(cols_features) == str) or len(cols_features) == 1:
        if type(cols_features) != str:
            col_str = cols_features[0]
        for i,value in enumerate(results[col_str].unique()):
            search = (results[col_str] == value)
            title_plot = f"{col_str} = {value}"
            plot_one_tpr(results[search],title=title_plot,axe=axs[i],fig=fig,top=0.78,bottom=0.2,see_predicts=see_predicts,legend=legend)
    # -- LEVEL 2 : 2 variables explicatives -----------------------------------------
    elif len(cols_features) == 2:
        for value_0 in results[cols_features[0]].unique():
            for value_1 in results[cols_features[1]].unique():
                if i == ncols:
                    i = 0
                    y += 1
                search = (results[cols_features[0]] == value_0) & (results[cols_features[1]] == value_1)
                title_plot = f"{cols_features[0]} = {value_0} | {cols_features[1]} = {value_1}"
                plot_one_tpr(results[search],title_plot,axs[y][i],fig,top=0.88,bottom=0.08,see_predicts=see_predicts,legend=legend)
                i += 1
    # -- LEVEL 3 : 3 variables explicatives -----------------------------------------
    elif len(cols_features) >= 3:
        for sex in results[cols_features[0]].unique():
            for cabinNa in results[cols_features[1]].unique():
                for pclass in results[cols_features[2]].unique():
                    if i == ncols:
                        i = 0
                        y += 1
                    search = (results[cols_features[2]] == pclass) & (results[cols_features[1]] == cabinNa) & (results[cols_features[0]] == sex)
                    title_plot = f"{cols_features[0]} = {sex} | {cols_features[1]} = {cabinNa} | {cols_features[2]} = {pclass} |"
                    plot_one_tpr(results[search],title_plot,axs[y][i],fig,top=0.9,bottom=0.05,see_predicts=see_predicts,legend=legend)
                    i += 1
    else:
        return print('le plot de plus de 3 variables n`est pas codé :(')
    plt.savefig(f'img/{folder}_barplot_tpr_{dataset_title}')
