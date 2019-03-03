import pandas as pd
#___________________________________________
# --- TRANSFORMATION DE FEATURES ---------------------
# - Embarked transform_embarked()
# - Sex transform_sex()
# - Fare transform_fare()

def transform_embarked(df):
    # -- Encoding [S=0,Q=1,C=2]
    df['Embarked'] = df['Embarked'].replace({'S':0,'Q':2,'C':1})

def transform_sex(df):
    # -- Encoding [male=0,female=1]
    df['Sex'] = (df['Sex'] == 'female').astype(int)

def transform_fare(df):
    return df

def transform_pclass(df):
    # One Hot Encoding
    df[['P1','P2','P3']] = pd.get_dummies(df['Pclass'])

def transform_deck(df):
    deck_encoding = {'N/A':0,'G':1,'F':2,'E':3,'D':4,'C':5,'B':6,'A':7}
    df['Deck'] = df['Deck'].replace(deck_encoding)
