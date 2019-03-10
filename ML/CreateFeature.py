import pandas as pd
import numpy as np
# ________________________________________________________
# --- CREATE FEATURES ------------
def extract_deck(df):
    # Cabin -> Deck
    df['Deck'] = df['Cabin'].str.extract(r'^([A-Z]).+?')

def extract_cabin_na(df):
    df['CabinNA'] = df['Cabin'].isna().astype(int)

def extract_title(df):
    # Name -> FullName; Title; Name
    df[['FullName','Title','Names']] = df['Name'].str.extract(r'([A-z]+), ([A-z]+). (.+)')

# SibSp + Parch -> FullFamily

def extract_age_class(df):
    #df['AgeClass'] = np.digitize(df['Age'], bins=[1,6,16,26,31,37,45,63,100]) # June
    df['AgeClass'] = np.digitize(df['Age'], bins=[0,19]) # Cyrille


def extract_full_family(df):
    df['FullFamily']=df.SibSp+df.Parch

def extract_fare_class(df):
    df['Poor'] = (df.Fare < 12).astype(int)
    df['Mediocre'] = ((df.Fare > 12) & (df.Fare <= 34)).astype(int)
    df['Average'] = ((df.Fare > 34) & (df.Fare <= 69)).astype(int)
    df['HighPrice'] = ((df.Fare > 69) & (df.Fare < 134)).astype(int)
    df['Rich'] = ((df.Fare >134 ) & (df.Fare <= 222)).astype(int)
    df['Luxurious'] = (df.Fare > 222).astype(int)
