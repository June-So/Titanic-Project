import pandas as pd
import numpy as np
# ________________________________________________________
# --- CREATE FEATURES ------------
def extract_deck(df):
    # Cabin -> Deck
    df['Deck'] = df['Cabin'].str.extract(r'^([A-Z]).+')
# Name -> FullName; Title; Names
# SibSp + Parch -> FullFamily

def extract_age_class(df):
    df['AgeClass'] = np.digitize(df['Age'], bins=[1,6,16,26,31,37,42,100])
