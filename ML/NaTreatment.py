import pandas as pd

# _____________________________________________________
# --- TRAITEMENTS DES DONNEES MANQUANTES -------------
# - Age treatment_na_age(df)
def treatment_na_age(df):
    # Solution de Remplacement
    # A -- Par la moyenne d'âge
    # B -- Random 25-30
    # C -- Label 'N/A'
    df['Age'].loc[df['Age'].isna()] = df['Age'].mean()

def treatment_na_embarked(df):
    # -- Remplacement par le mode : S
    df['Embarked'] = df['Embarked'].fillna('S')

def treatment_na_deck(df):
    df['Deck'] = df['Deck'].fillna('N/A')

# Cherche ceux qui ont le même numéro de ticket et si possible complete les données manquantes de cabin
def treatment_partial_na_cabin(df):
    # Lise les ticket de passagers qui ont une cabine connue
    ticket_cabin_not_na = df[~df['Cabin'].isna()]['Ticket'].unique()
    # Liste des tickets permettant d'associer un passager à la cabine inconnue
    can_replace_cabin = (df['Cabin'].isna())& (df['Ticket'].isin(ticket_cabin_not_na))
    # Créé une liste unique des cabines correspondant aux tickets
    cabin_ticket = df[~df['Cabin'].isna()][['Ticket','Cabin']].drop_duplicates()
    # Liste les cabines manquantes dont on peut associer une cabine grace au ticket
    isna_same_ticket = df['Cabin'].isna() & df['Ticket'].isin(cabin_ticket['Ticket'])
    # Récupére la cabine correspondant au ticket
    cabin_replace = pd.merge(df[isna_same_ticket]['Ticket'].to_frame(),cabin_ticket,left_on='Ticket',right_on='Ticket')
    # Remplace les cabines manquantes devinables
    df.loc[isna_same_ticket,'Cabin'] = cabin_replace['Cabin']
