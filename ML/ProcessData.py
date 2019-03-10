import ML.TransformData as TransformData
import ML.NaTreatment as NaTreatment
import ML.CreateFeature as CreateFeature

def processRegressionLogistic(df):
    # _____________________________________________________
    # --- TRAITEMENTS DES DONNEES MANQUANTES -------------
    print('remplacement des données manquantes..')
    NaTreatment.treatment_na_embarked(df)
    NaTreatment.treatment_na_age(df) # Marwan
    NaTreatment.treatment_partial_na_cabin(df)
    TransformData.transform_cabin_t(df) # Marwan

    # ________________________________________________________
    # --- CREATE FEATURES ------------
    print('Création de Features..')
    CreateFeature.extract_full_family(df) # Marwan sib + Sp
    CreateFeature.extract_age_class(df) # Discretisation de l'age
    CreateFeature.extract_cabin_na(df) # Create CabinNA
    CreateFeature.extract_deck(df) # Create Deck, NumCabin, CabinNA
    CreateFeature.extract_fare_class(df) # Marwan discretisation du tarif en hot encoding
    CreateFeature.extract_title(df) # Create FullName, Title, Names
    NaTreatment.treatment_na_deck(df)

    # ________________________________________________________
    # --- TRANSFORMATION DE FEATURES ---------------------
    print('features encoding..')
    TransformData.encoding_title(df)
    TransformData.transform_sex(df)
    TransformData.transform_embarked(df)
    TransformData.transform_pclass(df)
    TransformData.transform_deck(df)
