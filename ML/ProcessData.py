import ML.TransformData as TransformData
import ML.NaTreatment as NaTreatment
import ML.CreateFeature as CreateFeature

def processRegressionLogistic(df):

    # _____________________________________________________
    # --- TRAITEMENTS DES DONNEES MANQUANTES -------------
    print('remplacement des données manquantes..')
    NaTreatment.treatment_na_embarked(df)
    NaTreatment.treatment_na_age(df)
    NaTreatment.treatment_partial_na_cabin(df)

    # ________________________________________________________
    # --- CREATE FEATURES ------------
    print('Création de Features..')
    CreateFeature.extract_age_class(df)
    CreateFeature.extract_deck(df) # Create Deck, NumCabin, CabinNA
    NaTreatment.treatment_na_deck(df)

    # ________________________________________________________
    # --- TRANSFORMATION DE FEATURES ---------------------
    print('features encoding..')
    TransformData.transform_sex(df)
    TransformData.transform_embarked(df)
    TransformData.transform_fare(df)
    TransformData.transform_pclass(df)
    TransformData.transform_deck(df)
