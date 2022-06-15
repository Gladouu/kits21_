"""

"""

import streamlit as st


from pathlib import Path
import pandas as pd

from  functions import fonctions_images as f_i
from   functions import fonctions_vie as f_v

import matplotlib.pyplot as plt
import numpy as np



title = "Application"
sidebar_name = "Application"

CURRENT_PATH = f_v.current_path()
print("CURRENT_PATH", CURRENT_PATH)
CE_PATH =  f_v.current_path(_add = "Streamlit_kidneySpy-rz", _verbose=True)
print("CE_PATH", CE_PATH)

models_hors_rein = ['m0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9']
models_rein =  ['m0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9']
models_tumor =  ['m0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9']
models_kyste =  ['m0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9']


dico_path_cases = {'cas 1': 'path_cas_1',
                   'cas 2': 'path_cas_2',
                   'cas 3': 'path_cas_3',}
dico_path_model = { 'model 1': 'path_model_1',
                    'model 2': 'path_mode_2',
                    'model 3': 'path_mode_3',}

list_cases = [k for k in dico_path_cases]
list_models = [k for k in dico_path_model]

TEXT_MODE = "mode niveaux"


# @st.cache permet de n'exeécutrer une fct que si qqchose a changé
## 

# path_df_comparatif_avec_round = str(CURRENT_PATH) + "\\DataFrames\\" + "df_77_comparatif_avec_round_3000imgs_20%sansrein___2022-05-26T10-42-31.csv"
path_df_comparatif_avec_round = r"DataFrames\df_77_comparatif_avec_round_3000imgs_20%sansrein___2022-05-26T10-42-31.csv"
# df_comparatif_avec_round = pd.read_csv(path_df_comparatif_avec_round)

# path_df_comparatif_sans_round = str(CURRENT_PATH) + "\\DataFrames\\" + "df_77_comparatif_sans_round_3000imgs_20%sansrein___2022-05-26T10-42-31.csv"
path_df_comparatif_sans_round = r"DataFrames\df_77_comparatif_sans_round_3000imgs_20%sansrein___2022-05-26T10-44-46.csv"
# df_comparatif_sans_round = pd.read_csv(path_df_comparatif_sans_round)

def maj_images_gauche(niveau):
    im_g = np.ones((512,512))*np.random.randint(255)
    return im_g
    # im_c = np.ones((512.512))
    # im_d = np.zeros((512.512))
    
#     # image gauche
#     arr = np.random.normal(1, 1, size=100)
#     fig_gauche, axg = plt.subplots()
#     axg.hist(arr, bins=20)
    
#     # image centrale
#     arr = np.random.normal(1, 1, size=100)
#     fig_centre, axc = plt.subplots()
#     axc.hist(arr, bins=20)
    ###
#     # image droite
#     arr = np.random.normal(1, 1, size=100)
#     fig_droite, axd = plt.subplots()
#     axd.hist(arr, bins=20)
    
# # def maj_mode_affichage(mode_affichage):
# #     if (mode_affichage == 'niveaux'):
# #         TEXT_MODE = 'Comparaison de 3 modèles'
# #     else:
# #         TEXT_MODE = 'compare 3 niveaux'
#     info_mode.text()

def maj_images():
    maj_images_gauche(1)
    # maj_images_centre()
    # maj_images_droite()


def run():
    # maj_images()
    
    
    



    st.title(title)
    col1, col2, col3, col4, col5 = st.columns(5)
    option = col1.selectbox(
     'cas à analyser :',
     (list_cases))
    
    col1.button('AFFICHER', key='bt_afficher', on_click=maj_images())
    

    mode_affichage = col2.radio(
    'Comparer :', ['niveaux','modèles'], key='mode_affichage')
    if (mode_affichage == 'niveaux'):
        TEXT_MODE = 'Comparaison'
        col2.text(str(TEXT_MODE))
    else:
        TEXT_MODE = 'Exploration'
        col2.text(str(TEXT_MODE))
    
    
    
    model_2 = col3.selectbox(
     'modèle 2:',
     list_models
     )
    
    model_1 = col4.selectbox(
     'modèle principal :',
     list_models
     )
    
    # MAX_CAS = 50
    # niveau = col2.slider('espace entre les niveauxx', 0, MAX_CAS, 1)
    
    space = col1.text_input('espacement des coupes')
    ESPACEMENT = space
    
    model_3 = col5.selectbox(
     'modèle 3 :',
     list_models
     )
    
    mode_affichage = col5.radio(
    'détection de zone :', ["rein","à risque", "tumorale"], key='zone')
    
    MAX_CAS = 1700
    niveau = col3.slider('niveau?', 0, MAX_CAS, 1)
    
    
    colG, colC, colD = st.columns(3)
    # im_g = maj_images_gauche(niveau)
    # fig, ax = plt.imshow(im_g)
    X = [x for x in range(-5, 5)]
    Y = [x for x in range(-5, 5)]
    figG = plt.figure(figsize=(20,20))
    ax = plt.plot(X, Y)
    colG.pyplot(figG)
    
    X = [x for x in range(-5, 5)]
    Y = [x for x in range(-5, 5)]
    figC = plt.figure(figsize=(20,20))

    ax = plt.plot(X, Y)
    colC.pyplot(figC)
    
    X = [x for x in range(-5, 5)]
    Y = [x for x in range(-5, 5)]
    figD = plt.figure(figsize=(20,20))
    ax = plt.plot(X, Y)
    colD.pyplot(figD)
    
    

    # # plt.show()
    # st.pyplot(plt.plot(X, Y))
    
    
    
    # fig_centre, ax = plt.imshow(im_c)
    # fig_droite, ax = plt.imshow(im_d)
    
    # col3.pyplot(fig_centre)
    # col5.pyplot(fig_droite)
    # maj_images()

    # option = col3.selectbox(
    #  'modèle 3 :',
    #  models_rein
    #  )
    
    # option = col4.selectbox(
    #  'modèle tumeur rein :',
    #  models_tumor
    #  )
    
    # option = col5.selectbox(
    #  'modèle kyste rein :',
    #  models_kyste
    #  )
    
    # if st.checkbox("visualiser", False):
    #    df_comparatif_avec_round