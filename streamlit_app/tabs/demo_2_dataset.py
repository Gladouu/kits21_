"""
Created on Sat May 28 14:02:35 2022
Last Modified on Jeudi 16 Juin 2022

@author: Vannetzel Damien, Foyou Samuel, Valenzuela Gladis
"""

import streamlit as st
import pandas as pd
import numpy as np


title = "Les données"
sidebar_name = "Les données"


def run():
    st.title(title)

    status = st.radio(" ", ('En bref', 'Dive into Data'))
    if (status == 'En bref'):
        st.markdown("----")
        st.title("Cohorte")
        st.markdown("""
        La cohorte KiTS21 comprend des patients qui ont subi une néphrectomie partielle ou radicale pour suspicion de malignité rénale entre 2010 et 2020. Une revue rétrospective de ces cas a été menée pour identifier tous les patients qui avaient subi une tomodensitométrie préopératoire avec injection de contraste qui comprend l'intégralité des reins.
        
        Le jeu de données est donc composé de 300 patients. Pour chacun de ses 300 patients, on retrouve une imagerie médicale de type tomodensitométrie (3D - nombre de coupes variables par patient, allant de 29 à 1059), ainsi que trois propositions de segmentations différentes.
        """
                    )

        st.markdown("----")
        st.title("Format des données")
        st.markdown("""
        L'imagerie TDM native ainsi que les vérités terrains (segmentations) ont été fournies dans un format NIFTI (3D) anonymisé. 
        
        La spécificité des images TDM est qu’elles sont en niveaux de gris (encodées selon les unités Hounsfiled et non selon la norme RGB).

        Les images ont donc des matrices correspondantes à (num_coupes, hauteur, largeur). Ici, num_coupes correspond à une vue axiale (ou transverse) et progresse du haut vers le bas à mesure que l'indice de coupe augmente. Il est également important de souligner l’épaisseur de coupe (slice thickness) variable d’un examen à l’autre. 
        """
                    )

        st.image("images/Medical-image-coordinate-systems.jpg")
        st.image("images/imgs_3axes.png")

        st.markdown("----")
        st.titlz("Vérité Terrain (Ground Truth)")
        st.markdown(
            """
        Le scanner préopératoire cortico-médullaire le plus récent de chaque cas a été segmenté indépendamment trois fois pour chaque instance des classes sémantiques suivantes :
            """)

        st
        st.markdown("""Rein,""")
        st.markdown("""Tumeur,""")
        st.markdown(
            """Kyste : Masses rénales radiologiquement (ou pathologiquement, si disponibles) déterminées comme des kystes.
        """)

    if (status == 'Dive into Data'):
        analysResult = st.selectbox(
            "Tour d'horizon : ",
            ['Jeu de données', 'Préparation de données', 'Analyse Exploratoire des données', 'Propriétaire de données'])
        if (analysResult == 'Jeu de données'):
            st.markdown("----")
            st.markdown("""
            
            """)
            st.markdown(
                """ Nous avons choisi les segmentation de type MAJ correspondant au diagnostic majoritaire du collège d'experts """)
            st.markdown("----")
        if (analysResult == 'Préparation de données'):
            st.markdown("---")
            st.markdown(""" Format des images :""")
            st.markdown(
                """ => Pour l’approche 2D, tous les niveaux de coupes ont été transformés en images .png, """)
            st.markdown(
                """ => Pour l’approche 3D, nous avons décidé de garder le format initial nifti de chaque cas. """)
            st.markdown("---")
            st.markdown(
                """ Tous les examens ont des dimensions de types (z, 512, 512), sauf le cas 160 (z, 512, 796) """)
            st.markdown(
                """ => Approche 2D : tf.image.resize(image, (512,512)) """)
            st.markdown(
                """ => Approche 3D :  ndimage.zoom(img, depth_factor, height_factor, width_factor), order=1) """)
            st.markdown("---")
            st.markdown(
                """ Tous les cas n’ont pas le même nombre de coupes : """)
            st.markdown(
                """ => Approche 2D : Le nombre de coupe n’a pas d’importance pour le modèle en 2D, """)
            st.markdown(
                """ => Approche 3D : créer des patchs de 16 coupes là où il y a du de la segmentation. """)
            st.markdown("---")
            st.markdown(""" Échelle des pixels : """)
            st.markdown(""" => Approche 2D : Dans l’image PNG, la valeur des  pixels va de 0 à 255. Leur valeur est divisée par 255 pour qu’ils aient une valeur comprise entre 0 et 1 """)
            st.markdown(""" <> im_std = tf.multiply(float(im),1/255) """)
            st.markdown(""" => Approche 3D : Unités Hounsfield """)
            st.markdown(
                """ <> np.clip(img, -1000, +1000), #interpatientvariability """)
            st.markdown(
                """ <> np.clip(img, -150, 600),#fenêtrage tissu mous """)
            st.markdown(
                """ <> cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) """)
            st.markdown("---")
            st.markdown(""" Le jeux est déséquilibré : """)
            st.markdown(
                """ => Déséquilibre en nombre de coupes : zone sans rein > zone avec rein >  zone avec tumeur ou un kyste : """)
            st.image(
                r"\Users\doudoux\Documents\GitHub\App_KidneySpyRZ\images\DS_Kits21_Dispersion_par_Coupe.png")
            st.markdown("---")
            st.markdown(""" => Déséquilibre en nombre de pixels par classe """)
            st.image(
                r"\Users\doudoux\Documents\GitHub\App_KidneySpyRZ\images\DS_Kits21_Dispersion_par_Pixels.png")

        if (analysResult == 'Analyse Exploratoire des données'):
            st.markdown(
                """ Exemple d'images correspondant à un niveau de coupe """)
            st.image(
                r"\Users\doudoux\Documents\GitHub\App_KidneySpyRZ\images\DS_Kits21_Scanner_Reins.png")
            st.image(
                r"\Users\doudoux\Documents\GitHub\App_KidneySpyRZ\images\DS_Kits21_Scanner_Reins_Segmentations.png")
            st.markdown("---")
            st.markdown(""" Exemple de distribution de données """)
            st.image(
                r"\Users\doudoux\Documents\GitHub\App_KidneySpyRZ\images\DS_Kits21_Distrib_Par_Malignite_Age_avant_30ans.png")
            st.markdown("---")
            st.image(
                r"\Users\doudoux\Documents\GitHub\App_KidneySpyRZ\images\DS_Kits21_Distrib_Par_Malignite_Age_Apres_30ans.png")
            st.markdown("---")
            st.image(
                r"\Users\doudoux\Documents\GitHub\App_KidneySpyRZ\images\DS_Kits21_Distrib_Par_Conso_Alcool.png")
            st.markdown("---")
            st.image(
                r"\Users\doudoux\Documents\GitHub\App_KidneySpyRZ\images\DS_Kits21_Distrib_Par_Conso_Tabac.png")
            st.markdown("---")
            st.image(
                r"\Users\doudoux\Documents\GitHub\App_KidneySpyRZ\images\DS_Kits21_Distrib_Par_Malignite_Sexe.png")
            st.markdown("---")
        if (analysResult == 'Propriétaire de données'):
            st.markdown("----")
            st.markdown(
                """ The official repository of the 2021 Kidney and Kidney Tumor Segmentation Challenge """)
            st.markdown("----")
            st.markdown(""" https://kits21.kits-challenge.org/  """)
            st.markdown("----")
            st.markdown(""" Licence MIT """)
            st.markdown("----")
