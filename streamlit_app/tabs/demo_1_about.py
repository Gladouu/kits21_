"""
Created on Sat May 28 14:02:35 2022
Last Modified on Jeudi 16 Juin 2022

@author: Vannetzel Damien, Foyou Samuel, Valenzuela Gladis
"""

import streamlit as st


title = "Kidney Tumor Segmentation Challenge"
sidebar_name = "Introduction"
subtitle = "Contexte"


def run():
    st.title(title)
    st.markdown("---")
    st.image("images/z_gif.gif")

    st.markdown("---")
    st.title(subtitle)

    st.markdown("---")
    # st.markdown(
    #     """
    #     Contexte :
    #     """
    # )
    # st.markdown("---")

    st.markdown(
        """
	Le cancer du rein est l'une des tumeurs malignes les plus courantes chez les adultes dans le monde. Heureusement, la plupart des tumeurs rénales sont découvertes tôt alors qu'elles sont encore localisées et opérables. 

    Les tumeurs rénales sont connues pour leur apparence remarquable en imagerie par tomodensitométrie (TDM), ce qui a permis aux radiologues et aux chirurgiens d'effectuer d'importants travaux pour étudier la relation entre la taille, la forme et l'apparence de la tumeur et ses perspectives de traitement. C'est cependant un travail laborieux qui repose sur des évaluations souvent subjectives et imprécises.

    La segmentation automatique des tumeurs rénales et de l'anatomie environnante pourrait-être un outil prometteur pour remédier à ces limitations. 
        """
    )
    st.markdown("---")

    st.markdown(
        """
        🎯 Objectif :
	    """
    )
    st.markdown("---")

    st.markdown(
        """
    Produire un modèle proposant une segmentation aussi précise que possible des reins, des tumeurs et des kystes.
	    """
    )
