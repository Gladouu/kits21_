"""
Created on Sat May 28 14:02:35 2022
Last Modified on Jeudi 16 Juin 2022

@author: Vannetzel Damien, Foyou Samuel, Valenzuela Gladis
"""

import streamlit as st


title = "Kidney Tumor Segmentation Challenge"
sidebar_name = "Introduction"
subtitle = " 🩻 Contexte"


def run():
    st.title(title)
    st.markdown("---")
    st.image("images/z_gif.gif")

    st.title(subtitle)
    st.markdown("---")

    st.markdown(
        """
        Le cancer du rein, est l'une des tumeurs malignes les plus courantes chez les adultes dans le monde. Près de 70 à 80% des tumeurs du rein sont des tumeurs malignes. Heureusement, la plupart des tumeurs rénales sont découvertes tôt, de façon fortuite, dans 70 à 80% des cas, lorsqu'elles sont encore localisées et opérables. 

        L’étude de la relation entre la taille, la forme et l'apparence de la tumeur et ses perspectives de traitement représente un travail laborieux reposant sur des appréciations souvent subjectives et imprécises. 

        En effet, le diagnostic reposant sur de l’analyse d’image est soumis à une grande variabilité. Il existe une variabilité interpersonnelle (deux personnes peuvent ne pas avoir le même diagnostic) et intrapersonnelle (une même personne peut ne pas être constante dans sa façon d’établir un diagnostic, prendre des mesures ou bien encore classer des éléments). 

        La segmentation automatique des tumeurs rénales pourrait donc être un des outils pour remédier à ces limitations : les évaluations basées sur la segmentation sont objectives et nécessairement bien définies. L'automatisation, réduisant ainsi l’effort, permettrait au radiologue de se concentrer sur des zones de plus fort intérêt clinique.

        """
    )
    st.markdown("---")

    st.title("🎯 Objectif")
    st.markdown("---")

    st.markdown(
        """
    Produire un modèle proposant une segmentation automatique aussi précise que possible des reins, des tumeurs et des kystes.
	    """
    )
    st.markdown("---")
