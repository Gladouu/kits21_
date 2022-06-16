"""
Created on Sat May 28 14:02:35 2022
Last Modified on Jeudi 16 Juin 2022

@author: Vannetzel Damien, Foyou Samuel, Valenzuela Gladis
"""

import streamlit as st

# pour model 2D
import functions.fonctions_images as f_i
import functions.fonctions_vie as f_v

title = "Modélisation"
sidebar_name = "Modélisation"


def run():
    st.title(title)