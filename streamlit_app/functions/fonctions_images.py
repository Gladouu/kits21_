# -*- coding: utf-8 -*-
"""
Created on Sat May 28 14:02:35 2022

@author: xbotobx
"""



from PIL import Image
import streamlit as st

__version__ = "1.1.a_beta"


def version():
    st.text(__version__)
    return __version__

def affiche_image_st(_path_image):
    """_summary_

    Args:
        _path_image (str): chemin de l'image à charger

    Returns:
        _type_: _description_
    """
    im = Image.open(_path_image)
    print(type(im))
    st.image(im)
    return im


    
    
    
if __name__ == "__main__":
    print(version())
    