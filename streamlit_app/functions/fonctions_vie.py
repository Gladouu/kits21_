# -*- coding: utf-8 -*-
"""

Created on Sat May 29 14:02:35 2022

@author: xbotobx


"""

#import
from datetime import datetime
from pathlib import Path
import functions.fonctions_images as f_i

# CONSTANTES
__version__ = "1.1.a_beta"

CE_PATH = Path(__file__).parent.parent
CE_PATH = str(CE_PATH) + "\\" + "Streamlit_kidneySpy-rz"


# fonctions
def __version__():
    return __version__

def current_path(_add = "", _verbose=False):
    """_summary_

    Args:
        _add (str, optional): ajouter un sous répertoire. Defaults to "".
        _verbose (bool, optional): affiche les paths. Defaults to False.

    Returns:
        _CE_PATH (str)_: retoune le chemin du dossier courant
    """
    CE_PATH = Path(__file__).parent.parent
    if _verbose: print('current_PATH :', CE_PATH)

    
    if _add != "":
        CE_PATH = str(CE_PATH) + "\\" + "Streamlit_kidneySpy-rz"
        if _verbose: print('CE_PATH :', CE_PATH)
    

    if _verbose: print('return :', CE_PATH)
    return CE_PATH

# definition du chronomètre

def go_chrono():
    """
    # exemple d'utilisation
    t0 = go_chrono()
    end_chro(t0)
    """
    t0 = datetime.now()
    t0tx = t0.isoformat(timespec='seconds')
    print(f"deb : {t0tx}")
    return t0


def end_chro(t0):
    """
    # exemple d'utilisation
    t0 = go_chrono()
    end_chro(t0)
    """
    t1 = datetime.now()
    t1tx = t1.isoformat(timespec='seconds')
    print(f"fin : {t1tx} ___ {t1-t0}")





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    print(__version__())