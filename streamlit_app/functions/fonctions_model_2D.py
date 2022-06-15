import os


from datetime import datetime

import numpy as np


import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.callbacks import LearningRateScheduler
import segmentation_models_3D as sm

import nibabel as nib # pour chargement de nifti
import cv2

"""
les fonctions utilisées pour l'applicatif du modèle 2D

"""

def prerequis_models_dice_2D():
    # from tensorflow.keras.callbacks import LearningRateScheduler
    # import pip
    # ! pip install segmentation-models-3D
    # import segmentation_models_3D as sm
    
    global wt0
    global wt1
    global wt2
    global wt3
    global dice_loss
    global focal_loss
    global total_loss
    global metrics_dice
    global schedule
    global lr_scheduler 
    global optimizer
    
    #Define loss, metrics and optimizer to be used for training
    wt0, wt1, wt2, wt3 = 0.25, 0.25, 0.25,  0.25
    dice_loss = sm.losses.DiceLoss() #class_weights=np.array([wt0, wt1, wt2, wt3])) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    metrics_dice = ['accuracy', 
            sm.metrics.IOUScore(threshold=0.5),
            ]
    schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001,
                                                        decay_steps=100000, 
                                                        decay_rate=0.96, 
                                                        staircase=True,
                                                        )
    lr_scheduler = LearningRateScheduler(schedule, verbose=1)  
    optimizer = tf.keras.optimizers.Adam(lr_scheduler)
    return 1
    

def lister_tous_les_modeles(_path_directory_a_fouiller, _extension='h5', _exclure_debute_par='model_test', _inactive_debute_par='NON_'):
    """


    """
    dico_path_models = {}
    for path, subdirs, files in os.walk(_path_directory_a_fouiller):
        for name in files:
            if (os.path.join(path, name).split(".")[-1] == _extension) \
            & (name[:len(_exclure_debute_par)]!=_exclure_debute_par)\
            & (name[:len(_inactive_debute_par)]!=_inactive_debute_par):
                # print(os.path.join(path, name))
                dico_path_models[name] = os.path.join(path, name)
    print(len(dico_path_models), "modèles trouvés")
    return dico_path_models

# def charger_les_models(dico_path_models):
#     models_dico = {}
#     for ce_model in dico_path_models:
#     print('ce_model :', ce_model)
#     if (len(ce_model.split('_dice')) > 1) | (ce_model[:4] == 'dice'):  # c'est un model dice
#         #  chargement model dice
#         models_dico[ce_model] = tf.keras.models.load_model(dico_path_models[ce_model],
#                                                             custom_objects={'dice_loss_plus_1focal_loss': total_loss,
#                                                                             'iou_score':sm.metrics.IOUScore(threshold=0.5)
#                                                                             }
#                                                             )
#     else:
#         models_dico[ce_model] = tf.keras.models.load_model(dico_path_models[ce_model])
#     return(models_dico)

def charger_les_models(dico_best_models_path_by_class):
  """
  "__version 2.3.d__"
  reçoit un dictionnaire contenant : 
  {key = classe_attrubuée au modèle : value = modèle chargé} 
  key est un entier int correspondant au numéro de la classe

  charge tous les modèles
  retourne un dictionnaire : key = classe(int), value = modèle chargé
  """
  prerequis_models_dice_2D()
  models_dico = {}
  for cette_classe, ce_model_path in dico_best_models_path_by_class.items():
    print("classe :", cette_classe)
    print('----------\nce_model_path :', ce_model_path)
    print('classe affectée :', cette_classe)
    ce_model = ce_model_path.split(r'/')[-1]
    print('nom du model :', ce_model)
    if (len(ce_model.split('_dice')) > 1) | (ce_model[:4] == 'dice'):  # c'est un model dice
      #  chargement spécifique des modèles avec dice
      models_dico[cette_classe] = tf.keras.models.load_model(ce_model_path,  # dico_path_models[ce_model],
                                                          custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                                                          'iou_score':sm.metrics.IOUScore(threshold=0.5)
                                                                          }
                                                          )
    else:
        models_dico[cette_classe] = tf.keras.models.load_model(ce_model_path,  # dico_path_models[ce_model]
                                                           )
    print(models_dico[cette_classe], "\n")
  return(models_dico)



def chargement_image_3D(_chemin):
    # décompression et chargement des données
    loaded = nib.load(_chemin)
    # extraction des données images 3D
    data_3D = loaded.get_fdata()
    return data_3D


def scanne_niveaux(image_3D, dico_models_charged, nbre_classes=4, axe=0, _resize=(512,512), _filtre_round=True):
    """
    "__version 4.2.h__"
    scanne tous les niveaux

    image_3D : la matrice 3D des images 
    dico_models_charged : dictionnaire contenant le model deja chargé à utiliser pour chaque classe :
    dico_models_charged = {0:model_hors_rein_charged, 1:model_rein_charged, 2:model_tumeur_charged, 3:model_kyste_charged}
    nbre_classes=4 : nombre de classe à évaluer
    axe=0 : avec d'exploration

    retourne un dictionnaire contenant les résultats de l'exploration:
    dico_rapport = {"niveaux_reins":niveaux_reins, "niveaux_tumeurs":niveaux_tumeurs, "niveaux_kystes":niveaux_kystes, 
                    "image_rein":image_rein, "image_tumeur:image_tumeur", "image_kyste:image_kyste",
                    "ref_image_rein":ref_image_rein, "ref_image_tumeur:ref_image_tumeur", "ref_image_kyste:ref_image_kyste",
                    }
    # les listes des niveaux contenant chaque classes
    niveaux_reins, niveaux_tumeurs, niveaux_kystes, image_rein, image_tumeur, image_kyste
    niveaux_reins : liste des niveaux dans lesquel du rein a été détecté
    niveaux_tumeurs : liste des niveaux dans lesquel de la tumeur a été détecté
    niveaux_kystes : liste des niveaux dans lesquel du kyste a été détecté

    # 1 image par classe ayant la plus grande surface pour cette classe
    image_rein : image png de l'image ayant la plus grande surface de rein
    image_tumeur : image png de l'image ayant la plus grande surface de tumeur
    image_kyste :  image png de l'image ayant la plus grande surface de kyste

    # la reference du niveau de ces images
    "ref_image_rein" : numéro du niveau correspondant à l'image sélectionnée pour le rein
    "ref_image_tumeur :numéro du niveau correspondant à l'image sélectionnée pour la tumeur
    "ref_image_kyste : numéro du niveau correspondant à l'image sélectionnée pour le kyste

    ces images sont initialisées avec des 0
    si la classe n'est pas rencontrée, l'image de sortie sera noir

    """
    print("__version 4.2.h__")
    # initilisations
    dico_rapport = {}
    niveaux_reins = []
    niveaux_tumeurs= []
    niveaux_kystes = []
    image_rein = np.zeros(_resize)
    ref_image_rein = -1
    image_tumeur = np.zeros(_resize)
    ref_image_tumeur = -1
    image_kyste = np.zeros(_resize)
    ref_image_kyste = -1

    # exploration niveau par niveau
    NB_CLASSES = 4
    v_si_vrai = 1
    v_si_faux = 0

    print("image_3D.shape", image_3D.shape)
    print("nombre de niveaux :", image_3D.shape[0])

    for ce_niveau in range(image_3D.shape[0]):
        print(f"niveau {ce_niveau} / {image_3D.shape[0]}")

        # extrait image
        # recupération de la matrice de la coupe
        img_coupe = image_3D[ce_niveau, :, :]  # img_data
        print("img_coupe.shape", img_coupe.shape)

        # convertir img_coupe en png
        cv2.imwrite('tmp.png', img_coupe)
        img_coupe_png = cv2.imread('tmp.png', 0)

        # standardise pixels
        img_std = img_coupe_png / 255.
        print("img_std.shape", img_std.shape)
        # resize
        im_x = cv2.resize(img_std, (512,512))
        print("im_x.shape", im_x.shape)

        # initialisation des masques : 
        print("dimensions image :", (image_3D.shape[1], image_3D.shape[2]))
        # mask = [np.zeros((image_3D.shape[1], image_3D.shape.shape[2])) for n in range(NB_CLASSES)]
        ########################### classe 1 ##########################################
        # predit rein : classe 1
        cette_classe = 1
        y_pred_rein = dico_models_charged[cette_classe].predict(im_x.reshape(-1,512,512,1))
        if _filtre_round:
            y_pred_rein_round = np.round(y_pred_rein, 0)
        im_y = y_pred_rein
        # argmax
        y_argmax_rein = tf.argmax(im_y, axis=-1).numpy().reshape((im_y.shape[1], im_y.shape[2]))
        # extraire les pixels == classe
        mask_rein = np.uint8(np.where(y_argmax_rein == cette_classe, v_si_vrai, v_si_faux))
        print("masque classe", cette_classe)
        # print(mask_rein[cette_classe])
        somme_rein = mask_rein.sum()
        print("somme_rein", somme_rein)
        somme_image_rein = image_rein.sum()

        # si somme_rein > 0 : on mémorise le niveau 
        if somme_rein > 0:
            print(f"         => DANS REIN NORMAL : niveau :{ce_niveau} surface = {somme_rein}" )
            niveaux_reins.append(ce_niveau)

        # on compare somme_rein avec somme_image_rein
        # si somme_rein > somme_image_rein: on mémorise le mask et le niveau
        if somme_rein > somme_image_rein:
            image_rein = mask_rein
            ref_image_rein = ce_niveau
            print(f"         => nouvelle surface rein max découverte : ancienne :{somme_image_rein} actuelle = {somme_rein}" )
    

        ########################### classe 2 ##########################################
        # predit tumeur : classe 2
        cette_classe = 2
        y_pred_tumeur = dico_models_charged[cette_classe].predict(im_x.reshape(-1,512,512,1))
        if _filtre_round:
            y_pred_tumeur_round = np.round(y_pred_tumeur, 0)
        im_y = y_pred_tumeur
        # argmax
        y_argmax_tumeur = tf.argmax(im_y, axis=-1).numpy().reshape((im_y.shape[1], im_y.shape[2]))
        # extraire les pixels == classe
        mask_tumeur = np.uint8(np.where(y_argmax_tumeur == cette_classe, v_si_vrai, v_si_faux))
        print("masque classe", cette_classe)
        # print(mask_tumeur[cette_classe])
        somme_tumeur = mask_tumeur.sum()
        print("somme_tumeur", somme_tumeur)
        somme_image_tumeur = image_tumeur.sum()

        # si somme_tumeur > 0 : on mémorise le niveau 
        if somme_tumeur > 0:
            print(f"         => DANS tumeur : niveau :{ce_niveau} surface = {somme_tumeur}" )
            niveaux_tumeurs.append(ce_niveau)

        # on compare somme_tumeur avec somme_image_tumeur
        # si somme_tumeur > somme_image_tumeur: on mémorise le mask et le niveau
        if somme_tumeur > somme_image_tumeur:
            image_tumeur = mask_tumeur
            ref_image_tumeur = ce_niveau
            print(f"         => nouvelle surface tumeur max découverte : ancienne :{somme_image_tumeur} actuelle = {somme_tumeur}" )

        ########################### classe 3 ##########################################
        # predit kyste : classe 3
        cette_classe = 3
        y_pred_kyste = dico_models_charged[cette_classe].predict(im_x.reshape(-1,512,512,1))
        if _filtre_round:
            y_pred_kyste_round = np.round(y_pred_kyste, 0)
        im_y = y_pred_kyste
        # argmax
        y_argmax_kyste = tf.argmax(im_y, axis=-1).numpy().reshape((im_y.shape[1], im_y.shape[2]))
        # extraire les pixels == classe
        mask_kyste = np.uint8(np.where(y_argmax_kyste == cette_classe, v_si_vrai, v_si_faux))
        print("masque classe", cette_classe)
        # print(mask_kyste[cette_classe])
        somme_kyste = mask_kyste.sum()
        print("somme_kyste", somme_kyste)
        somme_image_kyste = image_kyste.sum()

        # si somme_kyste > 0 : on mémorise le niveau 
        if somme_kyste > 0:
            print(f"         => DANS kyste : niveau :{ce_niveau} surface = {somme_kyste}" )
            niveaux_kystes.append(ce_niveau)

        # on compare somme_kyste avec somme_image_kyste
        # si somme_kyste > somme_image_kyste: on mémorise le mask et le niveau
        if somme_kyste > somme_image_kyste:
            image_kyste = mask_kyste
            ref_image_kyste = ce_niveau
            print(f"         => nouvelle surface kyste max découverte : ancienne :{somme_image_kyste} actuelle = {somme_kyste}" )


    dico_rapport["niveaux_reins"] = niveaux_reins
    dico_rapport["niveaux_tumeurs"] = niveaux_tumeurs
    dico_rapport["niveaux_kystes"] = niveaux_kystes
    dico_rapport["image_rein"] = image_rein
    dico_rapport["image_tumeur"] = image_tumeur
    dico_rapport["image_kyste"] = image_kyste
    dico_rapport["ref_image_rein"] = ref_image_rein
    dico_rapport["ref_image_tumeur"] = ref_image_tumeur
    dico_rapport["ref_image_kyste"] = ref_image_kyste
    

    return dico_rapport


def analyse_cas_tdm(chemin_nifti, nom_du_cas, dico_models_charged, chemin_segmentation=None, _resize=(512,512)):
    """
    dico_models_charged : dictionnaire contenant le model deja chargé à utiliser pour chaque classe la clé étant le numéro de la classe :
    dico_models_charged = {0:model_hors_rein_charged, 1:model_rein_charged, 2:model_tumeur_charged, 3:model_kyste_charged}


    retourne dico_rapport voir scanne_niveaux pour description

    """
    
    print(f"Analyse du scanner de {nom_du_cas} :")
    # extraction
    HAVE_SEGMENTATION = False
    print()
    print("chargement des images 3D :")
    img_3D = chargement_image_3D(chemin_nifti)
    if chemin_segmentation:
        print("chargement des images 3D de segmentation :")
        HAVE_SEGMENTATION = True
        img_segm_3D = chargement_image_3D(chemin_segmentation)

    
    print("segmentation :", HAVE_SEGMENTATION)
    print(f"nombre de plans de coupes : {img_3D.shape[0]} ")
    print(f"taille des images : {img_3D.shape[1:3]} ")

    print("exploration axe 0")
    dico_rapport  = scanne_niveaux(img_3D, dico_models_charged, nbre_classes=4, axe=0)
    return dico_rapport
    # vérifier qu'il n'y a pas deux reins
    # afficher 4*4 images pour chaque classe

    # afficher images avec curseur exploration ipwidget et bouton prise de photo


def affiche_images(_cas, _niveau, _chemin_nifti_image=None, _chemin_nifti_segmentation=None, _one_hot=True, _resize=(512,512)):
    "__version 1.8.n__"
    nb_col = 0
    # charge ce cas
    if _chemin_nifti_image:
        nb_col += 1
        img_3D = chargement_image_3D(_chemin_nifti_image)
        print("img_3D.shape", img_3D.shape)
        # charge cette image 2D
        img_2D = img_3D[_niveau, :, : ]
        img_2D = np.resize(img_2D, _resize)


    if _chemin_nifti_segmentation:
        nb_col += 1
        img_segm_3D = chargement_image_3D(_chemin_nifti_segmentation)
        print("img_segm_3D.shape", img_segm_3D.shape)
        # charge cette segmentation 2D
        img_segm_2D = img_segm_3D[_niveau, :, : ]
        img_segm_2D = np.resize(img_segm_2D, _resize)

        if _one_hot:
            nb_classes = 4
            nb_col += nb_classes
            img_segm_2D_mask_0 = np.uint8(np.where(img_segm_2D == 0, 1, 0))
            img_segm_2D_mask_1 = np.uint8(np.where(img_segm_2D == 1, 1, 0))
            img_segm_2D_mask_2 = np.uint8(np.where(img_segm_2D == 2, 1, 0))
            img_segm_2D_mask_3 = np.uint8(np.where(img_segm_2D == 3, 1, 0))

    plt.figure(figsize=(15,20))
    i = 0
    if _chemin_nifti_image:
        i += 1
        plt.subplot(1,nb_col, i)
        plt.title("image\n cas: " + str(_cas) + " \nniveau : " + str(_niveau))
        plt.imshow(img_2D, cmap='gray')

    if _chemin_nifti_segmentation:
        i += 1
        plt.subplot(1,nb_col, i)
        plt.title("segmentation réelle\n cas: " + str(_cas) + "\nniveau : " + str(_niveau))
        plt.colorbar()
        plt.imshow(img_segm_2D)

    if _one_hot:
        i += 1
        plt.subplot(1,nb_col, i)
        plt.title("hors_rein réel\n cas:" + str(_cas) + " \nniveau : " + str(_niveau))
        plt.imshow(img_segm_2D_mask_0, cmap='gray')

        i += 1
        plt.subplot(1,nb_col, i)
        plt.title("rein réel")
        plt.imshow(img_segm_2D_mask_1, cmap='gray')

        i += 1
        plt.subplot(1,nb_col, i)
        plt.title("tumeur réel")
        plt.imshow(img_segm_2D_mask_2, cmap='gray')

        i += 1
        plt.subplot(1,nb_col, i)
        plt.title("kyste réel")
        plt.imshow(img_segm_2D_mask_3, cmap='gray')


    plt.show()
    


    # charge cette image 2D

    # charge cette segmentation 2D

    # affiche image niveau
    # affiche segmentation niveau
    # affiche par classe niveau
    


def run_analyse_tdm(dico_models_charged, path_nii_imaging, numero_cas, nom_cas, path_nii_segmentation=None):
    """
    "__verion v2.1.h"
    _summary_

    Args:
        dico_models_charged (_type_): _description_
        path_nii_imaging (_type_): _description_
    """



    dico_rapport = analyse_cas_tdm(path_nii_imaging, nom_cas, 
                                   dico_models_charged, _chemin_segmentation=path_nii_segmentation)

    ce_cas = numero_cas
    # dico_rapport["niveaux_reins"] = niveaux_reins
    # dico_rapport["niveaux_tumeurs"] = niveaux_tumeurs
    # dico_rapport["niveaux_kystes"] = niveaux_kystes
    # dico_rapport["image_rein"] = image_rein
    # dico_rapport["image_tumeur"] = image_tumeur
    # dico_rapport["image_kyste"] = image_kyste
    # dico_rapport["ref_image_rein"] = ref_image_rein
    # dico_rapport["ref_image_tumeur"] = ref_image_tumeur
    # dico_rapport["ref_image_kyste"] = ref_image_kyste

    plt.figure(figsize=(10,15))
    plt.subplot(1,3,1)
    txt = "image rein (surface max)\nn=" + str(dico_rapport["ref_image_rein"]) + " " + nom_cas
    if dico_rapport["ref_image_rein"] == -1:
        txt = "image rein (surface max)\npas de rein détecté \n" + nom_cas
    plt.title(txt)
    plt.imshow(dico_rapport["image_rein"], cmap='gray')

    plt.subplot(1,3,2)
    txt = "image tumeur (surface max)\nn=" + str(dico_rapport["ref_image_tumeur"]) + " " + nom_cas
    if dico_rapport["ref_image_tumeur"] == -1:
        txt = "image rein (surface max)\npas de tumeur détectée \n" + nom_cas
    plt.title(txt)
    plt.imshow(dico_rapport["image_tumeur"], cmap='gray')

    plt.subplot(1,3,3)
    txt = "image kyste (surface max)\nn=" + str(dico_rapport["ref_image_kyste"]) + " " + nom_cas
    if dico_rapport["ref_image_kyste"] == -1:
        txt = "image rein (surface max)\npas de kyste détecté \n" + nom_cas
    plt.title(txt)
    plt.imshow(dico_rapport["image_kyste"], cmap='gray')

    plt.show()

    # chemin_image = path_nii_imaging[ce_cas]

    affiche_images( _cas=ce_cas, _niveau= dico_rapport["ref_image_rein"], 
                   _chemin_nifti_image=path_nii_imaging[ce_cas],
                   _chemin_nifti_segmentation=path_nii_segmentation[ce_cas],
                   _one_hot=True)
    affiche_images( _cas=ce_cas, _niveau= dico_rapport["ref_image_tumeur"], 
                   _chemin_nifti_image=path_nii_imaging[ce_cas], 
                   _chemin_nifti_segmentation=path_nii_segmentation[ce_cas],
                   _one_hot=True)
    affiche_images( _cas=ce_cas, _niveau= dico_rapport["ref_image_kyste"], 
                   _chemin_nifti_image=path_nii_imaging[ce_cas], 
                   _chemin_nifti_segmentation=path_nii_segmentation[ce_cas], 
                   _one_hot=True)
    