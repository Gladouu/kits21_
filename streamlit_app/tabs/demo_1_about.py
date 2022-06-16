"""

"""

import streamlit as st


title = "Kidney Tumor Segmentation Challenge"
sidebar_name = "Introduction"
subtitle = "Enjeux & attente"


def run():
    st.title(title)
    st.markdown("---")
    st.image("images/Coupe_anatomie.jpg")

    st.markdown("---")
    st.title(subtitle)

    st.markdown("---")
    st.markdown(
        """
        Contexte : 
        """
    )
    st.markdown("---")

    st.markdown(
        """
	Le cancer du rein est l'une des tumeurs malignes les plus courantes chez les adultes dans le monde, et on pense que son incidence est en augmentation. Heureusement, la plupart des tumeurs rénales sont découvertes tôt alors qu'elles sont encore localisées et opérables. Cependant, il existe des questions importantes concernant la prise en charge des tumeurs rénales localisées qui restent sans réponse, et le cancer du rein métastatique reste presque uniformément mortel.

    Les tumeurs rénales sont connues pour leur apparence remarquable en imagerie par tomodensitométrie (TDM), ce qui a permis aux radiologues et aux chirurgiens d'effectuer d'importants travaux pour étudier la relation entre la taille, la forme et l'apparence de la tumeur et ses perspectives de traitement. C'est cependant un travail laborieux qui repose sur des évaluations souvent subjectives et imprécises.

    La segmentation automatique des tumeurs rénales et de l'anatomie environnante pourrait-être un outil prometteur pour remédier à ces limitations : les évaluations basées sur la segmentation sont objectives et l'automatisation élimine tout effort.
    """
    )
    st.markdown("---")

    st.markdown(
        """
Enjeu économique : 
"""
    )
    st.markdown("---")

    st.markdown(
        """
	Le secteur de la Santé est en crise depuis de nombreuses années déjà. Les contraintes de temps, de ressources (humaines mais aussi économiques) et de moyens sont réelles. 

Les demandes d’examens ne cessent d’augmenter et les délais pour y répondre également. Un délai d’examen trop important peut, suivant le diagnostic, diminuer les chances de survie du patient. De ce fait, plus tôt le diagnostic est posé plus il y a de chances pour que le patient soit pris en charge de façon optimale, permettant ainsi d’éviter des complications secondaires lourdes et coûteuses.

De plus, l’automatisation de certaines tâches permet de faire gagner un temps précieux aux cliniciens, voire même de disposer d’informations supplémentaires qu’ils ne prendraient pas le temps de générer en routine car trop chronophage.

Les tumeurs rénales sont connues pour leur apparition remarquable dans l'imagerie par tomodensitométrie (TDM), ce qui a permis aux radiologues et aux chirurgiens d'étudier la relation entre la taille, la forme et l'apparence de la tumeur et ses perspectives de traitement. De ce fait, les données que proposerait ce projet pourraient aider à orienter vers le ou les meilleurs traitements en regard d’une situation. Hormis le fait d’éviter des examens, parfois invasifs, mais surtout non pertinents aux patients, cela permettrait donc aussi d’éviter les coûts engendrés par ces examens non pertinents. 

"""
    )
    st.markdown("---")

    st.markdown(
        """
Enjeu scientifique : 
"""
    )
    st.markdown("---")

    st.markdown(
        """
	Le cancer du rein, qui occupe le 6e rang des tumeurs solides malignes, est l'une des tumeurs malignes les plus courantes chez les adultes dans le monde. Près de 70 à 80% des tumeurs du rein sont des tumeurs malignes. Heureusement, la plupart des tumeurs rénales sont découvertes tôt, de façon fortuite, dans 70 à 80% des cas, lorsqu'elles sont encore localisées et opérables. 

L’étude de la relation entre la taille, la forme et l'apparence de la tumeur et ses perspectives de traitement représente un travail laborieux reposant sur des appréciations souvent subjectives et imprécises. 
En effet, le diagnostic reposant sur de l’analyse d’image est soumis à une grande variabilité. Il existe une variabilité interpersonnelle (deux personnes peuvent ne pas avoir le même diagnostic) et intrapersonnelle (une même personne peut ne pas être constante dans sa façon d’établir un diagnostic, prendre des mesures ou bien encore classer des éléments). 
La segmentation automatique des tumeurs rénales est donc un des outils prometteurs pour remédier à ces limitations : les évaluations basées sur la segmentation sont objectives et nécessairement bien définies. L'automatisation, réduisant ainsi l’effort, permettrait au radiologue de se concentrer sur des zones de plus fort intérêt clinique.

        """
    )
