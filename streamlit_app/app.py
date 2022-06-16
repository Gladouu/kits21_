"""
Created on Sat May 28 14:02:35 2022
Last Modified on Jeudi 16 Juin 2022

@author: Vannetzel Damien, Foyou Samuel, Valenzuela Gladis
"""

from collections import OrderedDict
import streamlit as st

# TODO : change TITLE, TEAM_MEMBERS and PROMOTION values in config.py.
import config

# TODO : you can (and should) rename and add tabs in the ./tabs folder, and import them here.
from tabs import demo_1_about, demo_2_dataset, demo_3_preprocessng, demo_4_modelisation, demo_5_applicatif


st.set_page_config(
    page_title=config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)

with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (demo_1_about.sidebar_name, demo_1_about),
        (demo_2_dataset.sidebar_name, demo_2_dataset),
        (demo_3_preprocessng.sidebar_name, demo_3_preprocessng),
        (demo_4_modelisation.sidebar_name, demo_4_modelisation),
        (demo_5_applicatif.sidebar_name, demo_5_applicatif),

    ]
)


def run():
    st.sidebar.image(
        "https://dst-studio-template.s3.eu-west-3.amazonaws.com/logo-datascientest.png",
        width=200,
    )
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    st.sidebar.markdown(f"## {config.STAFF}")

    tab = TABS[tab_name]

    tab.run()


if __name__ == "__main__":
    run()
