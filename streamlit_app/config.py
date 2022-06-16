"""

Config file for Streamlit App

"""

from member import Member


TITLE = "Kidneyppyng App"


TEAM_MEMBERS = [
    Member(
        name="Gladis Valenzuela",
        linkedin_url="https://www.linkedin.com/in/gladis-valenzuela/",
        github_url="https://github.com/Gladouu",
    ),
    Member(
        name="Samuel Foyou",
        linkedin_url="https://www.linkedin.com/in/charlessuttonprofile/",
        github_url="https://github.com/charlessutton",
    ),
    Member("Damien Vannetzel",
           linkedin_url="https://www.linkedin.com/in/damien-vann",
           github_url="https://github.com/charlessutton",
           ),
]

PROMOTION = "Promotion Continue Data Scientist - September 2021"

STAFF = 'Thomas BOEHLER, Raphaël KASSEL'
