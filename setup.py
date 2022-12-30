from setuptools import setup, find_packages
from typing import List


# Declaring variables for setup functions
PROJECT_NAME = "src"
VERSION = "0.0.1"
AUTHOR = "Chirag Tagadiya"
AUTHOR_EMAIL = "cr.tagadiya@gmail.com"
DESRCIPTION = "finance consumer complaint classification"

REQUIREMENT_FILE_NAME = "requirements.txt"

HYPHEN_E_DOT = "-e ."


def get_all_requirement_list() -> List[str]:
    """
    Get All Requirements from reading from requirements.txt
    :return: List of requirements
    :rtype: List[str]
    """

    with open(REQUIREMENT_FILE_NAME) as req_files:
        requirement_list: List[str] = []
        all_raw_requirements = req_files.readlines()
        requirement_list = [req_name.replace("\n", "") for req_name in all_raw_requirements]

        if HYPHEN_E_DOT in requirement_list:
            requirement_list.remove(HYPHEN_E_DOT)

        return requirement_list


setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESRCIPTION,
    packages=find_packages(),
    install_requires=get_all_requirement_list()
)