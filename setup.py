from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    req = []
    with open(file_path) as file_obj:
        req = file_obj.readlines()
        req = [req.replace("\n","") for req in req]
        if HYPEN_E_DOT in req:
            req.remove(HYPEN_E_DOT)
        return req
setup(
    name = 'JewelJuggernaut',
    version = '0.0.1',
    author = 'Mohd Uwaish',
    install_requires = get_requirements('requirements.txt'),
    packages = find_packages()

)